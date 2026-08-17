import hashlib
import json
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from data import PERSONAS, TAGS, build_persona_weight_vectors


class RandomRecommender:
    def __init__(self, item_ids: list[str], seen: dict[str, set], rng: np.random.Generator):
        self.item_ids = item_ids
        self.seen = seen
        self.rng = rng

    def recommend(self, user_id: str, k: int) -> list[str]:
        candidates = [i for i in self.item_ids if i not in self.seen.get(user_id, set())]
        idx = self.rng.choice(len(candidates), size=min(k, len(candidates)), replace=False)
        return [candidates[i] for i in idx]


class PopularityRecommender:
    """Non-personalized: same ranked list for every user, by train click/save counts."""

    def __init__(self, train_rows: list[dict], item_ids: list[str], seen: dict[str, set]):
        counts = Counter(r["item_id"] for r in train_rows if r["event_type"] in ("click", "save"))
        self.ranked = sorted(item_ids, key=lambda i: -counts.get(i, 0))
        self.seen = seen

    def recommend(self, user_id: str, k: int) -> list[str]:
        seen = self.seen.get(user_id, set())
        return [i for i in self.ranked if i not in seen][:k]


class OraclePersonaRecommender:
    """Ranks by the same affinity formula generate.py used to produce clicks/saves, reconstructed
    from each user's ground-truth persona labels + mix_weight (idiosyncratic noise is not persisted,
    so this is an approximation, not a perfect ceiling).

    Not a real recommender — persona labels are eval-only ground truth, never a model input
    (see synthetic_interactions/README.md).

    It is an oracle of *latent preference*, not of the logged data, and it scores well below
    `popularity` here. That is exposure bias, not a broken harness: generate.py samples
    impressions by popularity rather than persona fit, so held-out positives track exposure
    almost perfectly (corr(train impressions, test positives) = 0.98). Ranking purely by
    affinity surfaces high-fit cities the user was never shown, which can never be scored as
    hits. Restricted to items the user actually saw, this ranker reaches ~0.41 precision
    against ~0.17 for chance — the affinity logic is sound, the metric is exposure-limited.

    Consequence: `popularity` is the bar to beat here, and every recommender that ranks by
    inferred taste (this one, `llm`, and later CF/two-tower) is penalised the same way.
    """

    BUDGET_MATCH_BONUS = 1.5

    def __init__(
        self,
        users: list[dict],
        item_ids: list[str],
        item_tags: np.ndarray,
        item_budgets: list[str],
        seen: dict[str, set],
    ):
        self.persona_vecs = build_persona_weight_vectors()
        self.users = {u["user_id"]: u for u in users}
        self.item_ids = item_ids
        self.item_tags = item_tags
        self.item_budgets = item_budgets
        self.seen = seen

    def recommend(self, user_id: str, k: int) -> list[str]:
        u = self.users[user_id]
        purity = float(u["mix_weight"])
        weights = (
            purity * self.persona_vecs[u["primary_persona"]]
            + (1 - purity) * self.persona_vecs[u["secondary_persona"]]
        )
        preferred_budgets = set(PERSONAS[u["primary_persona"]]["budgets"])
        bonus = np.array([self.BUDGET_MATCH_BONUS if b in preferred_budgets else 0.0 for b in self.item_budgets])
        scores = self.item_tags @ weights + bonus

        seen = self.seen.get(user_id, set())
        out = []
        for idx in np.argsort(-scores):
            item_id = self.item_ids[idx]
            if item_id not in seen:
                out.append(item_id)
                if len(out) == k:
                    break
        return out


class LLMRecommender:
    """Zero-shot LLM baseline: the entire 560-city catalog goes in the prompt, the user's
    train-set history goes in the user turn, and the model returns a ranked shortlist.

    No training, no embeddings, no interaction matrix — the point is to measure what the
    classic funnel has to beat at this catalog size. It stops being viable the moment the
    catalog outgrows the context window, which is exactly the constraint the funnel exists
    to solve (see docs/roadmap-to-service.md).

    Costs real money per user, so it is opt-in (`--llm`) and pairs with `--sample-users`.
    Three things keep the bill down:
      - the catalog block is a cached prompt prefix (identical across every user)
      - responses are memoised to disk, so re-runs are free
      - `prefetch()` warms the cache with one call, then fans out concurrently
    """

    INSTRUCTIONS = (
        "You are a travel recommender. You will be given a catalog of cities and one "
        "traveler's history of cities they clicked or saved.\n\n"
        "Infer their taste from that history, then return the catalog indices of the "
        "cities they are most likely to engage with next, best first.\n\n"
        "Rules:\n"
        "- Return only indices that appear in the catalog.\n"
        "- Never return a city listed in the traveler's history.\n"
        "- Rank by fit to the inferred taste, not by general fame.\n"
    )

    # Cap history in the prompt: heavy users have hundreds of events and the tail adds
    # tokens without adding signal.
    MAX_HISTORY = 40

    def __init__(
        self,
        catalog_rows: list[dict],
        train_rows: list[dict],
        seen: dict[str, set],
        model: str = "claude-opus-5",
        effort: str = "low",
        max_tokens: int = 8192,
        max_workers: int = 8,
        cache_path: Path | None = None,
    ):
        self.catalog_rows = catalog_rows
        self.item_ids = [r["id"] for r in catalog_rows]
        self.index_of = {r["id"]: i for i, r in enumerate(catalog_rows)}
        self.seen = seen
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_workers = max_workers

        self.history = self._build_history(train_rows)
        self.catalog_block = self._render_catalog()

        self.cache_path = cache_path
        self._cache = self._load_cache()
        self._lock = threading.Lock()
        self.usage = Counter()
        self.failures = 0

        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - environment guard
            raise RuntimeError(
                "The LLM baseline needs the Anthropic SDK: pip install anthropic"
            ) from e
        self._client = anthropic.Anthropic()

    # ---- prompt construction -------------------------------------------------

    def _build_history(self, train_rows: list[dict]) -> dict[str, list[tuple[str, str]]]:
        """user_id -> [(timestamp, item_id)] for clicks/saves only, oldest first.
        Impressions are excluded: they are exposure, not preference."""
        by_user = defaultdict(list)
        for r in train_rows:
            if r["event_type"] in ("click", "save"):
                by_user[r["user_id"]].append((r["timestamp"], r["item_id"], r["event_type"]))
        for rows in by_user.values():
            rows.sort()
        return by_user

    def _render_catalog(self) -> str:
        header = (
            "CATALOG — one city per line:\n"
            "index|city, country|budget|" + " ".join(t[:3] for t in TAGS) + " (each rated 1-5)\n"
        )
        lines = [
            f"{i}|{r['city']}, {r['country']}|{r['budget_level']}|"
            + " ".join(str(v) for v in r["tags"])
            for i, r in enumerate(self.catalog_rows)
        ]
        return header + "\n".join(lines)

    def _user_block(self, user_id: str, n_request: int) -> str:
        rows = self.history.get(user_id, [])[-self.MAX_HISTORY :]
        if rows:
            lines = [
                f"- {self.catalog_rows[self.index_of[item_id]]['city']}"
                f"{' (saved)' if event == 'save' else ''}"
                for _, item_id, event in rows
                if item_id in self.index_of
            ]
            history = "This traveler clicked or saved:\n" + "\n".join(lines)
        else:
            history = "This traveler has no history yet — recommend broadly appealing cities."
        return (
            f"{history}\n\n"
            f"Return the {n_request} best next cities as catalog indices, best first."
        )

    # ---- response cache ------------------------------------------------------

    def _load_cache(self) -> dict:
        if self.cache_path and self.cache_path.exists():
            return json.loads(self.cache_path.read_text())
        return {}

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache))

    def _cache_key(self, user_block: str) -> str:
        material = "\x00".join([self.model, self.effort, self.INSTRUCTIONS, user_block])
        return hashlib.sha256(material.encode()).hexdigest()

    # ---- API call ------------------------------------------------------------

    def _picks(self, user_id: str, n_request: int) -> list[int]:
        user_block = self._user_block(user_id, n_request)
        key = self._cache_key(user_block)
        with self._lock:
            if key in self._cache:
                return self._cache[key]

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {"type": "text", "text": self.INSTRUCTIONS},
                # Identical for every user — cache it so only the user turn is billed in full.
                {"type": "text", "text": self.catalog_block,
                 "cache_control": {"type": "ephemeral"}},
            ],
            output_config={
                "effort": self.effort,
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"picks": {"type": "array", "items": {"type": "integer"}}},
                        "required": ["picks"],
                        "additionalProperties": False,
                    },
                },
            },
            messages=[{"role": "user", "content": user_block}],
        )

        if response.stop_reason == "refusal":
            picks = []
        else:
            text = next((b.text for b in response.content if b.type == "text"), "")
            try:
                picks = json.loads(text)["picks"]
            except (json.JSONDecodeError, KeyError, TypeError):
                picks = []

        with self._lock:
            if not picks:
                self.failures += 1
            u = response.usage
            self.usage["input"] += u.input_tokens
            self.usage["output"] += u.output_tokens
            self.usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
            self.usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
            self._cache[key] = picks
        return picks

    # ---- recommender interface ----------------------------------------------

    def prefetch(self, user_ids: list[str], k: int) -> None:
        """Warm the prompt cache with one sequential call, then fan out.

        Concurrent requests sharing a prefix all miss the cache — nothing can read an
        entry another request is still writing — so the first call goes alone.
        """
        pending = [u for u in user_ids if self._cache_key(self._user_block(u, k + 10)) not in self._cache]
        if not pending:
            return
        self._picks(pending[0], k + 10)
        if len(pending) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                list(pool.map(lambda u: self._picks(u, k + 10), pending[1:]))
        self._save_cache()

    def recommend(self, user_id: str, k: int) -> list[str]:
        # Ask for a buffer: some picks get dropped as already-seen or duplicated.
        picks = self._picks(user_id, k + 10)
        seen = self.seen.get(user_id, set())
        out, used = [], set()
        for idx in picks:
            if not isinstance(idx, int) or not 0 <= idx < len(self.item_ids):
                continue
            item_id = self.item_ids[idx]
            if item_id in seen or item_id in used:
                continue
            used.add(item_id)
            out.append(item_id)
            if len(out) == k:
                break
        return out

    def usage_summary(self) -> str:
        u = self.usage
        billed = u["input"] + u["cache_write"] + u["cache_read"]
        cached_pct = 100 * u["cache_read"] / billed if billed else 0.0
        return (
            f"llm tokens: {u['input']:,} in / {u['output']:,} out / "
            f"{u['cache_write']:,} cache-write / {u['cache_read']:,} cache-read "
            f"({cached_pct:.0f}% of input served from cache); {self.failures} empty responses"
        )
