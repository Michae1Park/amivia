from collections import Counter

import numpy as np

from data import PERSONAS, build_persona_weight_vectors


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
    (see synthetic_interactions/README.md). This exists purely to sanity-check the harness: if this
    doesn't clearly beat popularity/random, something in the eval pipeline is broken.
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
