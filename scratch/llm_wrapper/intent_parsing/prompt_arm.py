"""Arm A: prompting a hosted LLM for structured intent extraction.

Same API shape as eval/baselines.py's LLMRecommender (structured JSON output
via `output_config`, disk-memoised responses) so the two opt-in-LLM call
sites in this repo don't diverge for no reason.

Three prompt conditions, swept independently of model tier:
- bare       — schema only, no field descriptions, no examples
- described  — schema + the field `description`s already written into
                schema.INTENT_JSON_SCHEMA
- few_shot   — described + k worked examples drawn from data/train.jsonl

DEVIATION from the original README sketch: "structured outputs on/off" was
dropped as a fourth axis. Free-text-then-regex output is strictly worse and
not informative to compare — every production path would use structured
output — so model tier (a genuinely open question: does a bigger/slower
model generalise better to the OOD set) replaces it as the second axis.

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY=...

    python3 prompt_arm.py --split val --condition described
    python3 prompt_arm.py --split ood_eval --condition few_shot --model claude-opus-5
    python3 prompt_arm.py --split ood_eval --condition bare --n-shots 0   # all three conditions, one pass:
    for c in bare described few_shot; do python3 prompt_arm.py --split ood_eval --condition $c; done
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from schema import INTENT_JSON_SCHEMA

DATA_DIR = Path(__file__).resolve().parent / "data"

SYSTEM_BARE = (
    "Extract a travel query's constraints into the given JSON schema. "
    "Return only fields the query actually states; leave everything else empty/null."
)

SYSTEM_DESCRIBED = SYSTEM_BARE + (
    "\n\nField semantics:\n"
    "- required_tags: hard requirements ('beach getaway' -> beaches >= 4)\n"
    "- excluded_tags: negation ('no nightlife' -> nightlife max_rating <= 2)\n"
    "- budget_at_most: an upper bound on spend, not a preference for luxury\n"
    "- month_temp_at_least / month_temp_at_most: only set when a specific month AND "
    "a specific temperature are both stated\n"
    "Do not invent a constraint from vague descriptive language alone — "
    "'a nice relaxing trip' states no hard constraint at all."
)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_few_shot_block(n_shots: int, seed: int) -> str:
    train = load_jsonl(DATA_DIR / "train.jsonl")
    rng = random.Random(seed)
    shots = rng.sample(train, min(n_shots, len(train)))
    lines = ["Worked examples:"]
    for ex in shots:
        lines.append(f"Query: {ex['query']}\nJSON: {json.dumps(ex['intent'])}")
    return "\n\n".join(lines)


def system_prompt(condition: str, n_shots: int, seed: int) -> str:
    if condition == "bare":
        return SYSTEM_BARE
    if condition == "described":
        return SYSTEM_DESCRIBED
    if condition == "few_shot":
        return SYSTEM_DESCRIBED + "\n\n" + build_few_shot_block(n_shots, seed)
    raise ValueError(f"unknown condition: {condition}")


class PromptArm:
    def __init__(self, model: str, effort: str, condition: str, n_shots: int = 8,
                 seed: int = 42, cache_path: Path | None = None):
        self.model = model
        self.effort = effort
        self.condition = condition
        self.system = system_prompt(condition, n_shots, seed)
        self.cache_path = cache_path or (DATA_DIR / f"prompt_cache_{model}_{condition}.json")
        self._cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}

        import anthropic  # deferred: only required when actually calling the API
        self._client = anthropic.Anthropic()

    def _cache_key(self, query: str) -> str:
        return hashlib.sha256(f"{self.model}\x00{self.system}\x00{query}".encode()).hexdigest()

    def parse(self, query: str) -> dict:
        """Returns {"intent": dict, "raw_text": str, "latency_s": float, "cache_hit": bool}."""
        key = self._cache_key(query)
        if key in self._cache:
            return {**self._cache[key], "cache_hit": True}

        start = time.monotonic()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self.system,
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": INTENT_JSON_SCHEMA},
            },
            messages=[{"role": "user", "content": query}],
        )
        latency = time.monotonic() - start

        raw_text = "" if response.stop_reason == "refusal" else next(
            (b.text for b in response.content if b.type == "text"), "")
        try:
            intent = json.loads(raw_text)
        except json.JSONDecodeError:
            intent = {}

        u = response.usage
        result = {
            "intent": intent,
            "raw_text": raw_text,
            "latency_s": latency,
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
        }
        self._cache[key] = result
        return {**result, "cache_hit": False}

    def save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["val", "ood_eval"], default="ood_eval")
    parser.add_argument("--condition", choices=["bare", "described", "few_shot"], default="described")
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--n-shots", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    examples = load_jsonl(DATA_DIR / f"{args.split}.jsonl")
    arm = PromptArm(args.model, args.effort, args.condition, args.n_shots, args.seed)

    predictions = []
    for ex in examples:
        result = arm.parse(ex["query"])
        predictions.append({
            "query": ex["query"],
            "gold_intent": ex["intent"],
            "predicted_intent": result["intent"],
            "latency_s": result["latency_s"],
            "cache_hit": result["cache_hit"],
        })
    arm.save_cache()

    out_path = args.out or DATA_DIR / f"predictions_prompt_{args.model}_{args.condition}_{args.split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    n_cached = sum(p["cache_hit"] for p in predictions)
    print(f"{len(predictions)} parsed, {n_cached} from cache -> {out_path}")


if __name__ == "__main__":
    main()
