"""Builds the three JSONL splits both arms train/evaluate against:

- data/train.jsonl, data/val.jsonl — in-distribution (ID), weak-labelled at
  scale from synthetic_interactions/data/sessions.csv's templated query_text.
  Free (no hand-labelling), but structurally limited: every label comes from
  `schema.weak_label_from_session`, which can only ever produce a required
  tag and/or a budget cap — never negation, never a temperature threshold.
- data/ood_eval.jsonl — out-of-distribution, hand-labelled, from
  filter_constraints.LABELLED_QUERIES + ood_queries.ADDITIONAL_OOD_QUERIES.
  This is where negation, numeric thresholds, and "no constraint at all"
  first appear — patterns Arm B's fine-tune has literally never seen a
  labelled example of, by construction.

Run:
    python3 build_dataset.py
    python3 build_dataset.py --n-train 4000 --n-val 500 --seed 42
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

from schema import Intent, weak_label_from_session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "candidate_generation" / "filter_constraints"))
from constraints import LABELLED_QUERIES  # noqa: E402
from ood_queries import ADDITIONAL_OOD_QUERIES  # noqa: E402

SESSIONS_CSV = Path(__file__).resolve().parent.parent.parent / "synthetic_interactions" / "data" / "sessions.csv"
DATA_DIR = Path(__file__).resolve().parent / "data"


def load_sessions() -> list[dict]:
    with open(SESSIONS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_id_examples(sessions: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Sample n sessions and weak-label them. Sampling (not taking the first n)
    matters here: sessions.csv is written in timestamp order per user, so a
    naive head() would skew toward early users and miss the tail."""
    picked = rng.sample(sessions, min(n, len(sessions)))
    examples = []
    for row in picked:
        intent = weak_label_from_session(row["filter_budget_level"], row["filter_required_tag"])
        examples.append({
            "query": row["query_text"],
            "intent": intent.to_dict(),
            "source": "session_weak_label",
        })
    return examples


def build_ood_examples() -> list[dict]:
    examples = []
    for query, predicates in LABELLED_QUERIES:
        examples.append({
            "query": query,
            "intent": Intent.from_predicates(predicates).to_dict(),
            "source": "filter_constraints.LABELLED_QUERIES",
        })
    for query, predicates in ADDITIONAL_OOD_QUERIES:
        examples.append({
            "query": query,
            "intent": Intent.from_predicates(predicates).to_dict(),
            "source": "ood_queries.ADDITIONAL_OOD_QUERIES",
        })
    return examples


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sessions = load_sessions()
    rng.shuffle(sessions)

    train_sessions = sessions[: args.n_train]
    val_sessions = sessions[args.n_train : args.n_train + args.n_val]

    train = build_id_examples(train_sessions, args.n_train, rng)
    val = build_id_examples(val_sessions, args.n_val, rng)
    ood = build_ood_examples()

    write_jsonl(DATA_DIR / "train.jsonl", train)
    write_jsonl(DATA_DIR / "val.jsonl", val)
    write_jsonl(DATA_DIR / "ood_eval.jsonl", ood)

    n_train_constrained = sum(1 for r in train if r["intent"] != Intent().to_dict())
    n_ood_constrained = sum(1 for r in ood if r["intent"] != Intent().to_dict())
    print(f"train.jsonl:    {len(train)} examples ({n_train_constrained} with >=1 constraint, "
          f"{len(train) - n_train_constrained} empty intent)")
    print(f"val.jsonl:      {len(val)} examples")
    print(f"ood_eval.jsonl: {len(ood)} examples ({n_ood_constrained} with >=1 constraint, "
          f"{len(ood) - n_ood_constrained} empty intent)")
    print(f"\nwritten to {DATA_DIR}/")


if __name__ == "__main__":
    main()
