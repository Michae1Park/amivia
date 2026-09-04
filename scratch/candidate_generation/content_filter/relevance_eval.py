#!/usr/bin/env python3
"""Does content_filter's embedding retrieval actually surface tag-relevant candidates?

Ground truth here is deterministic, not behavioral: for each of filter_constraints's
LABELLED_QUERIES, "relevant" = every city in the catalog satisfying that query's
hand-labelled predicates (see constraints.py). This only validates the tag/budget/
temperature semantics those predicates capture — not free-text nuance the catalog has
no column for (e.g. "romantic", "family-friendly"). Session-click ground truth doesn't
help there either: synthetic_interactions' clicks are generated purely from the same
9 tag columns, so no signal in this repo grades nuance beyond them.

Random is the floor, included for the same reason eval/README.md's baselines exist —
without it there's nothing to confirm the metrics actually discriminate.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "filter_constraints"))
from constraints import LABELLED_QUERIES, satisfies_all  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eval"))
from metrics import hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402

from content_filter import DATA_PATH, EMBED_CACHE, MODEL_NAME, load_cities

K_VALUES = [5, 10, 20, 50]
METRICS = {
    "hit_rate": hit_rate_at_k,
    "recall": recall_at_k,
    "precision": precision_at_k,
    "ndcg": ndcg_at_k,
}


def get_embeddings(cities: list[dict]) -> np.ndarray:
    if EMBED_CACHE.exists():
        return np.load(EMBED_CACHE)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        [c["short_description"] for c in cities], normalize_embeddings=True, show_progress_bar=True
    )
    np.save(EMBED_CACHE, embeddings)
    return embeddings


def rank_all(query_vec: np.ndarray, cities: list[dict], embeddings: np.ndarray) -> list[str]:
    """Every city, ranked by cosine similarity — not just top-k, so recall@k is
    meaningful across the full K_VALUES sweep from one ranking."""
    scores = embeddings @ query_vec
    order = np.argsort(-scores)
    return [cities[i]["id"] for i in order]


def relevant_set(cities: list[dict], predicates: tuple) -> dict[str, int]:
    return {c["id"]: 1 for c in cities if satisfies_all(c, predicates)}


def mean_of(vals: list) -> float:
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Tag-grounded relevance eval for content_filter")
    parser.add_argument("--k-values", type=int, nargs="+", default=K_VALUES)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data" / "relevance_eval.csv")
    args = parser.parse_args()

    cities = load_cities(DATA_PATH)
    embeddings = get_embeddings(cities)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    query_vecs = model.encode([q for q, _ in LABELLED_QUERIES], normalize_embeddings=True)

    rng = np.random.default_rng(args.random_seed)
    all_ids = [c["id"] for c in cities]

    sizes = []
    rows = {"embedding": {k: [] for k in args.k_values}, "random": {k: [] for k in args.k_values}}

    for (query, predicates), qvec in zip(LABELLED_QUERIES, query_vecs):
        relevant = relevant_set(cities, predicates)
        sizes.append((query, len(relevant)))

        rankings = {"embedding": rank_all(qvec, cities, embeddings), "random": rng.permutation(all_ids).tolist()}
        for name, ranked in rankings.items():
            for k in args.k_values:
                rows[name][k].append({m: fn(ranked, relevant, k) for m, fn in METRICS.items()})

    print(f"\n{'=' * 74}\nRelevant-set size per query (out of {len(cities)} cities)\n{'=' * 74}")
    for query, n in sorted(sizes, key=lambda x: x[1]):
        print(f"  {n:>4}  {query}")

    print(f"\n{'=' * 74}\ncontent_filter vs. random, {len(LABELLED_QUERIES)} labelled queries\n{'=' * 74}")
    header = f"{'model':<10}{'k':>4}" + "".join(f"{m:>10}" for m in METRICS)
    print(header)
    out_rows = []
    for name in ("embedding", "random"):
        for k in args.k_values:
            means = {m: mean_of([r[m] for r in rows[name][k]]) for m in METRICS}
            print(f"{name:<10}{k:>4}" + "".join(f"{means[m]:>10.3f}" for m in METRICS))
            out_rows.append({"model": name, "k": k, **{m: round(means[m], 4) for m in METRICS}})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "k", *METRICS])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    main()
