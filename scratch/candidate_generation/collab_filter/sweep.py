#!/usr/bin/env python3
"""Sweep the four CF models against the shared eval harness.

The headline metrics are not the whole result here. Because `synthetic_interactions`
samples impressions by popularity, a model can score well simply by relearning
"recommend what's popular" — so every row also reports the Spearman correlation
between that model's per-user ranking and the global popularity ranking. High
correlation plus a small NDCG gain over the `popularity` baseline means the model
learned exposure, not taste, and gets reported as such.
"""
import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "eval"
sys.path.insert(0, str(EVAL_DIR))

from baselines import PopularityRecommender, RandomRecommender  # noqa: E402
from data import (  # noqa: E402
    CATALOG_PATH,
    build_train_test_split,
    load_catalog,
    load_interactions,
    load_users,
    relevance_and_seen,
)
from run_eval import evaluate  # noqa: E402

from models import MODELS, build_matrix  # noqa: E402

FULL_GRID = {
    "item_knn": [{"n_neighbors": n} for n in (10, 50, 200, 560)],
    "svd": [{"n_factors": f} for f in (16, 32, 64, 128)],
    # alpha is the implicit-feedback knob the project README asks about: how hard an
    # observed interaction is treated as evidence, as opposed to its graded value.
    # reg goes to 100: the first pass topped out at 1.0 and made ALS look worse than
    # `popularity`, but the optimum sits far higher — cross-checking against `implicit`
    # is what caught it (see verify_vs_implicit.py).
    "als": [{"n_factors": f, "reg": r, "alpha": a}
            for f in (16, 32, 64, 128) for r in (0.01, 0.1, 1.0, 10.0, 100.0)
            for a in (1.0, 40.0)],
    "bpr": [{"n_factors": f, "reg": r} for f in (16, 32, 64, 128)
            for r in (0.001, 0.01, 0.1, 1.0)],
}

# Standard BPR samples from the *positions* of observed interactions and never reads
# their values, so binary and confidence weighting produce an identical model. Running
# both would print duplicate rows that look like independent evidence.
WEIGHTING_INSENSITIVE = {"bpr"}

QUICK_GRID = {
    "item_knn": [{"n_neighbors": 50}],
    "svd": [{"n_factors": 32}],
    "als": [{"n_factors": 32, "reg": 0.1}],
    "bpr": [{"n_factors": 32, "reg": 0.01}],
}


def popularity_correlation(model, user_sample, popularity_counts) -> float:
    """Mean Spearman rho between a model's per-user item ranking and global popularity."""
    rhos = []
    for user_id in user_sample:
        if user_id not in model.user_index:
            continue
        rho = spearmanr(model.scores(user_id), popularity_counts).statistic
        if not np.isnan(rho):
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else float("nan")


def params_label(params: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in params.items()) or "-"


def main():
    parser = argparse.ArgumentParser(description="CF sweep over synthetic_interactions")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="one config per model, for a smoke run")
    parser.add_argument("--models", nargs="+", default=list(FULL_GRID), choices=list(FULL_GRID))
    parser.add_argument("--weightings", nargs="+", default=["binary", "confidence"])
    parser.add_argument("--corr-sample", type=int, default=300,
                        help="users sampled for the popularity-correlation diagnostic")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data" / "cf_sweep.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(args.random_seed)
    users = load_users()
    interactions = load_interactions()
    train, test = build_train_test_split(interactions, args.test_fraction)
    _, train_seen = relevance_and_seen(train)
    test_relevance, _ = relevance_and_seen(test)
    item_ids, item_tags, item_budgets = load_catalog(CATALOG_PATH)

    counts = Counter(r["item_id"] for r in train if r["event_type"] in ("click", "save"))
    popularity_counts = np.array([counts.get(i, 0) for i in item_ids], dtype=float)
    corr_users = list(rng.choice(sorted(test_relevance),
                                 size=min(args.corr_sample, len(test_relevance)), replace=False))

    grid = QUICK_GRID if args.quick else FULL_GRID
    max_k = max(args.k)

    print(f"train rows: {len(train):,}  test users with >=1 held-out positive: {len(test_relevance):,}")
    header = (f"{'model':<10}{'weighting':<12}{'params':<26}"
              + "".join(f"{f'ndcg@{k}':>10}" for k in args.k)
              + f"{'recall@10':>11}{'hit@10':>9}{'pop_rho':>9}{'train_s':>9}")
    print("\n" + header)
    print("-" * len(header))

    rows = []

    # Baselines first, through the same evaluate() the models use, so the comparison
    # is like-for-like rather than quoted from the eval README.
    for name, rec in (("random", RandomRecommender(item_ids, train_seen, rng)),
                      ("popularity", PopularityRecommender(train, item_ids, train_seen))):
        scores = evaluate(rec, test_relevance, args.k)
        row = {"model": name, "weighting": "-", "params": "-", "train_s": 0.0,
               "pop_rho": float("nan")}
        for k in args.k:
            row[f"ndcg@{k}"] = scores[k]["ndcg"]
            row[f"recall@{k}"] = scores[k]["recall"]
            row[f"hit@{k}"] = scores[k]["hit_rate"]
        rows.append(row)
        print(f"{name:<10}{'-':<12}{'-':<26}"
              + "".join(f"{row[f'ndcg@{k}']:>10.4f}" for k in args.k)
              + f"{row['recall@10']:>11.4f}{row['hit@10']:>9.4f}{'-':>9}{'-':>9}")

    for weighting_idx, weighting in enumerate(args.weightings):
        matrix, user_index = build_matrix(train, item_ids, weighting)
        for model_name in args.models:
            if model_name in WEIGHTING_INSENSITIVE and weighting_idx > 0:
                continue
            for params in grid[model_name]:
                t0 = time.perf_counter()
                model = MODELS[model_name](matrix, user_index, item_ids, train_seen,
                                           random_state=args.random_seed, **params)
                train_s = time.perf_counter() - t0

                scores = evaluate(model, test_relevance, args.k)
                rho = popularity_correlation(model, corr_users, popularity_counts)

                # Don't label a weighting-insensitive model with whichever matrix
                # happened to be built first.
                label = "either" if model_name in WEIGHTING_INSENSITIVE else weighting
                row = {"model": model_name, "weighting": label,
                       "params": params_label(params), "train_s": round(train_s, 2),
                       "pop_rho": round(rho, 4)}
                for k in args.k:
                    row[f"ndcg@{k}"] = scores[k]["ndcg"]
                    row[f"recall@{k}"] = scores[k]["recall"]
                    row[f"hit@{k}"] = scores[k]["hit_rate"]
                rows.append(row)
                print(f"{model_name:<10}{label:<12}{params_label(params):<26}"
                      + "".join(f"{row[f'ndcg@{k}']:>10.4f}" for k in args.k)
                      + f"{row['recall@10']:>11.4f}{row['hit@10']:>9.4f}"
                      f"{rho:>9.3f}{train_s:>9.1f}", flush=True)

    fields = (["model", "weighting", "params"]
              + [f"{m}@{k}" for k in args.k for m in ("ndcg", "recall", "hit")]
              + ["pop_rho", "train_s"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({f: (round(row[f], 4) if isinstance(row.get(f), float) else row.get(f, ""))
                             for f in fields})
    print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    main()
