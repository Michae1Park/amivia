#!/usr/bin/env python3
"""Offline eval harness: scores a recommender's ranked top-K against held-out
synthetic_interactions ground truth. See README.md for metric definitions and scope."""
import argparse
import sys

import numpy as np

from pathlib import Path

from baselines import (
    LLMRecommender,
    OraclePersonaRecommender,
    PopularityRecommender,
    RandomRecommender,
)
from data import (
    build_train_test_split,
    load_catalog,
    load_catalog_rows,
    load_interactions,
    load_users,
    relevance_and_seen,
    CATALOG_PATH,
)
from metrics import hit_rate_at_k, ndcg_at_k, recall_at_k

LLM_CACHE_PATH = Path(__file__).resolve().parent / "data" / "llm_cache.json"

CF_DIR = Path(__file__).resolve().parent.parent / "candidate_generation" / "collab_filter"
# Best config per model from that project's sweep — see its README for the full grid.
CF_BEST_PARAMS = {
    "item_knn": {"n_neighbors": 560},  # 560 == no truncation; the full matrix scored best
    "svd": {"n_factors": 16},
    "als": {"n_factors": 16, "reg": 10.0, "alpha": 1.0},
    "bpr": {"n_factors": 64, "reg": 0.1},
}


def evaluate(recommender, test_relevance: dict, k_values: list[int]) -> dict:
    results = {k: {"hit_rate": [], "recall": [], "ndcg": []} for k in k_values}
    max_k = max(k_values)
    for user_id, relevant in test_relevance.items():
        recs = recommender.recommend(user_id, max_k)
        for k in k_values:
            results[k]["hit_rate"].append(hit_rate_at_k(recs, relevant, k))
            r = recall_at_k(recs, relevant, k)
            if r is not None:
                results[k]["recall"].append(r)
            n = ndcg_at_k(recs, relevant, k)
            if n is not None:
                results[k]["ndcg"].append(n)
    return {
        k: {metric: float(np.mean(vals)) if vals else float("nan") for metric, vals in m.items()}
        for k, m in results.items()
    }


def main():
    parser = argparse.ArgumentParser(description="Offline eval harness over synthetic_interactions")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--sample-users", type=int, default=None,
                        help="evaluate on a random subsample of test users (all models see "
                             "the same sample). Required in practice for --llm, which costs "
                             "an API call per user.")
    parser.add_argument("--cf", action="store_true",
                        help="add candidate_generation/collab_filter's four models at their "
                             "swept-best settings (see that project's README)")
    parser.add_argument("--cf-weighting", default="confidence", choices=["binary", "confidence"])
    parser.add_argument("--llm", action="store_true",
                        help="add the zero-shot LLM baseline (needs `pip install anthropic` "
                             "and credentials; costs money per uncached user)")
    parser.add_argument("--llm-model", default="claude-opus-5")
    parser.add_argument("--llm-effort", default="low",
                        choices=["low", "medium", "high", "xhigh", "max"])
    args = parser.parse_args()

    rng = np.random.default_rng(args.random_seed)
    users = load_users()
    interactions = load_interactions()
    train, test = build_train_test_split(interactions, args.test_fraction)
    _, train_seen = relevance_and_seen(train)
    test_relevance, _ = relevance_and_seen(test)
    item_ids, item_tags, item_budgets = load_catalog(CATALOG_PATH)

    n_eligible = len(test_relevance)
    if args.sample_users is not None and args.sample_users < n_eligible:
        # Subsample once, for every model — otherwise the LLM's numbers aren't
        # comparable to the baselines it is supposed to be measured against.
        keep = rng.choice(sorted(test_relevance), size=args.sample_users, replace=False)
        test_relevance = {u: test_relevance[u] for u in keep}

    recommenders = {
        "random": RandomRecommender(item_ids, train_seen, rng),
        "popularity": PopularityRecommender(train, item_ids, train_seen),
        "oracle_persona": OraclePersonaRecommender(users, item_ids, item_tags, item_budgets, train_seen),
    }
    if args.cf:
        # Imported lazily: the CF models live in another project and pull in scipy,
        # so a bare `run_eval.py` run stays independent of them.
        sys.path.insert(0, str(CF_DIR))
        from models import MODELS, build_matrix

        matrix, user_index = build_matrix(train, item_ids, args.cf_weighting)
        for name, params in CF_BEST_PARAMS.items():
            recommenders[name] = MODELS[name](
                matrix, user_index, item_ids, train_seen,
                random_state=args.random_seed, **params
            )

    llm = None
    if args.llm:
        llm = LLMRecommender(
            load_catalog_rows(), train, train_seen,
            model=args.llm_model, effort=args.llm_effort, cache_path=LLM_CACHE_PATH,
        )
        print(f"prefetching {len(test_relevance)} LLM recommendations ({args.llm_model}, "
              f"effort={args.llm_effort})...")
        llm.prefetch(sorted(test_relevance), max(args.k))
        recommenders["llm"] = llm

    print(f"test users with >=1 held-out click/save: {n_eligible} / {len(users)}"
          + (f" (evaluating {len(test_relevance)})" if len(test_relevance) != n_eligible else ""))
    print(f"{'model':<16}{'k':>4}{'hit_rate':>10}{'recall':>10}{'ndcg':>10}")
    for name, rec in recommenders.items():
        scores = evaluate(rec, test_relevance, args.k)
        for k in args.k:
            m = scores[k]
            print(f"{name:<16}{k:>4}{m['hit_rate']:>10.4f}{m['recall']:>10.4f}{m['ndcg']:>10.4f}")

    if llm is not None:
        print(llm.usage_summary())


if __name__ == "__main__":
    main()
