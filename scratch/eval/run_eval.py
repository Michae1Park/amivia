#!/usr/bin/env python3
"""Offline eval harness: scores a recommender's ranked top-K against held-out
synthetic_interactions ground truth. See README.md for metric definitions and scope."""
import argparse

import numpy as np

from baselines import OraclePersonaRecommender, PopularityRecommender, RandomRecommender
from data import build_train_test_split, load_catalog, load_interactions, load_users, relevance_and_seen, CATALOG_PATH
from metrics import hit_rate_at_k, ndcg_at_k, recall_at_k


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
    args = parser.parse_args()

    rng = np.random.default_rng(args.random_seed)
    users = load_users()
    interactions = load_interactions()
    train, test = build_train_test_split(interactions, args.test_fraction)
    _, train_seen = relevance_and_seen(train)
    test_relevance, _ = relevance_and_seen(test)
    item_ids, item_tags, item_budgets = load_catalog(CATALOG_PATH)

    recommenders = {
        "random": RandomRecommender(item_ids, train_seen, rng),
        "popularity": PopularityRecommender(train, item_ids, train_seen),
        "oracle_persona": OraclePersonaRecommender(users, item_ids, item_tags, item_budgets, train_seen),
    }

    print(f"test users with >=1 held-out click/save: {len(test_relevance)} / {len(users)}")
    print(f"{'model':<16}{'k':>4}{'hit_rate':>10}{'recall':>10}{'ndcg':>10}")
    for name, rec in recommenders.items():
        scores = evaluate(rec, test_relevance, args.k)
        for k in args.k:
            m = scores[k]
            print(f"{name:<16}{k:>4}{m['hit_rate']:>10.4f}{m['recall']:>10.4f}{m['ndcg']:>10.4f}")


if __name__ == "__main__":
    main()
