#!/usr/bin/env python3
"""Does the collab_filter README's finding — item-kNN wins, all four beat
popularity — survive at realistic, MovieLens-scale sparsity, or is it an
artifact of this project's small (560-item, 2.8%-dense) default catalog?

Real interaction logs are much sparser than that: MovieLens-25M sits around
0.25% density with a long tail of near-cold items. `synthetic_interactions`'s
default dataset has *zero* items with no click/save at all — every city gets
seen. `synthetic_interactions/scale_catalog.py` fixes that by perturbing the
catalog up to thousands of synthetic items before generating interactions
over it; see that script's docstring and this project's README for how to
build a sparse variant.

STATED SIMPLIFICATION: hyperparameters are fixed across every density point
(DEFAULT_PARAMS below), not re-swept per variant. `collab_filter/sweep.py`
already tuned these for the 560-item dense case; re-tuning at every scale is
its own project. That means this isolates the effect of density on a fixed
config, not "the best each model could do at each density" — a real
difference, worth remembering when reading the table.
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eval"))
from baselines import PopularityRecommender, RandomRecommender  # noqa: E402
from data import build_train_test_split, relevance_and_seen  # noqa: E402
from run_eval import evaluate  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "synthetic_interactions"))
from generate import load_catalog  # noqa: E402

from models import MODELS, build_matrix  # noqa: E402

DEFAULT_PARAMS = {
    "item_knn": {"n_neighbors": 50},
    "svd": {"n_factors": 32},
    "als": {"n_factors": 32, "reg": 10.0, "alpha": 40.0},
    "bpr": {"n_factors": 64, "reg": 0.1},
}


def load_csv_rows(path: Path) -> list[dict]:
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def popularity_correlation(model, user_sample, popularity_counts) -> float:
    rhos = []
    for user_id in user_sample:
        if user_id not in model.user_index:
            continue
        rho = spearmanr(model.scores(user_id), popularity_counts).statistic
        if not np.isnan(rho):
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else float("nan")


def parse_variant(spec: str) -> tuple[str, Path, Path]:
    """label:interactions_dir:catalog_csv"""
    label, interactions_dir, catalog_csv = spec.split(":", 2)
    return label, Path(interactions_dir), Path(catalog_csv)


def run_variant(label: str, interactions_dir: Path, catalog_csv: Path,
                 k_values: list[int], test_fraction: float, corr_sample: int, seed: int) -> list[dict]:
    users = load_csv_rows(interactions_dir / "users.csv")
    interactions = load_csv_rows(interactions_dir / "interactions.csv")
    item_ids, _, _ = load_catalog(catalog_csv)

    train, test = build_train_test_split(interactions, test_fraction)
    _, train_seen = relevance_and_seen(train)
    test_relevance, _ = relevance_and_seen(test)

    n_users, n_items = len(users), len(item_ids)
    click_save_cells = len({(r["user_id"], r["item_id"]) for r in interactions
                             if r["event_type"] in ("click", "save")})
    density = click_save_cells / (n_users * n_items)
    train_item_counts = Counter(r["item_id"] for r in train if r["event_type"] in ("click", "save"))
    cold_items = n_items - len(train_item_counts)

    rng = np.random.default_rng(seed)
    popularity_counts = np.array([train_item_counts.get(i, 0) for i in item_ids], dtype=float)
    corr_users = list(rng.choice(sorted(test_relevance),
                                  size=min(corr_sample, len(test_relevance)), replace=False)) if test_relevance else []

    rows = []
    for name, rec in (("random", RandomRecommender(item_ids, train_seen, rng)),
                       ("popularity", PopularityRecommender(train, item_ids, train_seen))):
        scores = evaluate(rec, test_relevance, k_values)
        rows.append({"variant": label, "density_pct": density * 100, "cold_item_pct": cold_items / n_items * 100,
                     "model": name, "pop_rho": float("nan"), **{f"{m}@{k}": scores[k][m]
                     for k in k_values for m in ("recall", "ndcg", "hit_rate")}})

    matrix, user_index = build_matrix(train, item_ids, "confidence")
    for model_name, params in DEFAULT_PARAMS.items():
        t0 = time.perf_counter()
        model = MODELS[model_name](matrix, user_index, item_ids, train_seen, random_state=seed, **params)
        train_s = time.perf_counter() - t0
        scores = evaluate(model, test_relevance, k_values)
        rho = popularity_correlation(model, corr_users, popularity_counts) if corr_users else float("nan")
        rows.append({"variant": label, "density_pct": density * 100, "cold_item_pct": cold_items / n_items * 100,
                     "model": model_name, "pop_rho": rho, "train_s": round(train_s, 2),
                     **{f"{m}@{k}": scores[k][m] for k in k_values for m in ("recall", "ndcg", "hit_rate")}})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", action="append", required=True,
                         help="label:interactions_dir:catalog_csv, repeatable. "
                              "e.g. 'dense (560)':../../synthetic_interactions/data:../data/cities.csv")
    parser.add_argument("--k", type=int, nargs="+", default=[10])
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--corr-sample", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_rows = []
    for spec in args.variant:
        label, interactions_dir, catalog_csv = parse_variant(spec)
        all_rows.extend(run_variant(label, interactions_dir, catalog_csv, args.k,
                                     args.test_fraction, args.corr_sample, args.seed))

    metric_cols = [f"{m}@{k}" for k in args.k for m in ("recall", "ndcg", "hit_rate")]
    header = (f"{'variant':<24}{'density%':>9}{'cold%':>7}{'model':<12}"
              + "".join(f"{c:>12}" for c in metric_cols) + f"{'pop_rho':>9}")
    print(header)
    print("-" * len(header))
    for row in all_rows:
        print(f"{row['variant']:<24}{row['density_pct']:>9.3f}{row['cold_item_pct']:>7.1f}{row['model']:<12}"
              + "".join(f"{row.get(c, float('nan')):>12.4f}" for c in metric_cols)
              + f"{row['pop_rho']:>9.3f}")


if __name__ == "__main__":
    main()
