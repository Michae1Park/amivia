#!/usr/bin/env python3
"""Cross-check the hand-rolled ALS and BPR against the `implicit` library.

These two are the only models here with a training loop that can be silently wrong:
a sign error in the BPR gradient or a mis-derived ALS normal equation still produces
plausible-looking numbers, which would then be written up as a finding. This compares
both against a reference implementation on the same matrix.

Exact agreement is not the bar and not achievable — different RNG, different
initialisation, and implicit's BPR carries an item bias term the hand-rolled one
does not. The bar is that the two produce *the same ranking*: high rank correlation
between their scores and substantial overlap in the top-10 they recommend.
"""
import os

# implicit warns (and slows down) if BLAS is also threading underneath it.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "eval"
sys.path.insert(0, str(EVAL_DIR))

from data import (  # noqa: E402
    CATALOG_PATH,
    build_train_test_split,
    load_catalog,
    load_interactions,
    relevance_and_seen,
)

from models import ALS, BPR, build_matrix  # noqa: E402

N_FACTORS = 32
REG = 0.1
ALPHA = 40.0
ALS_ITERATIONS = 15
BPR_EPOCHS = 30
BPR_LR = 0.05


def agreement(mine, theirs, user_ids, user_index, item_ids, k=10):
    """Rank correlation over all items + top-k overlap, averaged across users."""
    rhos, overlaps = [], []
    for user_id in user_ids:
        row = user_index[user_id]
        a = mine.scores(user_id)
        b = theirs[0][row] @ theirs[1].T
        rho = spearmanr(a, b).statistic
        if not np.isnan(rho):
            rhos.append(rho)
        top_a = set(np.argsort(-a)[:k].tolist())
        top_b = set(np.argsort(-b)[:k].tolist())
        overlaps.append(len(top_a & top_b) / k)
    return float(np.mean(rhos)), float(np.mean(overlaps))


def main():
    parser = argparse.ArgumentParser(description="Verify hand-rolled ALS/BPR against `implicit`")
    parser.add_argument("--n-users", type=int, default=200, help="users sampled for the comparison")
    parser.add_argument("--weighting", default="confidence", choices=["binary", "confidence"])
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    import implicit

    rng = np.random.default_rng(args.random_seed)
    interactions = load_interactions()
    train, _ = build_train_test_split(interactions, 0.2)
    _, train_seen = relevance_and_seen(train)
    item_ids, _, _ = load_catalog(CATALOG_PATH)

    matrix, user_index = build_matrix(train, item_ids, args.weighting)
    sample = list(rng.choice(sorted(user_index), size=min(args.n_users, len(user_index)), replace=False))
    csr32 = matrix.astype(np.float32).tocsr()

    print(f"matrix: {matrix.shape[0]:,} users x {matrix.shape[1]} items, "
          f"{matrix.nnz:,} nonzeros ({args.weighting} weighting)\n")

    print(f"ALS  (factors={N_FACTORS}, reg={REG}, alpha={ALPHA}, iterations={ALS_ITERATIONS})")
    mine = ALS(matrix, user_index, item_ids, train_seen, n_factors=N_FACTORS, reg=REG,
               alpha=ALPHA, n_iterations=ALS_ITERATIONS, random_state=args.random_seed)
    ref = implicit.als.AlternatingLeastSquares(
        factors=N_FACTORS, regularization=REG, alpha=ALPHA,
        iterations=ALS_ITERATIONS, random_state=args.random_seed, use_gpu=False,
    )
    ref.fit(csr32, show_progress=False)
    rho, overlap = agreement(mine, (np.asarray(ref.user_factors), np.asarray(ref.item_factors)),
                             sample, user_index, item_ids)
    print(f"  mean Spearman rho vs implicit : {rho:.4f}")
    print(f"  mean top-10 overlap           : {overlap:.4f}\n")

    print(f"BPR  (factors={N_FACTORS}, reg={REG}, lr={BPR_LR}, epochs={BPR_EPOCHS})")
    mine_bpr = BPR(matrix, user_index, item_ids, train_seen, n_factors=N_FACTORS, reg=REG,
                   learning_rate=BPR_LR, n_epochs=BPR_EPOCHS, random_state=args.random_seed)
    ref_bpr = implicit.bpr.BayesianPersonalizedRanking(
        factors=N_FACTORS, learning_rate=BPR_LR, regularization=REG,
        iterations=BPR_EPOCHS, random_state=args.random_seed, use_gpu=False,
    )
    ref_bpr.fit(csr32, show_progress=False)
    # implicit's BPR appends an item-bias column, so its item factors are one wider
    # than its user factors; the matching bias column of ones is on the user side.
    rho_bpr, overlap_bpr = agreement(
        mine_bpr, (np.asarray(ref_bpr.user_factors), np.asarray(ref_bpr.item_factors)),
        sample, user_index, item_ids,
    )
    print(f"  mean Spearman rho vs implicit : {rho_bpr:.4f}")
    print(f"  mean top-10 overlap           : {overlap_bpr:.4f}")


if __name__ == "__main__":
    main()
