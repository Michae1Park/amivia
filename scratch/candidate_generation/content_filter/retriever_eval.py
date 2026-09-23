#!/usr/bin/env python3
"""Unified comparison across every retriever this project now has: two
embedding models (whatever's cached under data/description_embeddings*.npy),
BM25, RRF hybrid, and random. Same ground truth as relevance_eval.py
(filter_constraints.LABELLED_QUERIES), extended two ways:

- adds MRR alongside hit_rate/recall/precision/ndcg
- adds a bootstrap 95% CI on every metric, because 15-49 labelled queries is
  small enough that "0.093 beat 0.087" can easily be noise, not a finding —
  relevance_eval.py never checked that, and at this sample size it matters

This doesn't replace relevance_eval.py (that script's job — a minimal,
single-model sanity check — still stands on its own); this is the fuller
comparison once there's more than one retriever to compare.
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "filter_constraints"))
from constraints import LABELLED_QUERIES, satisfies_all  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eval"))
from metrics import hit_rate_at_k, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402

from bm25_search import build_index as build_bm25  # noqa: E402
from bm25_search import rank_all as bm25_rank_all  # noqa: E402
from content_filter import DATA_PATH, QUERY_INSTRUCTION, load_cities, load_model  # noqa: E402
from hybrid_search import dense_rank_all, rrf_fuse  # noqa: E402

K_VALUES = [5, 10, 20, 50]
METRICS = {"hit_rate": hit_rate_at_k, "recall": recall_at_k, "precision": precision_at_k,
           "ndcg": ndcg_at_k, "mrr": mrr_at_k}
N_BOOTSTRAP = 1000


def relevant_set(cities: list[dict], predicates: tuple) -> dict[str, int]:
    return {c["id"]: 1 for c in cities if satisfies_all(c, predicates)}


# The pre-migration cache filename (data/description_embeddings.npy, no model
# suffix) predates content_filter.py embedding the model name into the cache
# path — it's always all-MiniLM-L6-v2, the original default.
LEGACY_CACHE_MODEL = "all-MiniLM-L6-v2"


def discover_embedding_caches() -> dict[str, tuple[Path, str]]:
    """Every data/description_embeddings*.npy file is a model comparison arm —
    point content_filter.MODEL_NAME at a different sentence-transformers model,
    run it once to populate the cache, and this eval picks it up with no code
    change. Returns {label: (cache_path, model_name)}."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    caches = {}
    for path in glob.glob(str(data_dir / "description_embeddings*.npy")):
        path = Path(path)
        suffix = path.stem.replace("description_embeddings", "").lstrip("_")
        if not suffix:
            caches[f"embedding[{LEGACY_CACHE_MODEL}]"] = (path, LEGACY_CACHE_MODEL)
        else:
            # Inverts content_filter.py's `model_name.replace("/", "_")`. Only exact
            # for the common case of one "/" in the HF model id (org/model) — flagged
            # rather than silently wrong if a model name ever contains its own "_".
            model_name = suffix.replace("_", "/", 1)
            caches[f"embedding[{model_name}]"] = (path, model_name)
    return caches


def bootstrap_ci(values: list[float], n_bootstrap: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Percentile bootstrap over per-query metric values. Returns (mean, lo95, hi95)."""
    values = np.array(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    boot_means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_bootstrap)])
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return mean, float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k-values", type=int, nargs="+", default=K_VALUES)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    cities = load_cities(DATA_PATH)
    rng = np.random.default_rng(args.random_seed)
    all_ids = [c["id"] for c in cities]
    queries = [q for q, _ in LABELLED_QUERIES]

    # -- build every ranker's full ranking for every query, once --
    embedding_caches = discover_embedding_caches()
    embedding_rankings: dict[str, list[list[str]]] = {}
    for name, (cache_path, model_name) in embedding_caches.items():
        embeddings = np.load(cache_path)
        model = load_model(model_name)
        # See hybrid_search.py's _QUERY_PREFIX note: bge-family models need this
        # instruction prefix on the query side to match their passage embedding space.
        prefix = QUERY_INSTRUCTION if "bge" in model_name.lower() else ""
        query_vecs = model.encode([prefix + q for q in queries], normalize_embeddings=True)
        embedding_rankings[name] = [dense_rank_all(qv, cities, embeddings) for qv in query_vecs]

    bm25 = build_bm25(cities)
    bm25_rankings = [bm25_rank_all(q, bm25, cities) for q in queries]

    hybrid_rankings = None
    if embedding_rankings:
        first_dense = next(iter(embedding_rankings.values()))
        hybrid_rankings = [rrf_fuse([d, b]) for d, b in zip(first_dense, bm25_rankings)]

    random_rankings = [rng.permutation(all_ids).tolist() for _ in queries]

    rankers: dict[str, list[list[str]]] = {**embedding_rankings, "bm25": bm25_rankings, "random": random_rankings}
    if hybrid_rankings is not None:
        rankers["hybrid[rrf]"] = hybrid_rankings

    # -- score --
    relevant_sets = [relevant_set(cities, predicates) for _, predicates in LABELLED_QUERIES]

    print(f"\n{'=' * 100}\n{len(LABELLED_QUERIES)} labelled queries, {args.n_bootstrap}x bootstrap resamples, "
          f"mean(95% CI lo-hi) per cell\n{'=' * 100}")
    col_width = 22
    name_width = max(20, max(len(n) for n in rankers) + 2)
    header = f"{'model':<{name_width}}{'k':>4}" + "".join(f"{m:>{col_width}}" for m in METRICS)
    print(header)

    for name, rankings in rankers.items():
        for k in args.k_values:
            cells = []
            for m, fn in METRICS.items():
                vals = [v for v in (fn(r, rel, k) for r, rel in zip(rankings, relevant_sets)) if v is not None]
                if not vals:
                    cells.append(f"{'--':>{col_width}}")
                    continue
                mean, lo, hi = bootstrap_ci(vals, args.n_bootstrap, rng)
                cells.append(f"{f'{mean:.3f} ({lo:.2f}-{hi:.2f})':>{col_width}}")
            print(f"{name:<{name_width}}{k:>4}" + "".join(cells))


if __name__ == "__main__":
    main()
