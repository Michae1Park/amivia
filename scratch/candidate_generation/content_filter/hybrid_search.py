#!/usr/bin/env python3
"""Hybrid retrieval: fuse the dense embedding ranking and the BM25 ranking
with Reciprocal Rank Fusion (RRF), the standard way to combine retrievers
that score on incomparable scales (cosine similarity vs. BM25's unbounded
term-weighted score) without having to normalize either one.

RRF(d) = sum over rankers of 1 / (k + rank_of(d))

`k=60` is Cormack et al. 2009's original constant, kept as the default here
for the same reason `filter_constraints` keeps its budget-mapping assumption
visible rather than burying it: it's a real modelling choice, not a value
the data determined, so it's named rather than silently hardcoded.
"""
import argparse

import numpy as np

from bm25_search import build_index as build_bm25
from bm25_search import rank_all as bm25_rank_all
from content_filter import DATA_PATH, MODEL_NAME, QUERY_INSTRUCTION, load_cities, load_model
from content_filter import get_embeddings as get_dense_embeddings

# bge-family models are asymmetric (see content_filter.py's QUERY_INSTRUCTION note) —
# this heuristic covers the model this repo actually uses; a differently-asymmetric
# model added later would need its own instruction, not automatically this one.
_QUERY_PREFIX = QUERY_INSTRUCTION if "bge" in MODEL_NAME.lower() else ""

RRF_K = 60


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda item_id: -scores[item_id])


def dense_rank_all(query_vec: np.ndarray, cities: list[dict], embeddings: np.ndarray) -> list[str]:
    scores = embeddings @ query_vec
    order = np.argsort(-scores)
    return [cities[i]["id"] for i in order]


def main():
    parser = argparse.ArgumentParser(description="Hybrid (dense + BM25, RRF-fused) retrieval over travel cities")
    parser.add_argument("query", nargs="?")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    args = parser.parse_args()

    cities = load_cities(DATA_PATH)
    by_id = {c["id"]: c for c in cities}

    model = load_model()
    embeddings = get_dense_embeddings(model, cities)
    bm25 = build_bm25(cities)

    query = args.query or input("Query: ")
    query_vec = model.encode([_QUERY_PREFIX + query], normalize_embeddings=True)[0]

    dense_ranking = dense_rank_all(query_vec, cities, embeddings)
    bm25_ranking = bm25_rank_all(query, bm25, cities)
    fused = rrf_fuse([dense_ranking, bm25_ranking], k=args.rrf_k)[: args.k]

    print(f"\nTop {args.k} hybrid (RRF) matches for: {query!r}\n")
    for item_id in fused:
        city = by_id[item_id]
        print(f"{city['city']}, {city['country']}")
        print(f"       {city['short_description']}")


if __name__ == "__main__":
    main()
