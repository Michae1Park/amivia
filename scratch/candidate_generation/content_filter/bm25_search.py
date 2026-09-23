#!/usr/bin/env python3
"""Lexical retrieval baseline: BM25 over the same city descriptions
content_filter.py embeds. Same reason this repo cross-checks ALS/BPR against
`implicit` in collab_filter, or benchmarks four ANN libraries against each
other here — a dense-embedding result only means something next to a
lexical baseline, not in isolation. BM25 also answers a question embeddings
structurally can't: exact-term/rare-word matches ("Reykjavik", "$50") that
never needed semantic generalization in the first place.

Same tokenization on both sides of the index (query and corpus) — the
classic BM25 footgun is scoring a query against a differently-normalized
corpus and blaming the algorithm for a preprocessing mismatch.
"""
import argparse
import re

from rank_bm25 import BM25Okapi

from content_filter import DATA_PATH, load_cities

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_index(cities: list[dict]) -> BM25Okapi:
    corpus = [tokenize(c["short_description"]) for c in cities]
    return BM25Okapi(corpus)


def rank_all(query: str, bm25: BM25Okapi, cities: list[dict]) -> list[str]:
    """Every city ranked by BM25 score, ties broken by catalog order —
    matches content_filter.rank_all's contract for the shared eval harness."""
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(cities)), key=lambda i: -scores[i])
    return [cities[i]["id"] for i in order]


def search(query: str, bm25: BM25Okapi, cities: list[dict], k: int = 5):
    scores = bm25.get_scores(tokenize(query))
    top_idx = sorted(range(len(cities)), key=lambda i: -scores[i])[:k]
    return [(cities[i], scores[i]) for i in top_idx]


def main():
    parser = argparse.ArgumentParser(description="BM25 lexical retrieval over travel cities")
    parser.add_argument("query", nargs="?", help="Free-text travel query")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    cities = load_cities(DATA_PATH)
    bm25 = build_index(cities)
    query = args.query or input("Query: ")
    results = search(query, bm25, cities, k=args.k)

    print(f"\nTop {args.k} BM25 matches for: {query!r}\n")
    for city, score in results:
        print(f"{score:.3f}  {city['city']}, {city['country']}")
        print(f"       {city['short_description']}")


if __name__ == "__main__":
    main()
