#!/usr/bin/env python3
"""Run a batch of test queries to eyeball retrieval behavior across the query set at once.

Group queries as multiple phrasings of the same intent to check paraphrase
stability, or as a single phrasing to probe a specific failure mode
(negation, numeric constraints, etc).
"""
from sentence_transformers import SentenceTransformer

from content_filter import DATA_PATH, MODEL_NAME, get_embeddings, load_cities, search

QUERY_GROUPS = [
    ("paraphrase: beach relaxation", [
        "relaxing beach vacation",
        "chill seaside getaway, nothing fancy",
    ]),
    ("paraphrase: vibrant nightlife", [
        "vibrant nightlife and clubbing scene",
        "city that comes alive after dark",
    ]),
    ("negation: no nightlife", [
        "quiet town, definitely no nightlife",
    ]),
    ("negation: not touristy", [
        "authentic place, avoid touristy spots",
    ]),
    ("numeric: budget constraint", [
        "under $50 a day",
    ]),
    ("numeric: temperature constraint", [
        "average July temperature above 30 degrees Celsius",
    ]),
]

K = 5


def run_group(label, queries, model, cities, embeddings):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    result_sets = []
    for query in queries:
        results = search(query, model, cities, embeddings, k=K)
        result_sets.append({city["id"] for city, _ in results})
        print(f"\n  Query: {query!r}")
        for city, score in results:
            print(f"    {score:.3f}  {city['city']}, {city['country']}")

    if len(result_sets) > 1:
        overlap = len(result_sets[0].intersection(*result_sets[1:]))
        print(f"\n  Overlap across phrasings: {overlap}/{K} cities in common")


def main():
    cities = load_cities(DATA_PATH)
    model = SentenceTransformer(MODEL_NAME)
    embeddings = get_embeddings(model, cities)

    for label, queries in QUERY_GROUPS:
        run_group(label, queries, model, cities, embeddings)


if __name__ == "__main__":
    main()
