#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_PATH = Path(__file__).parent / "data" / "Worldwide Travel Cities Dataset (Ratings and Climate).csv"
EMBED_CACHE = Path(__file__).parent / "data" / "description_embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_cities(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_embeddings(model: SentenceTransformer, cities: list[dict]) -> np.ndarray:
    if EMBED_CACHE.exists():
        return np.load(EMBED_CACHE)
    descriptions = [c["short_description"] for c in cities]
    embeddings = model.encode(descriptions, normalize_embeddings=True, show_progress_bar=True)
    np.save(EMBED_CACHE, embeddings)
    return embeddings


def search(query: str, model: SentenceTransformer, cities: list[dict], embeddings: np.ndarray, k: int = 5):
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ query_vec  # both sides are unit-normalized, so dot product == cosine similarity
    top_idx = np.argsort(-scores)[:k]
    return [(cities[i], scores[i]) for i in top_idx]


def main():
    parser = argparse.ArgumentParser(description="Embedding-based candidate retrieval over travel cities")
    parser.add_argument("query", nargs="?", help="Free-text travel query")
    parser.add_argument("-k", type=int, default=5, help="Number of candidates to return")
    args = parser.parse_args()

    cities = load_cities(DATA_PATH)
    model = SentenceTransformer(MODEL_NAME)
    embeddings = get_embeddings(model, cities)

    query = args.query or input("Query: ")
    results = search(query, model, cities, embeddings, k=args.k)

    print(f"\nTop {args.k} matches for: {query!r}\n")
    for city, score in results:
        print(f"{score:.3f}  {city['city']}, {city['country']}")
        print(f"       {city['short_description']}")


if __name__ == "__main__":
    main()
