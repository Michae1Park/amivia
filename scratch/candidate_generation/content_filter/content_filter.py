#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DATA_PATH = Path(__file__).parent.parent / "data" / "cities.csv"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBED_CACHE = Path(__file__).parent.parent / "data" / f"description_embeddings_{MODEL_NAME.replace('/', '_')}.npy"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# bge is an asymmetric retrieval model: queries need this instruction prefix to
# align with the passage embedding space, but passages (city descriptions) don't.
# https://huggingface.co/BAAI/bge-base-en-v1.5#usage
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def load_cities(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=DEVICE)


def get_embeddings(model: SentenceTransformer, cities: list[dict]) -> np.ndarray:
    if EMBED_CACHE.exists():
        return np.load(EMBED_CACHE)
    descriptions = [c["short_description"] for c in cities]
    embeddings = model.encode(descriptions, normalize_embeddings=True, show_progress_bar=True)
    np.save(EMBED_CACHE, embeddings)
    return embeddings


def search(query: str, model: SentenceTransformer, cities: list[dict], embeddings: np.ndarray, k: int = 5):
    query_vec = model.encode([QUERY_INSTRUCTION + query], normalize_embeddings=True)[0]
    scores = embeddings @ query_vec  # both sides are unit-normalized, so dot product == cosine similarity
    top_idx = np.argsort(-scores)[:k]
    return [(cities[i], scores[i]) for i in top_idx]


def main():
    parser = argparse.ArgumentParser(description="Content-based candidate retrieval over travel cities")
    parser.add_argument("query", nargs="?", help="Free-text travel query")
    parser.add_argument("-k", type=int, default=5, help="Number of candidates to return")
    args = parser.parse_args()

    cities = load_cities(DATA_PATH)
    model = load_model()
    embeddings = get_embeddings(model, cities)

    query = args.query or input("Query: ")
    results = search(query, model, cities, embeddings, k=args.k)

    print(f"\nTop {args.k} matches for: {query!r}\n")
    for city, score in results:
        print(f"{score:.3f}  {city['city']}, {city['country']}")
        print(f"       {city['short_description']}")


if __name__ == "__main__":
    main()
