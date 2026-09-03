#!/usr/bin/env python3
"""Visualize a single embedding vector (or two, side by side) as a terminal heatmap.

Reads directly from the cached description_embeddings.npy — never re-embeds,
so it also works as a quick sanity check that the cache matches cities.csv.
"""
import argparse
import sys

import numpy as np

from content_filter import DATA_PATH, EMBED_CACHE, load_cities

COLS = 32  # dims per row in the heatmap


def find_city(query: str, cities: list[dict]) -> tuple[int, dict]:
    query_lower = query.lower()
    matches = [(i, c) for i, c in enumerate(cities) if query_lower in c["city"].lower()]
    if not matches:
        sys.exit(f"No city matching {query!r}")
    if len(matches) > 1:
        print(f"Multiple cities match {query!r}, be more specific:")
        for _, c in matches:
            print(f"  {c['city']}, {c['country']}")
        sys.exit(1)
    return matches[0]


def color_block(value: float, vmax: float) -> str:
    """ANSI truecolor block: blue = negative, red = positive, white = ~0."""
    t = value / vmax if vmax else 0.0
    if t >= 0:
        r, g, b = 255, int(255 * (1 - t)), int(255 * (1 - t))
    else:
        r, g, b = int(255 * (1 + t)), int(255 * (1 + t)), 255
    return f"\x1b[48;2;{r};{g};{b}m  \x1b[0m"


def print_heatmap(vec: np.ndarray, cols: int = COLS) -> None:
    vmax = float(np.abs(vec).max())
    for row_start in range(0, len(vec), cols):
        row = vec[row_start : row_start + cols]
        print("".join(color_block(v, vmax) for v in row))


def print_legend(vmax: float) -> None:
    steps = 10
    blocks = "".join(color_block(-vmax + 2 * vmax * i / (steps - 1), vmax) for i in range(steps))
    print(f"{blocks}  (blue={-vmax:.3f} .. white=0 .. red={vmax:.3f})")


def print_stats(vec: np.ndarray) -> None:
    print(
        f"dims={vec.shape[0]}  norm={np.linalg.norm(vec):.4f}  "
        f"min={vec.min():.4f}  max={vec.max():.4f}  mean={vec.mean():.4f}  std={vec.std():.4f}"
    )


def inspect(city: dict, vec: np.ndarray) -> None:
    print(f"\n{city['city']}, {city['country']}")
    print(f"  {city['short_description']}")
    print_stats(vec)
    print_heatmap(vec)
    print_legend(float(np.abs(vec).max()))


def main():
    global COLS
    parser = argparse.ArgumentParser(description="Visualize embedding vectors as a terminal heatmap")
    parser.add_argument("city", help="City name (case-insensitive substring match)")
    parser.add_argument("city2", nargs="?", help="Optional second city to compare against")
    parser.add_argument("--cols", type=int, default=COLS, help="Heatmap columns per row")
    args = parser.parse_args()
    COLS = args.cols

    if not EMBED_CACHE.exists():
        sys.exit(f"No cached embeddings at {EMBED_CACHE} — run content_filter.py once to build it")

    cities = load_cities(DATA_PATH)
    embeddings = np.load(EMBED_CACHE)

    idx1, city1 = find_city(args.city, cities)
    inspect(city1, embeddings[idx1])

    if args.city2:
        idx2, city2 = find_city(args.city2, cities)
        inspect(city2, embeddings[idx2])
        similarity = float(embeddings[idx1] @ embeddings[idx2])
        print(f"\nCosine similarity ({city1['city']} vs {city2['city']}): {similarity:.4f}")


if __name__ == "__main__":
    main()
