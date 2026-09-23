#!/usr/bin/env python3
"""Perturb the real 560-city catalog up to a larger synthetic one, for
generating a sparser, more realistic interaction log than the default 560-item
catalog can produce (see collab_filter/sparsity_sweep.py for why that matters).

Same idiom `content_filter/ann_benchmark.py` uses to scale the catalog to
10k/100k/1M for the ANN sweep — perturb real vectors rather than draw uniform
random ones, because uniformly random items are near-equidistant in tag space
and would understate how confusable real, correlated items actually are.

Only writes what `generate.load_catalog` reads: `id`, `budget_level`, and the
9 tag columns. `city`/`country` are synthetic placeholders for readability,
not used by anything downstream of this catalog.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import CATALOG_PATH, TAGS  # noqa: E402


def load_real_catalog(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def perturb(rows: list[dict], n_items: int, noise_sigma: float, rng: np.random.Generator) -> list[dict]:
    base_tags = np.array([[int(r[t]) for t in TAGS] for r in rows], dtype=float)
    centered = (base_tags - 3.0) / 2.0  # 1-5 -> [-1, 1], same convention as generate.load_catalog

    base_idx = rng.integers(0, len(rows), size=n_items)
    noise = rng.normal(0, noise_sigma, size=(n_items, len(TAGS)))
    perturbed = np.clip(centered[base_idx] + noise, -1.0, 1.0)
    tags_1to5 = np.rint(perturbed * 2.0 + 3.0).astype(int).clip(1, 5)

    out = []
    for i in range(n_items):
        base = rows[base_idx[i]]
        row = {"id": f"synth-{i:06d}", "city": f"Synthetic City {i}", "country": "Synthetica",
               "budget_level": base["budget_level"]}  # kept from the base city: preserves tag<->budget correlation
        row.update({tag: int(tags_1to5[i, j]) for j, tag in enumerate(TAGS)})
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-items", type=int, required=True, help="e.g. 2000, 5000, 20000")
    parser.add_argument("--noise-sigma", type=float, default=0.25,
                         help="higher = perturbed items drift further from their real base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_path = args.out or Path(__file__).parent / "data" / f"catalog_{args.n_items}.csv"
    rng = np.random.default_rng(args.seed)

    real_rows = load_real_catalog(args.catalog)
    synthetic_rows = perturb(real_rows, args.n_items, args.noise_sigma, rng)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "city", "country", "budget_level", *TAGS])
        w.writeheader()
        w.writerows(synthetic_rows)

    print(f"wrote {len(synthetic_rows)} synthetic items (perturbed from {len(real_rows)} real ones) to {out_path}")
    print(f"next: python3 generate.py --catalog {out_path} --out-dir data/sparse_{args.n_items} "
          f"--n-users 5000 --popularity-skew 1.5")


if __name__ == "__main__":
    main()
