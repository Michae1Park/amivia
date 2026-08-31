#!/usr/bin/env python3
"""Does filtering after retrieval actually fix the negation/numeric failures, and
what does it cost?

Three conditions over embed_retrieve's top-100 candidates:
  no_filter     - what the retriever returns today
  hard_fail     - drop every candidate violating a constraint
  soft_penalty  - demote violators by a fixed margin instead of dropping them

Primary metric is constraint-violation rate in the returned top-10; the cost side
is how much of the unfiltered top-10 survives.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "embed_retrieve"))
from embed_retrieve import DATA_PATH, EMBED_CACHE, MODEL_NAME, load_cities  # noqa: E402

from constraints import (  # noqa: E402
    LABELLED_QUERIES,
    UNMAPPABLE_QUERIES,
    satisfies_all,
    violations,
)

N_CANDIDATES = 100
K = 10
PENALTIES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


def get_embeddings(cities):
    if EMBED_CACHE.exists():
        return np.load(EMBED_CACHE)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        [c["short_description"] for c in cities], normalize_embeddings=True, show_progress_bar=True
    )
    np.save(EMBED_CACHE, embeddings)
    return embeddings


def retrieve(query_vec, cities, embeddings, n):
    """embed_retrieve's brute-force retrieval: the candidate set this layer filters."""
    scores = embeddings @ query_vec
    top = np.argsort(-scores)[:n]
    return [(cities[i], float(scores[i])) for i in top]


# ---- the three conditions --------------------------------------------------


def no_filter(candidates, predicates, k):
    return candidates[:k]


def hard_fail(candidates, predicates, k):
    return [(c, s) for c, s in candidates if satisfies_all(c, predicates)][:k]


def soft_penalty(candidates, predicates, k, penalty):
    rescored = [(c, s - penalty * violations(c, predicates)) for c, s in candidates]
    rescored.sort(key=lambda cs: -cs[1])
    return rescored[:k]


# ---- metrics ---------------------------------------------------------------


def evaluate(results, baseline, predicates, k):
    """violation rate over what was actually returned; retention against the
    unfiltered top-k, which is the recall the filter costs."""
    if not results:
        return {"violation_rate": 0.0, "retention": 0.0, "mean_score": float("nan"), "n_returned": 0}
    baseline_ids = {c["id"] for c, _ in baseline}
    n_violating = sum(1 for c, _ in results if violations(c, predicates) > 0)
    kept = sum(1 for c, _ in results if c["id"] in baseline_ids)
    return {
        "violation_rate": n_violating / len(results),
        "retention": kept / min(k, len(baseline)),
        # Scores here are the ORIGINAL cosine, not the penalised one — the question is
        # how far down the similarity ranking filtering pushes you.
        "mean_score": float(np.mean([s for _, s in results])),
        "n_returned": len(results),
    }


def mean_of(rows, field):
    vals = [r[field] for r in rows if not np.isnan(r[field])]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Hard-constraint filtering over retrieved candidates")
    parser.add_argument("--n-candidates", type=int, default=N_CANDIDATES)
    parser.add_argument("-k", type=int, default=K)
    parser.add_argument("--detail", action="store_true", help="print per-query top-10 for each condition")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data" / "filter_results.csv")
    args = parser.parse_args()

    cities = load_cities(DATA_PATH)
    embeddings = get_embeddings(cities)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    query_vecs = model.encode([q for q, _ in LABELLED_QUERIES], normalize_embeddings=True)

    conditions = {"no_filter": no_filter, "hard_fail": hard_fail}
    for p in PENALTIES:
        conditions[f"soft_p={p}"] = (lambda c, pr, k, p=p: soft_penalty(c, pr, k, p))

    per_condition = {name: [] for name in conditions}
    feasibility = []

    for (query, predicates), qvec in zip(LABELLED_QUERIES, query_vecs):
        candidates = retrieve(qvec, cities, embeddings, args.n_candidates)
        baseline = no_filter(candidates, predicates, args.k)
        n_valid = sum(1 for c, _ in candidates if satisfies_all(c, predicates))
        feasibility.append({"query": query, "n_valid": n_valid,
                            "constraints": " AND ".join(p.describe() for p in predicates)})

        for name, fn in conditions.items():
            results = fn(candidates, predicates, args.k)
            per_condition[name].append(evaluate(results, baseline, predicates, args.k))

        if args.detail:
            print(f"\n{'=' * 74}\n{query!r}\n  constraints: "
                  f"{' AND '.join(p.describe() for p in predicates)}\n"
                  f"  satisfying candidates in top-{args.n_candidates}: {n_valid}\n{'=' * 74}")
            for name in ("no_filter", "hard_fail"):
                print(f"  -- {name} --")
                for c, s in conditions[name](candidates, predicates, args.k):
                    flag = "  VIOLATES" if violations(c, predicates) else ""
                    print(f"    {s:.3f}  {c['city']}, {c['country']}{flag}")

    print(f"\n{'=' * 74}\nCandidate feasibility (satisfying items within the top-"
          f"{args.n_candidates})\n{'=' * 74}")
    for f in sorted(feasibility, key=lambda x: x["n_valid"]):
        print(f"  {f['n_valid']:>3}/{args.n_candidates}  {f['query'][:44]:<46} [{f['constraints']}]")

    print(f"\n{'=' * 74}\nConditions ({len(LABELLED_QUERIES)} labelled queries, top-{args.k})\n{'=' * 74}")
    print(f"{'condition':<16}{'violation_rate':>16}{'retention':>12}{'mean_score':>12}{'n_returned':>12}")
    for name in conditions:
        rows = per_condition[name]
        print(f"{name:<16}{mean_of(rows, 'violation_rate'):>16.3f}{mean_of(rows, 'retention'):>12.3f}"
              f"{mean_of(rows, 'mean_score'):>12.3f}{mean_of(rows, 'n_returned'):>12.1f}")

    if UNMAPPABLE_QUERIES:
        print(f"\nNot labelled (no honest predicate in this schema):")
        for q, why in UNMAPPABLE_QUERIES:
            print(f"  {q!r} - {why}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["condition", "violation_rate", "retention",
                                           "mean_score", "n_returned"])
        w.writeheader()
        for name in conditions:
            rows = per_condition[name]
            w.writerow({"condition": name,
                        "violation_rate": round(mean_of(rows, "violation_rate"), 4),
                        "retention": round(mean_of(rows, "retention"), 4),
                        "mean_score": round(mean_of(rows, "mean_score"), 4),
                        "n_returned": round(mean_of(rows, "n_returned"), 2)})
    print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    main()
