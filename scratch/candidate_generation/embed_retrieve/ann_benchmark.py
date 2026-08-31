#!/usr/bin/env python3
"""ANN sweep: at what catalog size does an index actually beat brute force?

At 560 cities brute force is instant and nothing here is needed — that is the
point. The question this answers is the *crossover*: how large the catalog has
to get before IVF/HNSW/LSH earn their build cost and recall loss, which is what
tells Phase 2 whether per-city POI retrieval needs an index at all.

Ground truth is exact brute-force top-k on the same corpus, so recall is
self-consistent at every size.

Timing is single-query and single-threaded on both sides (faiss OMP pinned to 1,
NumPy's BLAS pinned via threadpoolctl). Batched or multi-threaded search would
flatter brute force, which parallelises perfectly, and hide the per-query
latency that actually matters online. Index *build* is left multi-threaded,
since that is how you would really build one.
"""
import argparse
import csv
import gc
import os
import tempfile
import time
from pathlib import Path

import faiss
import numpy as np
from threadpoolctl import threadpool_limits

from embed_retrieve import DATA_PATH, EMBED_CACHE, MODEL_NAME, load_cities

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_PATH = DATA_DIR / "ann_benchmark.csv"
PROBE_CACHE = DATA_DIR / "ann_probe_queries.npy"

# The real text probes from batch_test.py. Too few for stable p95, so the query
# set is topped up with perturbed catalog vectors — but these keep real queries
# in the mix rather than benchmarking purely on synthetic ones.
PROBE_QUERIES = [
    "relaxing beach vacation",
    "chill seaside getaway, nothing fancy",
    "vibrant nightlife and clubbing scene",
    "city that comes alive after dark",
    "quiet town, definitely no nightlife",
    "authentic place, avoid touristy spots",
    "under $50 a day",
    "average July temperature above 30 degrees Celsius",
    "mountain hiking and alpine scenery",
]

CSV_FIELDS = ["size", "index", "params", "build_s", "recall_at_k", "p50_ms", "p95_ms", "index_mb"]


# ---- corpus and queries ----------------------------------------------------


def load_base_embeddings() -> np.ndarray:
    """The 560 real city embeddings, unit-normalised (dot product == cosine)."""
    if EMBED_CACHE.exists():
        return np.ascontiguousarray(np.load(EMBED_CACHE), dtype=np.float32)
    from sentence_transformers import SentenceTransformer

    cities = load_cities(DATA_PATH)
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        [c["short_description"] for c in cities], normalize_embeddings=True, show_progress_bar=True
    )
    np.save(EMBED_CACHE, embeddings)
    return np.ascontiguousarray(embeddings, dtype=np.float32)


def load_probe_vectors(dim: int) -> np.ndarray:
    """Encoded real text queries, cached so the sweep doesn't reload the model."""
    if PROBE_CACHE.exists():
        cached = np.load(PROBE_CACHE)
        if cached.shape == (len(PROBE_QUERIES), dim):
            return np.ascontiguousarray(cached, dtype=np.float32)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    vecs = model.encode(PROBE_QUERIES, normalize_embeddings=True)
    np.save(PROBE_CACHE, vecs)
    return np.ascontiguousarray(vecs, dtype=np.float32)


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def scale_corpus(base: np.ndarray, n: int, rng: np.random.Generator, noise: float = 0.05) -> np.ndarray:
    """Grow the real catalog to n vectors by perturbing sampled real ones.

    Deliberately not uniform random vectors: in 384 dims those are all roughly
    equidistant, which destroys the cluster structure every one of these indexes
    exploits and would make the recall numbers meaningless. The real 560 stay as
    the first rows.

    Noise level matters more than it looks. At 0.05 each synthetic point sits at
    cosine ~0.71 from its parent, giving 560 moderately tight clusters — and HNSW
    turns out to be sensitive to exactly that structure (see the README: high
    efConstruction builds a *worse* graph on clustered data, an effect that
    disappears once clusters are loosened). Raising the noise is not a fix:
    at 0.3 the perturbation dominates the unit-norm base vector ~6:1, so the
    corpus degenerates toward uniform random, which is the case this scale-up
    was designed to avoid.
    """
    if n <= len(base):
        return np.ascontiguousarray(base[:n], dtype=np.float32)
    idx = rng.integers(0, len(base), size=n - len(base))
    extra = base[idx] + rng.normal(0, noise, size=(n - len(base), base.shape[1]))
    return np.ascontiguousarray(unit(np.vstack([base, extra])), dtype=np.float32)


def build_queries(base: np.ndarray, n_queries: int, rng: np.random.Generator) -> np.ndarray:
    probes = load_probe_vectors(base.shape[1])
    n_extra = max(0, n_queries - len(probes))
    idx = rng.integers(0, len(base), size=n_extra)
    extra = base[idx] + rng.normal(0, 0.1, size=(n_extra, base.shape[1]))
    return np.ascontiguousarray(unit(np.vstack([probes, extra])), dtype=np.float32)


# ---- measurement -----------------------------------------------------------


def exact_topk(corpus: np.ndarray, queries: np.ndarray, k: int, chunk: int = 25) -> np.ndarray:
    """Ground truth. Chunked over queries so a 1M corpus doesn't allocate a huge score matrix."""
    out = np.empty((len(queries), k), dtype=np.int64)
    for start in range(0, len(queries), chunk):
        q = queries[start : start + chunk]
        scores = corpus @ q.T  # (n_items, chunk)
        top = np.argpartition(-scores, k, axis=0)[:k]
        order = np.take_along_axis(scores, top, axis=0)
        for j in range(q.shape[0]):
            out[start + j] = top[np.argsort(-order[:, j]), j]
    return out


def recall_at_k(retrieved: np.ndarray, truth: np.ndarray) -> float:
    k = truth.shape[1]
    hits = [len(set(r.tolist()) & set(t.tolist())) for r, t in zip(retrieved, truth)]
    return float(np.mean(hits) / k)


def percentiles(latencies_s: list[float]) -> tuple[float, float]:
    arr = np.array(latencies_s) * 1000.0
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 95))


def time_brute_force(corpus: np.ndarray, queries: np.ndarray, k: int) -> tuple[list[float], np.ndarray]:
    """What embed_retrieve.py ships today: a full dot product, one query at a time."""
    retrieved = np.empty((len(queries), k), dtype=np.int64)
    latencies = []
    with threadpool_limits(limits=1):
        for _ in range(3):  # warm up
            _ = corpus @ queries[0]
        for i, q in enumerate(queries):
            t0 = time.perf_counter()
            scores = corpus @ q
            top = np.argpartition(-scores, k)[:k]
            top = top[np.argsort(-scores[top])]
            latencies.append(time.perf_counter() - t0)
            retrieved[i] = top
    return latencies, retrieved


def time_index(index, queries: np.ndarray, k: int) -> tuple[list[float], np.ndarray]:
    retrieved = np.empty((len(queries), k), dtype=np.int64)
    latencies = []
    for _ in range(3):
        index.search(queries[:1], k)
    for i, q in enumerate(queries):
        qv = q.reshape(1, -1)
        t0 = time.perf_counter()
        _, ids = index.search(qv, k)
        latencies.append(time.perf_counter() - t0)
        retrieved[i] = ids[0]
    return latencies, retrieved


def index_megabytes(index) -> float:
    """Via a temp file rather than faiss.serialize_index — that returns an in-memory
    copy, which at 1M vectors would double peak RSS just to measure it."""
    fd, path = tempfile.mkstemp(suffix=".faiss")
    os.close(fd)
    try:
        faiss.write_index(index, path)
        return os.path.getsize(path) / 1e6
    finally:
        os.unlink(path)


def measure(index, queries, truth, k, build_s, name, params, size) -> dict:
    latencies, retrieved = time_index(index, queries, k)
    p50, p95 = percentiles(latencies)
    return {
        "size": size,
        "index": name,
        "params": params,
        "build_s": round(build_s, 3),
        "recall_at_k": round(recall_at_k(retrieved, truth), 4),
        "p50_ms": round(p50, 4),
        "p95_ms": round(p95, 4),
        "index_mb": round(index_megabytes(index), 2),
    }


# ---- index builders --------------------------------------------------------


def timed_build(fn, *fn_args):
    """Build multi-threaded (how you'd really build one), then pin back to a single
    thread so the search timing that follows stays apples-to-apples with NumPy."""
    faiss.omp_set_num_threads(os.cpu_count() or 1)
    t0 = time.perf_counter()
    try:
        index = fn(*fn_args)
    finally:
        faiss.omp_set_num_threads(1)
    return index, time.perf_counter() - t0


def build_ivf(corpus: np.ndarray, nlist: int):
    d = corpus.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(corpus)
    index.add(corpus)
    return index


def build_hnsw(corpus: np.ndarray, m: int, ef_construction: int):
    # L2 rather than inner product: HNSW's graph construction assumes a true metric,
    # and inner product isn't one. On unit-normalised vectors the two rank identically
    # (||a-b||^2 == 2 - 2a.b), so this costs nothing and keeps the graph well-defined.
    # Verified equivalent at n=10k before switching.
    index = faiss.IndexHNSWFlat(corpus.shape[1], m, faiss.METRIC_L2)
    index.hnsw.efConstruction = ef_construction
    index.add(corpus)
    return index


def build_lsh(corpus: np.ndarray, nbits: int):
    index = faiss.IndexLSH(corpus.shape[1], nbits)
    index.train(corpus)
    index.add(corpus)
    return index


def viable_nlists(nlists: list[int], n: int) -> list[int]:
    """k-means on ~10 points per centroid or fewer is degenerate clustering, not a
    tuning choice — skip those rather than reporting a meaningless recall."""
    return [nl for nl in nlists if nl <= max(1, n // 10)]


# ---- sweep -----------------------------------------------------------------


def sweep_size(size, corpus, queries, truth, k, grid, writer, best_only=None):
    """Yields measured rows for one corpus size. best_only: {index_name: params_dict}
    to run just the carried-over winners instead of the whole grid."""
    rows = []

    def emit(row):
        rows.append(row)
        writer(row)

    t0 = time.perf_counter()
    latencies, retrieved = time_brute_force(corpus, queries, k)
    p50, p95 = percentiles(latencies)
    emit({
        "size": size, "index": "brute_force", "params": "-",
        "build_s": 0.0,  # no index to build: that is brute force's whole advantage
        "recall_at_k": round(recall_at_k(retrieved, truth), 4),
        "p50_ms": round(p50, 4), "p95_ms": round(p95, 4),
        "index_mb": round(corpus.nbytes / 1e6, 2),
    })
    print(f"  brute_force            recall=1.0000 p50={p50:.3f}ms p95={p95:.3f}ms "
          f"({time.perf_counter() - t0:.1f}s)", flush=True)

    # IVF: one build per nlist, then sweep nprobe on the built index.
    ivf_jobs = ([best_only["ivf"]] if best_only and "ivf" in best_only
                else [{"nlist": nl, "nprobes": grid["nprobe"]} for nl in viable_nlists(grid["nlist"], size)])
    for job in ivf_jobs:
        nlist = job["nlist"]
        index, build_s = timed_build(build_ivf, corpus, nlist)
        for nprobe in job.get("nprobes", [job.get("nprobe")]):
            if nprobe > nlist:
                continue
            index.nprobe = nprobe
            row = measure(index, queries, truth, k, build_s, "ivf", f"nlist={nlist},nprobe={nprobe}", size)
            emit(row)
            print(f"  ivf  {row['params']:<24} recall={row['recall_at_k']:.4f} "
                  f"p50={row['p50_ms']:.3f}ms p95={row['p95_ms']:.3f}ms build={build_s:.1f}s", flush=True)
        del index
        gc.collect()

    # HNSW: one build per (M, efConstruction), then sweep efSearch.
    hnsw_jobs = ([best_only["hnsw"]] if best_only and "hnsw" in best_only
                 else [{"m": m, "ef_construction": efc, "ef_searches": grid["ef_search"]}
                       for m in grid["m"] for efc in grid["ef_construction"]])
    for job in hnsw_jobs:
        m, efc = job["m"], job["ef_construction"]
        index, build_s = timed_build(build_hnsw, corpus, m, efc)
        for ef in job.get("ef_searches", [job.get("ef_search")]):
            index.hnsw.efSearch = ef
            params = f"M={m},efC={efc},efS={ef}"
            row = measure(index, queries, truth, k, build_s, "hnsw", params, size)
            emit(row)
            print(f"  hnsw {row['params']:<24} recall={row['recall_at_k']:.4f} "
                  f"p50={row['p50_ms']:.3f}ms p95={row['p95_ms']:.3f}ms build={build_s:.1f}s", flush=True)
        del index
        gc.collect()

    lsh_jobs = ([best_only["lsh"]] if best_only and "lsh" in best_only
                else [{"nbits": nb} for nb in grid["nbits"]])
    for job in lsh_jobs:
        nbits = job["nbits"]
        index, build_s = timed_build(build_lsh, corpus, nbits)
        row = measure(index, queries, truth, k, build_s, "lsh", f"nbits={nbits}", size)
        emit(row)
        print(f"  lsh  {row['params']:<24} recall={row['recall_at_k']:.4f} "
              f"p50={row['p50_ms']:.3f}ms p95={row['p95_ms']:.3f}ms build={build_s:.1f}s", flush=True)
        del index
        gc.collect()

    return rows


def pick_winners(rows: list[dict], target: float) -> dict:
    """Per index type, the lowest-p95 config that still clears the recall target.
    Falls back to the best-recall config when nothing clears it, so the larger
    sizes still report something rather than silently skipping the index."""
    winners = {}
    for name in ("ivf", "hnsw", "lsh"):
        candidates = [r for r in rows if r["index"] == name]
        if not candidates:
            continue
        qualified = [r for r in candidates if r["recall_at_k"] >= target]
        best = (min(qualified, key=lambda r: r["p95_ms"]) if qualified
                else max(candidates, key=lambda r: r["recall_at_k"]))
        kv = dict(p.split("=") for p in best["params"].split(","))
        if name == "ivf":
            winners["ivf"] = {"nlist": int(kv["nlist"]), "nprobes": [int(kv["nprobe"])]}
        elif name == "hnsw":
            winners["hnsw"] = {"m": int(kv["M"]), "ef_construction": int(kv["efC"]),
                               "ef_searches": [int(kv["efS"])]}
        else:
            winners["lsh"] = {"nbits": int(kv["nbits"])}
    return winners


def report_crossover(all_rows: list[dict], target: float) -> None:
    """The question the whole sweep exists to answer."""
    print(f"\n{'=' * 72}\nCrossover: smallest catalog where the index beats brute-force p95 "
          f"at recall@10 >= {target}\n{'=' * 72}")
    brute = {r["size"]: r["p95_ms"] for r in all_rows if r["index"] == "brute_force"}
    for name in ("ivf", "hnsw", "lsh"):
        hit = None
        for size in sorted(brute):
            qualified = [r for r in all_rows
                         if r["index"] == name and r["size"] == size
                         and r["recall_at_k"] >= target and r["p95_ms"] < brute[size]]
            if qualified:
                best = min(qualified, key=lambda r: r["p95_ms"])
                hit = (size, best)
                break
        if hit:
            size, best = hit
            print(f"  {name:<5} crosses at n={size:>9,}  ({best['params']}, "
                  f"p95 {best['p95_ms']:.3f}ms vs brute {brute[size]:.3f}ms, "
                  f"recall={best['recall_at_k']:.4f})")
        else:
            print(f"  {name:<5} never beats brute force at recall@10 >= {target} within the sizes swept")


def main():
    parser = argparse.ArgumentParser(description="ANN index sweep over scaled city embeddings")
    parser.add_argument("--sizes", type=int, nargs="+", default=[560, 10_000, 100_000, 1_000_000])
    parser.add_argument("--max-size", type=int, default=None, help="skip sizes above this")
    parser.add_argument("--full-grid-max", type=int, default=100_000,
                        help="sizes above this run only the carried-over best config per index")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--recall-target", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    parser.add_argument("--only", nargs="+", choices=["ivf", "hnsw", "lsh"],
                        help="restrict the sweep to these index types")
    parser.add_argument("--ef-search", type=int, nargs="+", default=[16, 32, 64, 128],
                        help="HNSW efSearch grid; raise it to chase recall at large n")
    parser.add_argument("--ef-construction", type=int, nargs="+", default=[40, 200])
    parser.add_argument("--m", type=int, nargs="+", default=[8, 16, 32])
    args = parser.parse_args()

    faiss.omp_set_num_threads(1)  # search timing only; build is re-enabled below

    grid = {
        "nlist": [16, 64, 256, 1024],
        "nprobe": [1, 4, 8, 16, 32],
        "m": args.m,
        "ef_construction": args.ef_construction,
        "ef_search": args.ef_search,
        "nbits": [64, 128, 256, 512],
    }
    if args.only:
        # Emptying the outer driver key skips that index's builds entirely.
        # brute_force always runs: it is the reference the others are scored against.
        for name, driver_key in (("ivf", "nlist"), ("hnsw", "m"), ("lsh", "nbits")):
            if name not in args.only:
                grid[driver_key] = []

    rng = np.random.default_rng(args.seed)
    base = load_base_embeddings()
    queries = build_queries(base, args.n_queries, rng)
    sizes = [s for s in sorted(args.sizes) if args.max_size is None or s <= args.max_size]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists()
    fh = open(args.out, "a", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if new_file:
        csv_writer.writeheader()

    def write_row(row):  # flush per row: a long sweep shouldn't lose everything to a timeout
        csv_writer.writerow(row)
        fh.flush()

    all_rows, winners = [], None
    try:
        for size in sizes:
            print(f"\n{'=' * 72}\nn = {size:,} ({base.shape[1]}-dim, {args.n_queries} queries, k={args.k})\n{'=' * 72}",
                  flush=True)
            corpus = scale_corpus(base, size, rng)
            truth = exact_topk(corpus, queries, args.k)

            best_only = winners if size > args.full_grid_max else None
            if best_only:
                print(f"  (carrying over best config per index from n<={args.full_grid_max:,})", flush=True)

            rows = sweep_size(size, corpus, queries, truth, args.k, grid, write_row, best_only)
            all_rows.extend(rows)

            if size <= args.full_grid_max:
                winners = pick_winners(rows, args.recall_target)

            del corpus, truth
            gc.collect()
    finally:
        fh.close()

    report_crossover(all_rows, args.recall_target)
    print(f"\nraw results appended to {args.out}")


if __name__ == "__main__":
    main()
