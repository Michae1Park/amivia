# Project 1: Embedding-based candidate retrieval

Layer: candidate retrieval (embeddings) — see `docs/architecture.md` §3.

**Problem:**
- Narrow a catalog of thousands of destinations down to a small candidate
  set that's topically relevant to a free-text query, optimizing for recall
  over precision (this is the first funnel stage — later stages filter and
  rank the candidates this produces)
- Content-based (embedding similarity) is one of several retrieval sources a
  production system runs in parallel — `collab_filter` (Project 2) is the
  other one scoped here

**Data:** [Worldwide Travel Cities (Ratings and Climate)](https://www.kaggle.com/datasets/furkanima/worldwide-travel-cities-ratings-and-climate)
- 560 cities, each with a short text description plus structured columns
  (budget_level, and 1-5 tag ratings: culture, adventure, nature, beaches,
  nightlife, cuisine, wellness, urban, seclusion)
- Download the CSV and place it at:
  `data/Worldwide Travel Cities Dataset (Ratings and Climate).csv`
- License: MIT (commercial use permitted, preserve the copyright notice) —
  see `docs/roadmap-to-service.md` §Gap 8 for the caveats
- This catalog is also the base dataset `synthetic_interactions` builds
  synthetic users/interactions on top of, and `filter_constraints` filters
  by its structured columns

**Setup:**
```
pip install sentence-transformers numpy
```

**Usage:**
```
python3 embed_retrieve.py "your travel query"
python3 embed_retrieve.py "your travel query" -k 10   # return top 10 instead of top 5
python3 embed_retrieve.py                              # prompts for a query interactively

python3 batch_test.py                                  # fixed probe queries, all failure modes
python3 ann_benchmark.py                               # the full ANN sweep (~25 min, writes a CSV)
python3 ann_benchmark.py --max-size 10000              # quick version
```
- First run embeds all 560 descriptions with `all-MiniLM-L6-v2` and caches
  the result to `data/description_embeddings.npy`; later runs load the
  cache instead of re-embedding
- `python3 batch_test.py` runs a fixed set of test queries (see below) and
  prints results for all of them in one pass — no arguments needed

**Algorithms to compare:**
- Brute-force dot product over the full embedding matrix — **done**, this is
  what `embed_retrieve.py` currently does (embeddings are unit-normalized,
  so dot product == cosine similarity)
- FAISS IVF (inverted-file index) — **done** (`ann_benchmark.py`)
- HNSW (hierarchical navigable small world graph) — **done**
- LSH (locality-sensitive hashing) — **done**
- All three come from `faiss` (`IndexIVFFlat` / `IndexHNSWFlat` / `IndexLSH`),
  so the comparison isn't confounded by three libraries' differing quality
- At 560 items brute force is already instant; the point of building the ANN
  variants is to feel their recall/latency tradeoff and index-build cost,
  not because this dataset needs them

**Knobs to tune:**
- `nlist` / `nprobe` (IVF)
- `M` / `efConstruction` / `efSearch` (HNSW)
- Embedding model choice (`all-MiniLM-L6-v2` vs. a larger sentence-transformer)

**What `batch_test.py` surfaces:** runs grouped queries and prints results
for each so failure modes are visible side by side, not just described:
- **Paraphrase stability** — do different phrasings of the same intent
  ("relaxing beach vacation" vs. "chill seaside getaway") return overlapping
  cities? Measures result-set overlap directly.
- **Negation** — "quiet town, definitely no nightlife" tends to still surface
  nightlife-heavy cities, because embeddings conflate "no nightlife" with
  "vibrant nightlife" (both mention nightlife). This is the motivating
  example for `filter_constraints` (Project 3).
- **Numeric constraints** — "under $50 a day" and "average July temperature
  above 30°C" aren't reasoned about at all; embeddings have no notion of
  thresholds. Another motivating example for `filter_constraints`.

## Experiment

**Setup:** 560 city descriptions embedded with `all-MiniLM-L6-v2`
(384-dim, unit-normalised), so dot product == cosine. Brute force is the
reference: its exact top-k *is* the ground truth the ANN variants are scored
against.

**Conditions:** brute force · FAISS IVF (sweep `nlist`, `nprobe`) · HNSW
(sweep `M`, `efConstruction`, `efSearch`) · LSH (sweep bits, tables).

**Metrics:** recall@10 against exact brute-force top-10; query latency p50/p95;
index build time; index memory.

**Interpretation:** at 560 items brute force is expected to win outright —
which makes the *crossover point* the actual question, not the ranking at this
size. Find it by replicating the catalog to ~10k / 100k / 1M synthetic items
(perturbed copies of the real embeddings) and re-running the sweep, then report
the catalog size at which each index overtakes brute force on latency at
recall@10 ≥ 0.95. That extrapolation is what tells Phase 2 whether per-city
POI retrieval needs an index.

## Results — retrieval quality

- Brute force returns topically sound candidates: paraphrases of one intent
  ("relaxing beach vacation" / "chill seaside getaway") return heavily
  overlapping sets.
- Negation fails outright — *"quiet town, definitely no nightlife"* still
  surfaces nightlife-heavy cities.
- Numeric constraints are not reasoned about at all — *"under $50 a day"*,
  *"July average above 30°C"*.

`filter_constraints` (Project 3) has since quantified this: **79% of the
unfiltered top-10 violates the query's own stated constraint**, and for *"quiet
town, definitely no nightlife"* it is 10 out of 10.

## Results — ANN sweep

Best config per index at each size (lowest p95 that still clears recall@10 ≥
0.95; where nothing clears it, the highest-recall config is shown instead).
200 queries, k=10, single-threaded search on both sides. Raw sweep in
`../data/ann_benchmark.csv`.

| n | index | config | recall@10 | p50 ms | p95 ms | build s | MB |
|---|---|---|---|---|---|---|---|
| 560 | brute | — | 1.000 | 0.053 | 0.061 | 0 | 0.9 |
| 560 | ivf | nlist=16, nprobe=8 | 0.978 | 0.029 | 0.038 | 0.1 | 0.9 |
| 560 | hnsw | M=32, efC=200, efS=16 | 0.953 | 0.025 | 0.031 | 0.0 | 0.9 |
| 560 | lsh | nbits=512 | *0.396* | 0.031 | 0.051 | 0.4 | 0.8 |
| 10k | brute | — | 1.000 | 1.230 | 1.597 | 0 | 15.4 |
| 10k | ivf | nlist=256, nprobe=16 | 0.956 | 0.216 | 0.267 | 0.4 | 15.8 |
| 10k | hnsw | M=32, efC=200, efS=16 | 0.954 | 0.058 | 0.087 | 0.7 | 18.1 |
| 10k | lsh | nbits=512 | *0.369* | 0.069 | 0.083 | 0.4 | 1.4 |
| 100k | brute | — | 1.000 | 12.179 | 14.838 | 0 | 153.6 |
| 100k | ivf | nlist=1024, nprobe=8 | 0.972 | 0.204 | 0.342 | 5.1 | 156.0 |
| 100k | hnsw | M=32, efC=40, efS=128 | 0.955 | 0.447 | 0.840 | 6.2 | 180.8 |
| 100k | lsh | nbits=512 | *0.247* | 0.606 | 0.942 | 1.5 | 7.2 |
| 1M | brute | — | 1.000 | 115.079 | 124.227 | 0 | 1536.0 |
| 1M | ivf | nlist=1024, nprobe=8 | 0.986 | 1.170 | 2.173 | 63.0 | 1545.6 |
| 1M | hnsw | M=32, efC=40, efS=1024 | 0.963 | 2.583 | 4.547 | 108.5 | 1808.1 |
| 1M | lsh | nbits=512 | *0.134* | 5.211 | 6.141 | 11.0 | 64.8 |

*Italic recall = never reached 0.95 at any config in the grid.* HNSW rows come
from a corrected re-run (see below); brute-force rows are from the main sweep.

**The pre-specified prediction was wrong.** The Interpretation above expected
brute force to "win outright" at 560. It does not: IVF and HNSW both beat it on
p95 even at 560 (0.038 / 0.031 ms vs 0.061 ms). Building an index is already
"free" latency-wise at the smallest size.

**But the raw crossover is the wrong question, and answering it exposed why.**
Embedding the query costs **9.63 ms p50 / 11.17 ms p95** — a fixed toll every
query pays before the index is ever touched. Against that, search time at small
n is a rounding error:

| n | brute end-to-end | best ANN end-to-end | speedup |
|---|---|---|---|
| 560 | 9.68 ms | 9.66 ms | 1.00x |
| 10k | 10.86 ms | 9.70 ms | 1.12x |
| 100k | 21.81 ms | 9.83 ms | 2.2x |
| 1M | 124.71 ms | 10.80 ms | **11.6x** |

**So the answer Phase 2 should use is ~100k, not 560.** Below that, an index is
technically faster and practically pointless — at 560 it saves 30 microseconds
on a 9.7 ms query, 0.3%. At 100k, brute-force search finally costs more than
the embedding call and the index halves end-to-end latency; by 1M it is the
difference between 125 ms and 11 ms. Per-city POI retrieval needs an index only
if a city's POI count pushes the searched set toward six figures.

**IVF is the one to reach for.** It clears 0.95 recall at every size, scales
best (0.986 recall at 2.17 ms p95 at 1M — 57x faster than brute force), and
builds in 63 s at 1M.

**HNSW is fastest at small n, and loses to IVF at large n.** It is the quickest
index at 560 and 10k (0.031 / 0.087 ms p95), but at 1M it needs `efSearch=1024`
to reach 0.963 recall, by which point it is slower than IVF (4.55 ms vs 2.17 ms
p95), takes longer to build (109 s vs 63 s) and uses more memory (1808 MB vs
1546 MB). IVF wins the top end on all four axes.

**A finding worth more than the ranking: raising `efConstruction` made HNSW
*worse*, and it took a bug hunt to believe it.** The first sweep showed recall
falling as `efSearch` rose — impossible for a correct index — which turned out
to be two different builds being compared. Isolating it on one corpus at 100k:

| efConstruction | efS=16 | efS=128 | efS=1024 | build |
|---|---|---|---|---|
| 40 | 0.719 | 0.955 | **0.991** | 6.2 s |
| 200 | 0.659 | 0.781 | *0.803* | 36.2 s |

Six times the build time for a permanently worse graph. The cause is the
*corpus*, not HNSW: the scale-up perturbs 560 real vectors, producing 560 tight
clusters, and HNSW's neighbour-diversification heuristic prunes more
aggressively the more near-identical candidates it sees during construction —
leaving clusters well connected internally and poorly connected to each other.
Loosening the clusters removes the effect entirely (at noise 0.3, efC=200 scores
0.9895 against efC=40's 0.9870 — the expected ordering, restored).

Two things follow. First, the 100k/1M HNSW rows above use `efC=40` deliberately.
Second, and more usefully: **`efConstruction` is not a "higher is better" knob on
clustered data**, which is exactly what a real catalog of cities-by-type looks
like. That is a tuning trap worth carrying into Phase 2, and it would have been
invisible on uniformly random vectors.

*(Raising the noise is not the fix, tempting as the clean numbers look: at 0.3
the perturbation outweighs the unit-norm base vector roughly 6:1, so the corpus
degenerates toward uniform random — the case this scale-up exists to avoid.)*

**LSH fails, and fails worse as the catalog grows** — 0.396 recall at 560 down
to 0.134 at 1M, at its widest setting (512 bits). Random-hyperplane LSH needs
far more bits to preserve neighbourhoods in 384 dimensions, and by 1M it is
also *slower* than IVF (6.1 ms vs 2.2 ms) while retrieving mostly wrong
neighbours. Its one genuine advantage is memory: 64.8 MB at 1M against 1536 MB
for the flat index, 24x smaller, because it stores binary codes instead of
vectors.

**ANN buys latency, not memory.** IVF and HNSW both store the full vectors
alongside their index structures, so at 1M they cost *more* than brute force
(1546 MB and 1808 MB vs 1536 MB). Only LSH shrinks the footprint, and it pays
for that in recall. Shrinking memory without destroying recall means product
quantization (IVFPQ), which this sweep did not cover.

**Standing caveat:** only the first 560 vectors are real. The larger corpora are
perturbed copies of them (Gaussian noise, renormalised), chosen because
uniformly random 384-dim vectors are near-equidistant and would flatter every
index. The *shape* of the tradeoff is what transfers; the absolute recall
figures describe this synthetic distribution, not a real million-city catalog.

**What the negation failure means:** it is representational, not a ranking
wobble. "No nightlife" and "vibrant nightlife" sit close in embedding space
because both are *about* nightlife, so no choice of `k` or embedding model
fixes it. That is the whole justification for `filter_constraints` (Project 3)
existing as a separate deterministic layer, and it is why the fix belongs
outside the retriever rather than inside it.

**Status:** done
- Brute-force baseline: done
- ANN variants (FAISS IVF, HNSW, LSH): done — `ann_benchmark.py`, swept to 1M
- `batch_test.py`: done, the tool for eyeballing negation/numeric/paraphrase
  failure modes
- Not covered: product quantization (IVFPQ), the one option that would cut
  memory without LSH's recall collapse — worth a follow-up if footprint ever
  becomes the binding constraint rather than latency.
