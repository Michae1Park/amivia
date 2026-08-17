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
- FAISS IVF (inverted-file index) — not yet built
- HNSW (hierarchical navigable small world graph) — not yet built
- LSH (locality-sensitive hashing) — not yet built
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

**Results — partial.** Retrieval quality is measured; the ANN sweep is not
built.

- Brute force returns topically sound candidates: paraphrases of one intent
  ("relaxing beach vacation" / "chill seaside getaway") return heavily
  overlapping sets.
- Negation fails outright — *"quiet town, definitely no nightlife"* still
  surfaces nightlife-heavy cities.
- Numeric constraints are not reasoned about at all — *"under $50 a day"*,
  *"July average above 30°C"*.

**What the negation failure means:** it is representational, not a ranking
wobble. "No nightlife" and "vibrant nightlife" sit close in embedding space
because both are *about* nightlife, so no choice of `k` or embedding model
fixes it. That is the whole justification for `filter_constraints` (Project 3)
existing as a separate deterministic layer, and it is why the fix belongs
outside the retriever rather than inside it.

**Status:** in progress
- Brute-force baseline: done
- ANN variants (FAISS IVF, HNSW, LSH): not yet built
- `batch_test.py`: done, currently the main tool for eyeballing negation/
  numeric/paraphrase failure modes
