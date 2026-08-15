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

**Status:** in progress
- Brute-force baseline: done
- ANN variants (FAISS IVF, HNSW, LSH): not yet built
- `batch_test.py`: done, currently the main tool for eyeballing negation/
  numeric/paraphrase failure modes
