# Project 1: Content-based candidate retrieval

Layer: candidate retrieval (embeddings) — see `docs/architecture.md` §3.
Narrows a catalog of thousands of destinations down to a small candidate set
topically relevant to a free-text query, optimizing recall over precision —
the funnel's first stage. `collab_filter` (Project 2) is the other retrieval
source scoped in parallel; `filter_constraints` (Project 3) is the layer this
one's failures motivate.

## Preparation

Check yourself against [`PREPARATION_NOTES.md`](PREPARATION_NOTES.md) — background
concepts, not the lab's findings, so there's nothing to spoil by reading it first.

- [ ] Know why unit-normalized embeddings make dot product == cosine similarity
- [ ] Know what `recall@k` measures and why it needs a reference ranking to score against
- [ ] Know, at a sketch level, what IVF, HNSW, and LSH each do differently from a full scan (don't need the math — just what each one is willing to trade away)
- [ ] Know what BM25 scores (term frequency / inverse document frequency, length-normalized) and why that score isn't on the same scale as cosine similarity
- [ ] Know what `MultipleNegativesRankingLoss` does at a sketch level — in-batch negatives, no manual negative mining — before running the fine-tune
- [ ] Skim `filter_constraints/constraints.py`'s `LABELLED_QUERIES` — this project's eval set borrows it

## Objectives

- Feel where embedding similarity is good enough, and where it fails outright — a modeling-accuracy question
- Feel when an ANN index is worth its complexity, and when brute force is fine — a computation question
- Feel when lexical retrieval (BM25) beats semantic retrieval, and when fusing both beats either alone
- Feel whether fine-tuning a retriever on your own click data is worth it over swapping in a bigger pretrained model
- Practice building a quantitative eval instead of eyeballing results, and treating a small eval set's numbers as estimates with uncertainty, not exact scores

## Questions to answer

**Modeling accuracy:**
1. Does it retrieve relevant results for a well-formed query?
2. What can it never express, regardless of embedding model or `k` — and why is that a representational limit rather than something tuning fixes?
3. How much worse is it than random guessing on exactly the queries it's blind to — and does that mean the approach is broken in general?

**Computation:**
4. At this catalog's scale, does an index even need to exist — what actually dominates per-query cost?
5. Among IVF / HNSW / LSH, which wins on recall, latency, build time, and memory — is any one of them best on all four?
6. Which tuning knob behaves opposite to intuition on real (clustered) data, and why does that matter for a production catalog?

**Retrieval method choice:**
7. Where does BM25 beat dense embeddings, and where does it lose, on the same probe queries as Question 2?
8. Does RRF hybrid fusion actually beat *both* single-method retrievers, or just land between them?
9. Does swapping the embedding model (MiniLM vs. bge-base) change the negation/numeric failure at all, or is that failure model-independent?
10. Does fine-tuning the retriever on your own click data generalize past the query template it trained on, or does it just overfit to that template's shape?
11. When two retrievers' scores differ by a few points on 15 labelled queries, is that a real difference or noise?

**Before you touch the code, write down a guess for three of these** — the
catalog size where an index first beats brute force end-to-end; whether
embedding retrieval beats random ranking on the negation/numeric queries;
and whether you expect BM25 to handle "no nightlife" any better or worse
than embeddings do, and why. Nothing here checks that guess for you; that's
the point of writing it down somewhere else first.

## Background

*Definitions and equations live in [`PREPARATION_NOTES.md`](PREPARATION_NOTES.md) — this is just orientation.*

- **Catalog:** [Worldwide Travel Cities (Ratings and Climate)](https://www.kaggle.com/datasets/furkanima/worldwide-travel-cities-ratings-and-climate),
  560 cities with a text description plus `budget_level` and nine 1-5 tag
  ratings (culture, adventure, nature, beaches, nightlife, cuisine, wellness,
  urban, seclusion). Download it, then save/rename it to **`data/cities.csv`**
  — that's the exact path `content_filter.py`'s `DATA_PATH` reads; the
  original Kaggle filename won't be found.
- **Embedding models:** two are cached — `all-MiniLM-L6-v2` (384-dim) and
  `BAAI/bge-base-en-v1.5` (768-dim, asymmetric — see `QUERY_INSTRUCTION`) —
  as `data/description_embeddings*.npy`. `retriever_eval.py` auto-detects
  every cache it finds, so a third model is just: point `MODEL_NAME` at it,
  run once, re-run the eval.
- **ANN candidates** (all via `faiss`, so no library-quality confound):

  | Method | Searches | Knob |
  |---|---|---|
  | IVF | only the `nprobe` nearest of `nlist` pre-built clusters | `nlist`, `nprobe` |
  | HNSW | a greedy walk of a multi-layer neighbor graph | `efConstruction`, `efSearch` |
  | LSH | only the query's own hash bucket | number of hash bits |

- **BM25** ranks by shared, statistically-rare vocabulary — no meaning, no
  vectors. **RRF** fuses BM25 with dense retrieval by rank position, not raw
  score, since the two scores aren't on a comparable scale.
- **Vocabulary:** `recall@k` / `MRR` — see PREPARATION_NOTES.md; p50/p95 =
  median/tail latency; a bootstrap CI estimates how much a metric would
  wobble on a different sample of queries the same size.
- **Terminology:** this project is semantic search (query → items), not
  "content-based filtering" in the classic recsys sense (liked items → a
  user profile → similar items, no query). Both get called "content-based"
  loosely — know which one you mean.

## Procedure

**Setup** (run from inside this directory; the venv is shared across all
three `candidate_generation` projects, so skip the first two lines if you
already created one for another):
```
python3 -m venv ../.venv               # once, from anywhere in candidate_generation/
source ../.venv/bin/activate           # re-run this in every new shell
pip install -r ../../requirements.txt
```
System Python won't have these packages, and on many systems `pip install`
outside a venv fails outright (externally-managed-environment) — this isn't
optional setup, it's the actual first step.

**Part a — sanity check.** Run `python3 content_filter.py "<a query of your
own>"` a few times. Do the returned cities look topically plausible to you?

**Part b — probe the failure modes.** Run `python3 batch_test.py`. It prints
three grouped query sets side by side:
- Paraphrase stability — do different phrasings of one intent return
  overlapping cities?
- Negation — does "no nightlife" suppress nightlife-heavy results, or does
  mentioning nightlife at all pull them back in regardless of stated polarity?
- Numeric constraints — can it reason about "under $50 a day" or "July above
  30°C" at all, given the descriptions never state either value?

Record what you observe for each, against your Question 1/2 answers.

**Part c — quantify it.** Run `python3 relevance_eval.py`. It scores
embedding retrieval against a `random` baseline, using `filter_constraints`'s
`LABELLED_QUERIES` as ground truth, at k ∈ {5, 10, 20, 50} on hit_rate,
recall, precision, and ndcg. Where does embedding retrieval beat random?
Where doesn't it, and does that match what Part b predicted?

**Part d — benchmark the indexes.** Run `python3 ann_benchmark.py
--max-size 10000` (quick; the full sweep to 1M takes ~25 min via
`ann_benchmark.py` with no flag). For each of brute force / IVF / HNSW / LSH,
record recall@10, p50/p95 latency, build time, and memory.

**Part e — find the crossover.** Query embedding has a fixed cost per call —
time it. Using that plus Part d's numbers, at what catalog size does an
index's *end-to-end* latency (embedding + search) first beat brute force's?
Does it match your pre-registered guess?

**Part f — the tuning trap.** Sweep HNSW's `efConstruction` (e.g. 40 vs. 200)
at a fixed `efSearch`, on the same corpus size. Does higher `efConstruction`
always improve recall? If it doesn't, what does that tell you about the
assumption "more graph-building effort is never worse," and why might a
catalog of real cities-by-type break that assumption where uniformly random
vectors wouldn't?

**Part g — add the lexical baseline.** Run `python3 bm25_search.py "<query>"`
on a few of Part b's probe queries. Does BM25 fall into the same negation
trap embeddings do ("no nightlife" still matching on "nightlife")? Why might
a term-frequency method fail the same way as a meaning-based one here, or
differently?

**Part h — fuse them.** Run `python3 hybrid_search.py "<query>"`, then
`python3 retriever_eval.py --k-values 10 --n-bootstrap 200` for a quick pass
(drop both flags for the full run) to score embedding / BM25 / hybrid /
random side by side, with 95% CIs. Does hybrid actually beat *both*
single-method retrievers on any metric, or just land between them?

**Part i — swap the embedding model.** `retriever_eval.py` picks up every
`data/description_embeddings*.npy` cache automatically. If both MiniLM's and
bge-base's are cached (or you generate a second one by pointing
`content_filter.MODEL_NAME` elsewhere and running once), compare them
directly. Does the negation/numeric failure from Part b change at all with a
bigger or different model?

**Part j — fine-tune the retriever.** Needs `synthetic_interactions`'s data
generated first (`cd ../../synthetic_interactions && python3 generate.py`,
skip if you already did this for `collab_filter`). Run
`python3 finetune_retriever.py --max-pairs 500 --epochs 1` for a quick pass
(drop both flags for the full run — expect roughly 10-15 minutes on a
default CPU with the ~100k click/save pairs this generates, though hardware
varies enough that this is a ballpark, not a promise; the quick pass runs in
seconds). It prints ID (held-out session-query) and OOD (`LABELLED_QUERIES`)
scores before and after fine-tuning. Before you look at the OOD numbers:
given `sessions.csv`'s query template never contains negation or a numeric
threshold, what's your prediction for the OOD delta? Did fine-tuning move
the ID score, the OOD score, both, or neither?

## Deliverables

1. Your own answer to each of the eleven questions above, in your own numbers
2. The recall/p50/p95/build/memory table for brute force vs. IVF vs. HNSW vs.
   LSH, at whatever sizes you ran
3. The crossover catalog size you found in Part e, and whether it matched
   your pre-registered guess
4. One paragraph on the Part f result and what it implies about tuning ANN
   indexes on non-random, clustered data
5. A one-line recommendation: which index would you reach for by default,
   and under what condition would you reach for a different one instead
6. The embedding / BM25 / hybrid comparison table from Part h, with CIs, and
   a note on whether any observed difference actually clears the CI overlap
7. Your embedding-model comparison from Part i (if you ran a second model),
   and whether it changed the negation/numeric failure
8. Before/after fine-tuning numbers from Part j on both ID and OOD, and your
   explanation for the gap (or lack of one) between them
9. One paragraph distinguishing what this project builds from "content-based
   filtering" in the classic recsys sense, and why the distinction matters
   when someone asks you about it

## Optional / stretch (not built here)

Marked optional because they're real gaps but not essential to the six core
questions above — worth knowing they exist, not worth blocking on:

- **Quantization/compression** (product quantization / IVFPQ, int8, or
  Matryoshka-style truncated embeddings) — the option that would shrink
  memory footprint without LSH's recall collapse. Would mean adding a PQ arm
  to `ann_benchmark.py`.
- **Filtered vector search as its own experiment** — `filter_constraints`
  (Project 3) explores whether filtering after retrieval can ever fully
  recover, given a fixed candidate pool (its Procedure Part e). If the
  answer there is "no," the production fix is pushing the filter into the
  index itself (FAISS `IDSelector`, or partitioning by `budget_level`)
  rather than post-filtering — not built as a dedicated ANN-level experiment
  here.
- **A larger hand-labelled eval set** — Part h's bootstrap CIs would shrink
  with more than 15 labelled queries; building that set is its own
  time investment, not a code change.

## Notes

- `content_filter.py`, `batch_test.py`, `ann_benchmark.py`, `relevance_eval.py`,
  `bm25_search.py`, `hybrid_search.py`, `retriever_eval.py`, and
  `finetune_retriever.py` are all implemented and runnable — this lab is
  about running them and interpreting the output, not writing new code.
- New dependencies beyond the original setup: `rank_bm25`, `datasets`,
  `accelerate` (all in `scratch/requirements.txt`).
- `finetune_retriever.py` defaults to `all-MiniLM-L6-v2` for CPU-feasible
  training; swap `--model` for `BAAI/bge-base-en-v1.5` if you have a GPU.
- Not built (see Optional above): quantization/compression, filtered vector
  search as its own experiment, free-text nuance the catalog's tag columns
  don't capture (e.g. "romantic") — no ground truth in this repo grades that.
- A prior write-up with actual measured numbers (the original ANN sweep,
  predating the BM25/hybrid/fine-tuning arms) exists in this file's git
  history, if you want to check your Deliverables against it after — not
  before.
