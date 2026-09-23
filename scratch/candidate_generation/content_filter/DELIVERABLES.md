# Project 1: Content-based candidate retrieval — Deliverables

Fill this in as you go through `README.md`'s Procedure, not after finishing
it. Write a prediction before you run the relevant script, and your answer
right after — that gap is the point.

## Pre-registered predictions (before touching any code)

- Catalog size where an ANN index first beats brute force end-to-end:
- Do you expect embedding retrieval to beat random ranking on the negation/numeric queries? Why:
- Do you expect BM25 to handle "no nightlife" better, worse, or the same as embeddings, and why:

## Questions

### 1. Does it retrieve relevant results for a well-formed query?
**Prediction:**

**Answer (Part a/b):**

### 2. What can it never express, regardless of embedding model or `k` — and why is that a representational limit rather than something tuning fixes?
**Prediction:**

**Answer (Part b):**

### 3. How much worse is it than random guessing on exactly the queries it's blind to — and does that mean the approach is broken in general?
**Prediction:**

**Answer (Part c):**

### 4. At this catalog's scale, does an index even need to exist — what actually dominates per-query cost?
**Prediction:**

**Answer (Part d/e):**

### 5. Among IVF / HNSW / LSH, which wins on recall, latency, build time, and memory — is any one of them best on all four?
**Prediction:**

**Answer (Part d):**

### 6. Which tuning knob behaves opposite to intuition on real (clustered) data, and why does that matter for a production catalog?
**Prediction:**

**Answer (Part f):**

### 7. Where does BM25 beat dense embeddings, and where does it lose, on the same probe queries as Question 2?
**Prediction:**

**Answer (Part g):**

### 8. Does RRF hybrid fusion actually beat *both* single-method retrievers, or just land between them?
**Prediction:**

**Answer (Part h):**

### 9. Does swapping the embedding model (MiniLM vs. bge-base) change the negation/numeric failure at all, or is that failure model-independent?
**Prediction:**

**Answer (Part i):**

### 10. Does fine-tuning the retriever on your own click data generalize past the query template it trained on, or does it just overfit to that template's shape?
**Prediction:**

**Answer (Part j):**

### 11. When two retrievers' scores differ by a few points on 15 labelled queries, is that a real difference or noise?
**Prediction:**

**Answer (Part h):**

## Deliverables

### 1. Your own answer to each of the eleven questions above
See Questions section — filled in above.

### 2. Recall/p50/p95/build/memory table for brute force vs. IVF vs. HNSW vs. LSH

| n | index | config | recall@10 | p50 ms | p95 ms | build s | MB |
|---|---|---|---|---|---|---|---|
| | brute | — | | | | | |
| | ivf | | | | | | |
| | hnsw | | | | | | |
| | lsh | | | | | | |

(add more rows if you swept multiple catalog sizes)

### 3. Crossover catalog size, and whether it matched your pre-registered guess

**Crossover size found:**

**Matched pre-registered guess?**

### 4. Part f result and what it implies about tuning ANN indexes on non-random, clustered data

### 5. One-line recommendation: default index, and when you'd reach for a different one

### 6. Embedding / BM25 / hybrid comparison table (Part h), with CIs

| model | k | hit_rate | recall | precision | ndcg | mrr |
|---|---|---|---|---|---|---|
| embedding | | | | | | |
| bm25 | | | | | | |
| hybrid | | | | | | |
| random | | | | | | |

**Does any observed difference actually clear the CI overlap?**

### 7. Embedding-model comparison (Part i), and whether it changed the negation/numeric failure

### 8. Before/after fine-tuning numbers (Part j), ID and OOD, and your explanation for the gap (or lack of one)

| | ID hit_rate | ID mrr | OOD hit_rate | OOD mrr |
|---|---|---|---|---|
| before fine-tuning | | | | |
| after fine-tuning | | | | |

**Explanation:**

### 9. One paragraph: what this project builds vs. "content-based filtering" in the classic recsys sense, and why the distinction matters

## Status

- [ ] Pre-registered predictions written, before running anything
- [ ] All 11 questions answered
- [ ] All 9 deliverables filled in
- [ ] Checked against git history (only after finishing each item above)
