# Preparation notes

Background concepts for `README.md`'s Preparation checklist — not the lab's
findings, so there's nothing to spoil by reading this first. View this in a Markdown renderer that supports LaTeX (GitHub, VS Code preview) — the equations below are raw `$...$` / `$$...$$` math and will look like noise in a plain-text viewer. Check yourself
against it, don't treat it as a substitute for actually knowing the material.

| # | Concept |
|---|---|
| 1 | [Cosine similarity = dot product, when normalized](#1-cosine-similarity--dot-product-when-normalized) |
| 2 | [Retrieval metrics: recall/precision/hit_rate/MRR@k](#2-retrieval-metrics-recallprecisionhit_ratemrrk) |
| 3 | [IVF / HNSW / LSH — what each trades away](#3-ivf--hnsw--lsh--what-each-trades-away) |
| 4 | [BM25](#4-bm25) |
| 5 | [`MultipleNegativesRankingLoss`](#5-multiplenegativesrankingloss) |
| 6 | [`LABELLED_QUERIES`'s predicate schema](#6-labelled_queriess-predicate-schema) |

---

## 1. Cosine similarity = dot product, when normalized

**Plain English:** cosine similarity measures the *angle* between two
vectors, ignoring their length. The dot product measures both angle *and*
length. Strip the length out of both vectors first (normalize to length 1),
and the two formulas become identical — not close, identical.

**Equation:**

$$\cos(a,b) = \frac{a \cdot b}{\|a\|\,\|b\|}$$

If $\|a\| = \|b\| = 1$ (both unit-normalized), the denominator is just $1$, so:

$$\cos(a,b) = a \cdot b$$

**Worked example:** let $a = (0.6, 0.8)$ and $b = (0.8, 0.6)$ — both already
unit length ($0.6^2+0.8^2=1$).

| | Dot product | Cosine formula |
|---|---|---|
| Calculation | $0.6 \times 0.8 + 0.8 \times 0.6$ | $\dfrac{0.96}{1 \times 1}$ |
| Result | $0.96$ | $0.96$ |

Same number, because the normalization already happened.

**In the code:** `content_filter.py` calls
`model.encode(..., normalize_embeddings=True)` once, then scores every query
with a plain `embeddings @ query_vec`. It isn't cutting a corner — the
division every cosine-similarity call would otherwise do is just dividing by
1, so skipping it changes nothing.

---

## 2. Retrieval metrics: recall/precision/hit_rate/MRR@k

**Plain English:** `relevance_eval.py` and `retriever_eval.py` print four of
these per row, and it's easy to see the columns without ever pinning down
what each one actually asks. All four look at the same thing — the top-$k$
results against a known relevant set — from different angles.

**Equations:**

$$\text{recall@}k = \frac{|\,\text{relevant} \cap \text{top-}k\,|}{|\,\text{relevant}\,|} \qquad \text{precision@}k = \frac{|\,\text{relevant} \cap \text{top-}k\,|}{k}$$

$$\text{hit\_rate@}k = \big[\, |\,\text{relevant} \cap \text{top-}k\,| > 0 \,\big] \qquad \text{MRR@}k = \frac{1}{\text{rank of first relevant hit}}\ \big(0 \text{ if none in top-}k\big)$$

**Worked example:** 4 cities are truly relevant to a query — $\{A,B,C,D\}$,
out of a 560-city catalog. The retriever's top-5 is $[A, X, B, Y, Z]$ — 2
hits, first one at rank 1.

| Metric | Calculation | Result |
|---|---|---|
| recall@5 | $2 / 4$ | $0.5$ — half of what exists was found |
| precision@5 | $2 / 5$ | $0.4$ — 2 of the 5 slots returned were relevant |
| hit_rate@5 | at least one hit? | $1$ — found *something* |
| MRR@5 | $1/\text{rank}(A)$ | $1/1 = 1.0$ — first hit came at rank 1 |

**Why four metrics and not one:** `hit_rate` only asks "found anything at
all" (generous, saturates fast); `recall` asks "how complete"; `precision`
asks "how much of what came back was junk"; `MRR` cares only about *how
early* the first hit landed, ignoring everything found after it. A retriever
can score well on one and poorly on another — that's the point of reading
all four rather than picking a favorite.

**Two different "ground truths" are used in this project for `recall@k` — don't conflate them:**

| Script | What "relevant" means | What `recall@k` is actually asking |
|---|---|---|
| `ann_benchmark.py` | Brute force's *exact* top-$k$ | "How much of the true nearest-neighbor set did the approximate index reproduce?" |
| `relevance_eval.py` / `retriever_eval.py` | Every city satisfying `filter_constraints`'s hand-labelled predicates | "How much of what's *actually relevant* did retrieval surface?" |

**NDCG@k** — also printed by both scripts, but it needs *graded* relevance
(a save outranks a click) to be worth computing separately from recall.
`collab_filter/PREPARATION_NOTES.md` §7 has the equation; nothing here
duplicates it, since content_filter's ground truth (§2's table above) is
binary — every predicate-satisfying city counts equally, so NDCG collapses
toward `recall`'s ordering here in a way it doesn't for collab_filter's
graded save/click labels.

---

## 3. IVF / HNSW / LSH — what each trades away

All three exist to avoid comparing a query against every single vector. A
brute-force scan is always exact, at $O(n)$ per query. Each of these buys
speed by giving up some of that exactness — just differently:

| Method | Mechanism | Tunable knob | What it trades away |
|---|---|---|---|
| **IVF** | Pre-cluster all vectors into `nlist` groups (k-means); a query only searches the `nprobe` nearest clusters | `nlist`, `nprobe` | Exhaustiveness — a true neighbor in an unsearched cluster is simply missed |
| **HNSW** | A multi-layer graph linking each vector to its approximate near neighbors; a query greedily walks the graph, coarse layer to fine | `efConstruction` (build quality), `efSearch` (search effort) | Greedy search can get stuck in a locally-good-but-not-best region |
| **LSH** | Hash vectors so *similar ones probably* land in the same bucket; only compare within the query's bucket | number of hash bits/tables | Recall — "probably" is doing real work, and it gets worse in high dimensions |

**The one equation worth knowing here** is LSH's actual promise. For a
single random-hyperplane hash (what FAISS's `IndexLSH` uses), the
probability two vectors land in the same bucket is:

$$P[\text{same bucket}] = 1 - \frac{\theta(a,b)}{\pi}$$

where $\theta(a,b)$ is the angle between $a$ and $b$, in radians. Closer
vectors (smaller angle) → higher collision probability — but it's a
*probability*, never a guarantee, which is exactly why LSH's recall
degrades as dimensionality rises and $\theta$ stops being as informative.

---

## 4. BM25

**Plain English:** score a document by how many query terms it shares,
weighting rare shared terms more than common ones, and correcting for
document length so a long document doesn't win just by containing more
words overall. No meaning, no vectors — pure term statistics.

**Equation:**

$$\text{BM25}(q,d) = \sum_{t \,\in\, q} \text{IDF}(t) \cdot \frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\left(1 - b + b\dfrac{|d|}{\text{avgdl}}\right)}$$

$$\text{IDF}(t) = \ln\!\left(\frac{N - n(t) + 0.5}{n(t) + 0.5} + 1\right)$$

| Symbol | Meaning |
|---|---|
| $f(t,d)$ | how many times term $t$ appears in document $d$ |
| $\lvert d \rvert$, $\text{avgdl}$ | this document's length, and the corpus's average document length |
| $N$ | total number of documents in the corpus |
| $n(t)$ | number of documents containing term $t$ |
| $k_1$, $b$ | tuning constants — $k_1$ caps how much repeated terms keep adding score; $b$ controls how strongly length is penalized |

**Why it matters for `hybrid_search.py`:** this score is **unbounded** — its
range depends on the corpus (how rare terms are, how long documents run) —
unlike cosine similarity, which is mathematically boxed into $[-1, 1]$. Add
or average an unbounded BM25 score with a bounded cosine score, and whichever
one happens to have the larger raw magnitude silently wins. That's exactly
why fusion is done by **rank position** (RRF) instead of by score.

---

## 5. `MultipleNegativesRankingLoss`

**Plain English:** in a batch of $(query, positive)$ pairs, every *other*
pair's positive doubles as a free negative for this one. Train the model so
each anchor scores its true positive higher than all $N-1$ other items in
the batch — like an $N$-way classification problem where the correct class
changes every batch, and free.

**Equation** (InfoNCE / softmax cross-entropy over similarities):

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp\!\big(\text{sim}(a_i, p_i)/\tau\big)}{\displaystyle\sum_{j=1}^{N} \exp\!\big(\text{sim}(a_i, p_j)/\tau\big)}$$

| Symbol | Meaning |
|---|---|
| $a_i$, $p_i$ | the $i$-th anchor (query embedding) and its true positive (item embedding) |
| $\text{sim}(\cdot,\cdot)$ | cosine similarity between two embeddings |
| $\tau$ | temperature — lower $\tau$ makes the softmax sharper (more confident, less forgiving) |
| $N$ | batch size — also the number of negatives each anchor gets, for free |

**The catch, already flagged in `finetune_retriever.py`:** at 560 catalog
items and a batch of 16-32, the *same* item can appear twice in one batch as
two different anchors' true positive. When that happens, the loss
momentarily treats a genuine positive as one of the $N-1$ "negatives" for the
other anchor — a false negative baked into the method at small catalog
sizes and small batches, not a bug in this implementation specifically.

---

## 6. `LABELLED_QUERIES`'s predicate schema

15 hand-labelled travel queries in `filter_constraints/constraints.py`, each
paired with the hard constraint(s) a human reads out of it:

| Predicate | Meaning | Example query | Instantiation |
|---|---|---|---|
| `TagAtLeast(tag, min)` | tag rating $\geq$ `min` | "cultural city trip" | `TagAtLeast("culture", 4)` |
| `TagAtMost(tag, max)` | tag rating $\leq$ `max` — how **negation** is expressed | "no nightlife" | `TagAtMost("nightlife", 2)` |
| `BudgetAtMost(level)` | budget tier at or below `level` | "under $50 a day" | `BudgetAtMost("Budget")` |
| `MonthTempAtLeast(month, °C)` | that month's avg temp $\geq$ `°C` | "warm in July, above 25°" | `MonthTempAtLeast(7, 25.0)` |
| `MonthTempAtMost(month, °C)` | that month's avg temp $\leq$ `°C` | "cool July, below 20°" | `MonthTempAtMost(7, 20.0)` |

Queries fall into four groups: negation, numeric-budget, numeric-temperature,
and combined (several predicates at once). `relevance_eval.py` and
`retriever_eval.py` both use this list as ground truth (see §2's table
above); `llm_wrapper/intent_parsing`'s out-of-distribution eval set extends
this same list (`ood_queries.py`) rather than inventing a separate schema.
