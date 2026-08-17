# Project 8: Reranking for diversity

Layer: reranking — see `docs/architecture.md` §6.

**Problem:** a ranked list optimized purely for predicted relevance tends to
cluster near-duplicates at the top (e.g. eight similar beach cities in a
row). Reranking is the last pass before showing results: it trades a little
relevance for diversity, and can also apply business rules (deprioritize
closed-for-season destinations) and exploration (occasionally surface a
lower-scored but novel item to gather feedback).

**Data:** a ranked candidate list from Project 5/6/7, plus the city
description embeddings already cached by `embed_retrieve` (used as the
similarity signal between candidates).

**Algorithms to compare:**
- Maximal Marginal Relevance (MMR) — greedily pick the next item maximizing
  `λ * relevance - (1-λ) * max_similarity_to_selected`
- Determinantal point processes (DPP) — sample a diverse subset directly from
  a kernel encoding relevance and pairwise similarity, rather than MMR's
  greedy one-at-a-time approximation
- Simple rule-based tag/region bucketing (cap items per country/category), as
  the cheap deterministic baseline

**Knobs to tune:** MMR's relevance/diversity tradeoff weight (λ), DPP kernel
choice.

## Experiment

**Setup:** rerank top-50 → top-10 from Projects 5/6/7.

**Conditions:** MMR (sweep λ ∈ [0,1]) · DPP (sweep kernel bandwidth) · tag/
region bucketing (round-robin over buckets).

**Metrics:** a *pair* per configuration — relevance (NDCG@10) and diversity
(mean pairwise distance between selected items' tag vectors, plus count of
distinct regions).

**Interpretation:** the deliverable is the **Pareto frontier**, not a winning
algorithm. Plot relevance against diversity across the sweep; the questions
are which method dominates, and how much NDCG the knee costs. A single
configuration's numbers are not a result here.

**Results:** *not yet run.*

**Status:** not yet implemented (README/design only). Goal is a plotted
diversity-vs-relevance tradeoff curve over Project 5/6/7's output, not just
one algorithm working.
