# Project 2: Collaborative-filtering candidate retrieval

Layer: candidate retrieval (collaborative filtering) — see `docs/architecture.md` §3.

**Problem:** `embed_retrieve` retrieves cities by matching content (descriptions)
to a query. It has no notion of "travelers with taste similar to yours liked
these other cities" — the other major candidate-generation signal production
systems run in parallel with content-based retrieval.

**Data:** `scratch/synthetic_interactions` — synthetic users, personas, and an
impression/click/save interaction log built on top of `embed_retrieve`'s city
catalog. This project is blocked on that one being built first.

**Algorithms to compare:**
- Matrix factorization (ALS or SVD) over the user-item interaction matrix
- Item-based neighborhood collaborative filtering (cosine similarity over
  co-interaction patterns)
- Implicit-feedback Bayesian Personalized Ranking (BPR)

**Knobs to tune:** latent dimension, regularization strength, how implicit
feedback (impression/click/save) is weighted vs. treated as explicit rating.

**Status:** blocked on `synthetic_interactions` — not yet implemented.
