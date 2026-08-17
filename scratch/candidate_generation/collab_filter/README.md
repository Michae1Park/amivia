# Project 2: Collaborative-filtering candidate retrieval

Layer: candidate retrieval (collaborative filtering) — see `docs/architecture.md` §3.

**Problem:** `embed_retrieve` retrieves cities by matching content (descriptions)
to a query. It has no notion of "travelers with taste similar to yours liked
these other cities" — the other major candidate-generation signal production
systems run in parallel with content-based retrieval.

**Data:** `scratch/synthetic_interactions` — synthetic users, personas, and an
impression/click/save interaction log built on top of `embed_retrieve`'s city
catalog. **Built and ready** — 5,000 users, 450,852 impressions.

**Algorithms to compare:**
- Matrix factorization (ALS or SVD) over the user-item interaction matrix
- Item-based neighborhood collaborative filtering (cosine similarity over
  co-interaction patterns)
- Implicit-feedback Bayesian Personalized Ranking (BPR)

**Knobs to tune:** latent dimension, regularization strength, how implicit
feedback (impression/click/save) is weighted vs. treated as explicit rating.

## Experiment

**Setup:** `synthetic_interactions` (5,000 users, 450,852 impressions), per-user
temporal holdout via `eval/`. Graded relevance: save=2, click=1.

**Conditions:** ALS · SVD · item-kNN (cosine over co-interaction) · BPR.
Sweep latent dim {16, 32, 64, 128} and regularisation; for implicit feedback
compare binary (click∪save) against confidence-weighted (save=2, click=1).

**Metrics:** recall@k, NDCG@k, hit-rate@k for k ∈ {5, 10, 20}, against
`random` / `popularity` / `oracle_persona`.

**Interpretation:** the bar is `popularity`, not `random`. Because impressions
are popularity-sampled, CF trained on this log may simply *relearn popularity*
— so also report Spearman correlation between each model's per-user ranking
and the global popularity ranking. High correlation plus a small NDCG gain
means it learned exposure, not taste, and should be reported as such.

**Results:** *not yet run.*

**Status:** unblocked, not started — `synthetic_interactions` is built, so this
can begin whenever. Score it with `scratch/eval`.
