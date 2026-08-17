# Project 9: Feedback loop / taste profile

Layer: feedback loop, feeds the persistent taste profile in user
understanding — see `docs/architecture.md` §2 and §9.

**Problem:** a user's persistent taste profile shouldn't be static — it
should update from what they actually click/save over time, and recent
signal should generally matter more than old signal. This project builds
that update step and compares ways of aggregating interaction history into
one profile vector.

**Data:** `scratch/synthetic_interactions` — **built and ready**. Its per-user click/save history over time is exactly
what a taste profile is built from, and its 2-year synthetic timeline exists
specifically so decay experiments have something real to decay over.

**Algorithms to compare:**
- Simple average of liked-item embeddings
- Recency-decay-weighted average (recent clicks/saves weighted higher)
- A learned user-tower (reuses Project 6's two-tower training)

**Knobs to tune:** decay rate, aggregation window.

## Experiment

**Setup:** build a profile vector per user from train-set clicks/saves, use it
as a retrieval query, score against held-out interactions with `eval/`.

**Conditions:** unweighted mean of liked-item vectors · recency-decay-weighted
mean (sweep half-life {7, 30, 90, 365} days) · learned user tower (Project 6).

**Metrics:** recall@k / NDCG@k by condition and half-life.

**Interpretation:** **expect recency decay to show no gain, and check why
before tuning.** `generate.py` draws each user's persona once and never drifts
it, so preferences are stationary by construction and down-weighting old
signal can only discard usable data. A flat or declining curve across
half-lives confirms the generator behaved as designed — it is a null result
about the data, not about decay. Making it a real question means adding drift
to the generator; note that as the follow-up rather than tuning around it.

**Results:** *not yet run.*

**Status:** unblocked, not started — `synthetic_interactions` is built, so this
can begin whenever. Score it with `scratch/eval`.
