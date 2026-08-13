# Project 9: Feedback loop / taste profile

Layer: feedback loop, feeds the persistent taste profile in user
understanding — see `docs/architecture.md` §2 and §9.

**Problem:** a user's persistent taste profile shouldn't be static — it
should update from what they actually click/save over time, and recent
signal should generally matter more than old signal. This project builds
that update step and compares ways of aggregating interaction history into
one profile vector.

**Data:** `scratch/synthetic_interactions` — this project is blocked on that
one being built first. Its per-user click/save history over time is exactly
what a taste profile is built from, and its 2-year synthetic timeline exists
specifically so decay experiments have something real to decay over.

**Algorithms to compare:**
- Simple average of liked-item embeddings
- Recency-decay-weighted average (recent clicks/saves weighted higher)
- A learned user-tower (reuses Project 6's two-tower training)

**Knobs to tune:** decay rate, aggregation window.

**Status:** blocked on `synthetic_interactions` — not yet implemented.
