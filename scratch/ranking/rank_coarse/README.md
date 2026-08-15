# Project 4: Coarse ranking

Layer: coarse ranking — see `docs/architecture.md` §5a.

**Problem:** running an expensive ranker over every filtered candidate is too
costly at scale. Production systems first use a cheap model to narrow
hundreds of candidates down to tens, and only then run the expensive ranker
(Project 5) on that much smaller set.

**Data:** Project 3's (`filter_constraints`) filtered candidate set.

**Algorithms to compare:**
- Raw retrieval similarity score, used as-is (baseline)
- Logistic regression over a handful of cheap features
- A shallow gradient-boosted tree

**Knobs to tune:** feature set size, regularization/tree depth.

**Status:** not yet implemented (README/design only). The interesting
measurement here is how much of Project 5's eventual top results survive
coarse pruning, and at what latency savings.
