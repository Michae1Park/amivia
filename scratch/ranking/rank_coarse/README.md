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

## Experiment

**Setup:** filtered candidate sets (~hundreds/user) from Projects 1 + 3;
Project 5's precise ranker as the reference.

**Conditions:** raw retrieval score · logistic regression · shallow GBDT
(depth ≤ 4). Sweep the cut N ∈ {20, 50, 100}.

**Metrics:** **survival rate** — the fraction of the precise ranker's top-10
still present after coarse pruning to N — plus scoring latency per candidate.

**Interpretation:** the target here is agreement with the expensive ranker, not
ground-truth relevance; a coarse stage that drops the eventual winners is a
loss no matter how fast. Plot survival vs N and read off the smallest N holding
survival ≥ 0.95, then report the latency that buys.

**Results:** *not yet run.*

**Status:** not yet implemented (README/design only). The interesting
measurement here is how much of Project 5's eventual top results survive
coarse pruning, and at what latency savings.
