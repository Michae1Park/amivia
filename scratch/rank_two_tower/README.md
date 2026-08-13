# Project 6: Two-tower ranking

Layer: precise ranking (DNN / two-tower) — see `docs/architecture.md` §5b.

**Problem:** `rank_destinations` (Project 5) ranks with hand-built structured
features. A two-tower model instead *learns* a user embedding and an item
embedding jointly from interaction data, scoring by dot product — the
architecture also used for query-less retrieval (a user's tower output
doubles as the "query" vector, see the earlier discussion on how YouTube/
Netflix-style feeds recommend without a typed query).

**Data:** `scratch/synthetic_interactions` — this project is blocked on that
one being built first.

**Algorithms to compare:** two-tower DNN vs. Project 5's GBDT baseline, on
the same held-out interactions.

**Knobs to tune:** embedding dimension, negative sampling strategy (random /
in-batch / hard negatives), tower depth.

**Status:** blocked on `synthetic_interactions` — not yet implemented.
