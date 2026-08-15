# Project 7: Cross-encoder ranking

Layer: precise ranking (transformer) — see `docs/architecture.md` §5b.

**Problem:** two-tower models (Project 6) score a user and item independently
and combine them cheaply (dot product) — fast, but the two sides never
actually attend to each other. A cross-encoder instead scores a
(user-context, item) pair jointly through one transformer pass — higher
quality, much higher cost per pair. This project is about feeling that
quality/latency tradeoff directly, not about needing this much power at this
project's data scale.

**Data:** same candidate/interaction data as Projects 5 and 6, re-scored
pairwise.

**Algorithms to compare:** cross-encoder re-scoring vs. two-tower (Project 6)
vs. GBDT (Project 5).

**Knobs to tune:** model size, pair batch size.

**Status:** not yet implemented (README/design only).
