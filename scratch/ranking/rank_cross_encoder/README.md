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

## Experiment

**Setup:** re-score the top-50 from coarse ranking; same `eval/` split.

**Conditions:** cross-encoder (query+item pair, small and base checkpoints) ·
two-tower (Project 6) · GBDT (Project 5).

**Metrics:** NDCG@10 gain over the two-tower baseline, against p50/p95 latency
per query and per-query cost.

**Interpretation:** report the exchange rate — **milliseconds per NDCG point**
— not the raw scores. At 50 candidates over a 560-item catalog the expected
outcome is a small quality gain for a large latency cost; quantifying that
exchange rate is the result, and "not worth it at this scale" is a valid
conclusion if the number says so.

**Results:** *not yet run.*

**Status:** not yet implemented (README/design only).
