# Project 3: Hard-constraint filtering

Layer: filtering — see `docs/architecture.md` §4.

**Problem:** similarity search is bad at negation and numeric thresholds —
`embed_retrieve/batch_test.py` already shows embeddings conflating "no
nightlife" with "vibrant nightlife," and having no notion of "under
$50/day" at all. This layer enforces those hard constraints deterministically
after retrieval, instead of hoping the embedding model learns them.

**Data:** Project 1's (`embed_retrieve`) candidate output, plus its
structured columns (budget_level, tags) to filter on.

**Algorithm/tech:** not an algorithm-comparison layer — plain structured
predicate logic over item metadata (budget range, dates, required/excluded
tags) applied to the retrieved candidate set. No ML.

**Knob to tune:** predicate strictness — hard-fail a near-miss vs.
soft-penalize it (push it down instead of dropping it), and how that
tradeoff affects recall of the final list.

**Status:** not yet implemented (README/design only). Parse a query's hard
constraints and apply them as predicates over Project 1's candidates.
