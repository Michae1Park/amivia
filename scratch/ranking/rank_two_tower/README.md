# Project 6: Two-tower ranking

Layer: precise ranking (DNN / two-tower) — see `docs/architecture.md` §5b.

**Problem:** `rank_destinations` (Project 5) ranks with hand-built structured
features. A two-tower model instead *learns* a user embedding and an item
embedding jointly from interaction data, scoring by dot product — the
architecture also used for query-less retrieval (a user's tower output
doubles as the "query" vector, see the earlier discussion on how YouTube/
Netflix-style feeds recommend without a typed query).

**Data:** `scratch/synthetic_interactions` — **built and ready**.

**Algorithms to compare:** two-tower DNN vs. Project 5's GBDT baseline, on
the same held-out interactions.

**Knobs to tune:** embedding dimension, negative sampling strategy (random /
in-batch / hard negatives), tower depth.

## Experiment

**Setup:** two-tower DNN on `synthetic_interactions`, same `eval/` split.
Item tower over tag/budget features, user tower over interaction history.

**Conditions:** negative sampling is the main variable — random · in-batch ·
hard negatives (high-popularity non-clicks). Secondary: embedding dim
{32, 64, 128}, tower depth {1, 2, 3}. Baseline: Project 5's GBDT.

**Metrics:** recall@k / NDCG@k against GBDT; plus a **persona-recovery
diagnostic** — k-means (k=6) over learned user embeddings scored against the
held-out `primary_persona` labels (cluster purity / adjusted Rand index).

**Interpretation:** the two measure different things and must be reported
separately. Persona recovery validates the *implementation* — it shows the
model can recover structure that was deliberately planted. It says nothing
about whether real traveler behaviour contains structure of that shape. State
that explicitly; the honest claim is "recovers planted personas at ARI x",
never "learns traveller taste".

**Results:** *not yet run.*

**Status:** unblocked, not started — `synthetic_interactions` is built, so this
can begin whenever. Score it with `scratch/eval`.

Note: synthetic data validates the *implementation* (does the model recover the
persona structure that was planted?), not the *modeling decision* (does real
traveler behavior have structure of this shape?). See
`docs/roadmap-to-service.md` → Synthetic data can't serve real users.
