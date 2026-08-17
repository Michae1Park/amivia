# Project 10: Intent parsing (optional, LLM)

Layer: conversation layer / user understanding — see `docs/architecture.md`
§1 and §2. First of the optional LLM-wrapper projects; the backbone
(Projects 1-9) works without this.

**Problem:** turn a free-text query ("cheap beach trip in March, no crowds")
into the structured session intent the backbone actually consumes (budget,
dates, tag constraints) — the same slot-filling job a classic search UI's
form/filters do, done by an LLM instead.

**Data:** hand-written example queries covering the constraint types
`filter_constraints` (Project 3) needs to enforce.

**Algorithm/tech:** not an algorithm-comparison layer — LLM structured-output
extraction (JSON schema / tool-calling) of intent fields from free text.

**Knobs to tune:** prompt/schema design, temperature.

## Experiment

**Setup:** ~50 hand-written free-text travel queries, each hand-labelled with
its target intent JSON (destination hints, tags, budget, dates, party size,
negations).

**Conditions:** prompt variants (bare schema · schema + field descriptions ·
schema + few-shot) × structured outputs on/off × model tier.

**Metrics:** per-field exact match; F1 for set-valued fields (tags,
negations); schema-validity rate; latency and cost per parse.

**Interpretation:** report per-field, not as one average — the failures will
concentrate in dates and budget, and a single accuracy number hides that.
Negation extraction is the field that matters most downstream, since it is
what `filter_constraints` consumes and what embeddings provably cannot do.

**Results:** *not yet run.*

**Status:** not yet implemented (README/design only).
