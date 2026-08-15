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

**Status:** not yet implemented (README/design only).
