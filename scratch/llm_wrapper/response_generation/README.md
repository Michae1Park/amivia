# Project 11: Response generation (optional, LLM)

Layer: response generation — see `docs/architecture.md` §7. Combined with
candidate retrieval (Project 1), this is a RAG pipeline — there's no
separate "RAG technology" beyond these two layers already existing.

**Problem:** turn the backbone's final ranked list into a conversational,
grounded answer (an itinerary suggestion) instead of a bare results list —
without letting the model invent destinations that aren't in the candidate
list.

**Data:** a fixed candidate list (from Project 1/5/8's output) plus the
original parsed intent (Project 10, or a hand-written stand-in).

**Algorithm/tech:** not an algorithm-comparison layer — LLM prompted with
the ranked candidates + intent, generating grounded natural-language output.

**Knobs to tune:** prompt design, temperature.

## Experiment

**Setup:** fixed ranked candidate lists (10 cities) plus the parsed intent;
generate a conversational answer over ~30 scenarios.

**Conditions:** prompt variants (bare list · list + intent restatement · list
+ explicit grounding instruction) × model tier.

**Metrics:** **grounding rate** (primary) — fraction of place names in the
output that appear in the supplied candidate list, by string match; plus
answer length, latency, and an LLM-as-judge rubric score for helpfulness.

**Interpretation:** grounding is a gate, not a tradeoff — a fluent answer that
invents a city is a failure regardless of its rubric score, so report grounding
first and treat anything below 100% as the headline. This is the automatable
half of the hallucination check described in `docs/roadmap-to-service.md`.

**Results:** *not yet run.*

**Status:** not yet implemented (README/design only).
