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

**Status:** not yet implemented (README/design only).
