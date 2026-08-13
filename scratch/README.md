# Scratch: recsys toy projects

Small, self-contained projects for learning the algorithms behind production
recommendation/search pipelines, before wiring anything to an LLM. Each
subfolder is independent: its own README (problem it solves, algorithms
compared, knobs to tune), its own script(s), its own small dataset.

This is the working index — for the full pipeline diagram, per-layer
technology notes, and the reasoning behind the ordering, see
[`docs/architecture.md`](../docs/architecture.md). That document is the
source of truth for the roadmap; this file just tracks status and links in.

| # | Project | Layer | Status |
|---|---------|-------|--------|
| 1 | [embed_retrieve](embed_retrieve/README.md) | Candidate retrieval (embeddings) | In progress — brute-force baseline done, ANN variants not yet built |
| 2 | [collab_filter](collab_filter/README.md) | Candidate retrieval (collaborative filtering) | Blocked on `synthetic_interactions` |
| 3 | [filter_constraints](filter_constraints/README.md) | Filtering | Not started |
| 4 | [rank_coarse](rank_coarse/README.md) | Coarse ranking | Not started |
| 5 | [rank_destinations](rank_destinations/README.md) | Precise ranking (GBDT / LTR) | Scaffolded — dataset chosen, no code yet |
| 6 | [rank_two_tower](rank_two_tower/README.md) | Precise ranking (DNN / two-tower) | Blocked on `synthetic_interactions` |
| 7 | [rank_cross_encoder](rank_cross_encoder/README.md) | Precise ranking (transformer) | Not started |
| 8 | [rerank_diversity](rerank_diversity/README.md) | Reranking | Not started |
| 9 | [feedback_taste_profile](feedback_taste_profile/README.md) | Feedback loop / user understanding | Blocked on `synthetic_interactions` |
| 10 | [intent_parsing](intent_parsing/README.md) *(optional, LLM)* | Conversation layer | Not started |
| 11 | [response_generation](response_generation/README.md) *(optional, LLM)* | Response generation | Not started |
| 12 | [tool_calling_agent](tool_calling_agent/README.md) *(optional, LLM)* | Agent tooling | Not started |

**Shared prerequisite:** [synthetic_interactions](synthetic_interactions/README.md)
generates synthetic users, personas, and an impression/click/save
interaction log over the `embed_retrieve` city catalog. Projects 2, 6, and 9
all need real user-item interaction history that the catalog-only datasets
don't have, and consume this one shared log instead of each inventing their
own. Build it before starting any of those three.

Projects 1-9 are the classic recsys backbone and work with no LLM at all —
that's the learning priority. Projects 10-12 are an optional conversational
wrapper added around the backbone afterward.
