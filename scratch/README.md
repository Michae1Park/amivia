# Scratch: recsys toy projects

Small, self-contained projects for learning the algorithms behind production
recommendation/search pipelines, before wiring anything to an LLM.
- Each subfolder is independent: its own README (problem it solves,
  algorithms compared, knobs to tune), its own script(s), its own small
  dataset
- This is the working index — for the full pipeline diagram, per-layer
  technology notes, and the reasoning behind the ordering, see
  [`docs/architecture.md`](../docs/architecture.md). That document is the
  source of truth for the roadmap; this file just tracks status and links in.

Grouping:
- `candidate_generation/`, `ranking/` — the two classic-backbone funnel
  stages
- `llm_wrapper/` — the optional LLM wrapper layers
- `scratch/` root — `feedback_taste_profile` (feedback/user understanding)
  and the shared synthetic-data prerequisite, which don't fit any of the
  three

## Experiments and results

**Every project finishes with a written result, including a negative one.**
"Compared X and Y, Y was no better, here's why" is a finished project; a tuned
model with no writeup is not. The interpretation is the deliverable.

Each project README carries a pre-specified `## Experiment` section — written
*before* running anything, so the result can't be reverse-engineered from
whatever came out:

- **Setup** — data, split, and fixed parameters.
- **Conditions** — the variants compared and the knobs swept.
- **Metrics** — what is measured, primary first, and what it is measured
  against.
- **Interpretation** — what a win, a null, and a failure each look like, plus
  the diagnostic that distinguishes them.
- **Results** — filled in after the run.

Two standing caveats to state whenever they apply: a result from synthetic data
validates an implementation, not a hypothesis about real travellers; and on
this harness `popularity`, not `random`, is the bar to beat (see
[`eval/`](eval/README.md) for why).

The `candidate_generation/` trio are the worked examples. Each produced at least
one result that contradicted the expectation written down before the run:
`content_filter`'s predicted crossover point didn't exist, `collab_filter`'s
simplest model beat its three learned ones, and `filter_constraints` found that
soft-penalising cannot reach zero violations at any setting.

## `candidate_generation/` — retrieval + filtering

| # | Project | Layer | Status |
|---|---------|-------|--------|
| 1 | [content_filter](candidate_generation/content_filter/README.md) | Candidate retrieval (embeddings) | Done — brute force + IVF/HNSW/LSH swept to 1M; the index only earns its keep past ~100k |
| 2 | [collab_filter](candidate_generation/collab_filter/README.md) | Candidate retrieval (collaborative filtering) | Done — all four models beat `popularity`; item-kNN wins, and the popularity-correlation diagnostic catches ALS collapsing into `popularity` at high reg |
| 3 | [filter_constraints](candidate_generation/filter_constraints/README.md) | Filtering | Done — 79% of the unfiltered top-10 violates its query's constraint; hard-fail fixes it, soft-penalise provably cannot |

## `ranking/` — coarse, precise, and reranking

| # | Project | Layer | Status |
|---|---------|-------|--------|
| 4 | [rank_coarse](ranking/rank_coarse/README.md) | Coarse ranking | Not started |
| 5 | [rank_destinations](ranking/rank_destinations/README.md) | Precise ranking (GBDT / LTR) | Scaffolded — dataset chosen, no code yet |
| 6 | [rank_two_tower](ranking/rank_two_tower/README.md) | Precise ranking (DNN / two-tower) | Unblocked, not started |
| 7 | [rank_cross_encoder](ranking/rank_cross_encoder/README.md) | Precise ranking (transformer) | Not started |
| 8 | [rerank_diversity](ranking/rerank_diversity/README.md) | Reranking | Not started |

## `scratch/` root — feedback, shared prerequisite

| # | Project | Layer | Status |
|---|---------|-------|--------|
| 9 | [feedback_taste_profile](feedback_taste_profile/README.md) | Feedback loop / user understanding | Unblocked, not started |

## `llm_wrapper/` — optional conversational wrapper around the backbone

| # | Project | Layer | Status |
|---|---------|-------|--------|
| 10 | [intent_parsing](llm_wrapper/intent_parsing/README.md) *(optional, LLM)* | Conversation layer | In progress — schema + weak/OOD datasets built; prompting arm and a LoRA fine-tune arm scaffolded, neither run yet |
| 11 | [response_generation](llm_wrapper/response_generation/README.md) *(optional, LLM)* | Response generation | Not started |
| 12 | [tool_calling_agent](llm_wrapper/tool_calling_agent/README.md) *(optional, LLM)* | Agent tooling | Not started |

## `eval/` — offline metrics harness (cross-cutting)

| # | Project | Layer | Status |
|---|---------|-------|--------|
| — | [eval](eval/README.md) | Cross-cutting (scores `collab_filter`, `rank_two_tower`, `feedback_taste_profile`) | Done — harness + random/popularity/oracle_persona baselines validated; opt-in `llm` baseline added, not yet run against the API |

**Shared prerequisite:** [synthetic_interactions](synthetic_interactions/README.md)
— **built**, so projects 2, 6, and 9 are unblocked.
- Generates synthetic users, personas, and an impression/click/save
  interaction log over the `content_filter` city catalog
- Projects 2, 6, and 9 all need user-item interaction history that the
  catalog-only datasets don't have, and consume this one shared log instead
  of each inventing their own


Projects 1-9 are the classic recsys backbone and work with no LLM at all —
that's the learning priority. Projects 10-12 are an optional conversational
wrapper added around the backbone afterward.
