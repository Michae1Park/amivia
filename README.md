# amivia

Friend on the road. Your companion from intent to itinerary.

A travel recommender system, built as a portfolio demo: implement each layer of
a production recsys funnel as a small standalone project first, then assemble
into something functioning end to end.

## Start here

| Document | What it answers |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | **The plan being followed.** What each funnel layer does, the technology behind it, and the 12-project roadmap for learning them one at a time. |
| [`scratch/README.md`](scratch/README.md) | Status index — which projects are done, unblocked, or not started, with links into each. |
| [`docs/roadmap-to-service.md`](docs/roadmap-to-service.md) | What a complete, assembled version needs — entity types, data gaps, build phases. A destination map, not the current work queue. |

Reading order for a full picture: `architecture.md` → `scratch/README.md` →
individual project READMEs, with `roadmap-to-service.md` for where it all
lands.

## Layout

```
docs/       architecture + service roadmap
scratch/    one directory per toy project, each with its own README
  candidate_generation/   retrieval + filtering  (projects 1-3)
  ranking/                coarse, precise, rerank (projects 4-8)
  llm_wrapper/            optional conversational layer (projects 10-12)
  feedback_taste_profile/ feedback loop (project 9)
  synthetic_interactions/ shared user-interaction data (built)
  eval/                   shared offline metrics harness
```

## Running anything

Each project README lists its own setup. Most need only the venv plus
`sentence-transformers` / `numpy`; the opt-in LLM eval baseline additionally
needs `anthropic` and API credentials.
