# Project 3: Hard-constraint filtering

Layer: filtering — see `docs/architecture.md` §4. `content_filter`'s
similarity search is bad at negation and numeric thresholds — it conflates
"no nightlife" with "vibrant nightlife," and has no notion of "under
$50/day" at all (`content_filter/batch_test.py` shows this directly). This
layer enforces those hard constraints deterministically, downstream of
retrieval, instead of hoping the embedding model learns them.

Not an algorithm-comparison layer like Projects 1 and 2 — plain structured
predicate logic over item metadata, no ML. The thing being measured here is
a design tradeoff (how strictly to enforce a constraint), not a model choice.

## Preparation

Check yourself against [`PREPARATION_NOTES.md`](PREPARATION_NOTES.md) — background
concepts, not the lab's findings, so there's nothing to spoil by reading it first.

- [ ] Know what "violating a predicate" means here, and that it's boolean per predicate, summed into a count
- [ ] Know why hard-fail (dropping) and soft-penalty (demoting) are fundamentally different operations — one can shrink a result list, the other can't
- [ ] Know why a filter applied after retrieval can only work with the candidates retrieval already handed it
- [ ] Know what "retention" measures here, and why a low number isn't automatically a bad sign
- [ ] Skim `content_filter/batch_test.py`'s negation and numeric probe queries — this project's motivating failure mode

## Objectives

- Feel that filtering and retrieval are different *kinds* of layers — one
  probabilistic and tunable, one deterministic and boolean
- Feel the actual cost of enforcing correctness: how much topical relevance
  gets traded away to guarantee it
- Feel why soft enforcement structurally can't do what hard enforcement can,
  no matter how it's tuned
- Practice catching a metric that measures something other than what it
  looks like it measures

## Questions to answer

1. How often does the unfiltered retriever actually violate the constraint its own query stated?
2. Does hard-fail reach zero violations? Does soft-penalty, at any strength?
3. If soft-penalty can't reach zero, is that a tuning failure or a structural limit — what's the argument either way?
4. What does "retention" actually measure — lost relevance, or something else? What would you wrongly conclude if you read it the other way?
5. For which queries does the pool of constraint-satisfying candidates inside the retrieved top-N run out? What does that imply about filtering after retrieval vs. during it?
6. Which queries have no honest predicate in this schema at all, and what should happen to them instead of forcing a fit?

**Before you touch the code, write down a guess** — what fraction of the
unfiltered top-10 you expect to violate its own query's constraint, and
whether you think soft-penalty can reach zero violations if you push the
penalty high enough. Nothing here checks that guess for you.

## Background

*Definitions and equations live in [`PREPARATION_NOTES.md`](PREPARATION_NOTES.md) — this is just orientation.*

- **Input:** `content_filter`'s top-100 candidates for each labelled query,
  plus the catalog's structured columns (`budget_level`, tags) to filter on.
- **Predicates** (`constraints.py`): `TagAtLeast` / `TagAtMost` (negation),
  `BudgetAtMost`, `MonthTempAtLeast` / `MonthTempAtMost` — same schema
  `content_filter`'s prep notes document, defined here.
- **A stated assumption:** the catalog has no price column, only a
  three-level `budget_level` enum, so "under $50 a day" maps onto
  `BudgetAtMost("Budget")` under an assumed scale (documented at the top of
  `constraints.py`, not left implicit).
- **The knob:** predicate strictness — hard-fail a violator vs. soft-penalize
  it (demote by a fixed margin instead of dropping it).
- **Vocabulary:** violation rate = fraction of *returned* results that break
  at least one constraint; retention = how much of the unfiltered top-$k$
  survived filtering; mean_score = mean of the *original* (unpenalized)
  cosine score of what's returned, i.e. how far down the similarity ranking
  the filter reached.

## Procedure

**Setup** (skip the venv creation if you already made one for `content_filter`
or `collab_filter` — it's shared across all three):
```
python3 -m venv ../.venv               # once, from anywhere in candidate_generation/
source ../.venv/bin/activate           # re-run this in every new shell
pip install -r ../../requirements.txt
```

**Part a — sanity check.** `python3 filter_experiment.py --detail` and read
one query's `no_filter` vs. `hard_fail` output. Does hard-fail visibly drop
the constraint-violating cities?

**Part b — the full run.** `python3 filter_experiment.py` (no flags). Read
the condition table across all 15 labelled queries and the soft-penalty
sweep. What's the unfiltered violation rate — does it match your
pre-registered guess?

**Part c — hard-fail's actual cost.** Read `retention` and `mean_score`
together for `hard_fail`. Retention looks low — is that alarming, or exactly
what you'd expect given Part b's violation rate? What does `mean_score`'s
drop from the unfiltered baseline tell you that `retention` alone doesn't?

**Part d — soft-penalty's floor.** Across the penalty sweep, does violation
rate keep falling as the penalty rises, or does it plateau? At what value,
and does it ever reach the same zero hard-fail reaches?

**Part e — find the capacity limit.** The plain `python3 filter_experiment.py`
run from Part b already prints a "Candidate feasibility" table — the count
of constraint-satisfying candidates inside each query's top-100 (`--detail`
isn't needed for this table specifically; it only adds extra per-query top-10
dumps on top of it). For the queries with the smallest counts, what happens
to `hard_fail`'s `n_returned`? Connect this back to Part d: is a low
feasibility count and a stuck violation-rate floor the same underlying
problem?

**Part f — what doesn't map.** Check `UNMAPPABLE_QUERIES` in
`constraints.py`. Why can't "avoid touristy spots" become a predicate in this
schema, and what would it take to change that (a new catalog column? a
different kind of signal entirely)?

## Deliverables

Fill these in as you go in [`DELIVERABLES.md`](DELIVERABLES.md) — a template with every question and table already laid out, so you're writing answers, not reformatting.

1. Your own answer to each of the six questions above, in your own numbers
2. The full condition table from Part b (no-filter, hard-fail, and the
   soft-penalty sweep), with violation rate, retention, mean score, n returned
3. One paragraph on Part c: why a low retention number here doesn't mean
   what it would mean in a different context
4. Your Part d/e finding stated as a general claim: under what condition
   (in terms of feasibility count vs. $k$) can soft-penalty *never* reach
   zero violations, regardless of tuning?
5. A one-line recommendation: where should hard constraints actually be
   enforced in a production version of this funnel, and why — pushed into
   retrieval, or kept as this kind of post-pass?

## Notes

- `constraints.py` and `filter_experiment.py` are both implemented and
  runnable — this lab is about running them and interpreting the output,
  not writing new code.
- Not covered: constraints the catalog's schema can't express at all (e.g.
  "touristy") — see `UNMAPPABLE_QUERIES` and Part f. A deterministic filter
  can only enforce what the catalog actually represents.
- A prior write-up with actual measured numbers exists in this file's git
  history, if you want to check your Deliverables against it after — not
  before.
