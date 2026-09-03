# Project 3: Hard-constraint filtering

Layer: filtering — see `docs/architecture.md` §4.

**Problem:** similarity search is bad at negation and numeric thresholds —
`content_filter/batch_test.py` already shows embeddings conflating "no
nightlife" with "vibrant nightlife," and having no notion of "under
$50/day" at all. This layer enforces those hard constraints deterministically
after retrieval, instead of hoping the embedding model learns them.

**Data:** Project 1's (`content_filter`) candidate output, plus its
structured columns (budget_level, tags) to filter on.

**Algorithm/tech:** not an algorithm-comparison layer — plain structured
predicate logic over item metadata (budget range, dates, required/excluded
tags) applied to the retrieved candidate set. No ML.

**Knob to tune:** predicate strictness — hard-fail a near-miss vs.
soft-penalize it (push it down instead of dropping it), and how that
tradeoff affects recall of the final list.

**Usage:**
```
python3 filter_experiment.py            # the three conditions + the tradeoff table
python3 filter_experiment.py --detail   # per-query top-10, filtered and unfiltered
```

**Predicates** (`constraints.py`): `TagAtMost` (negation), `TagAtLeast`,
`BudgetAtMost`, `MonthTempAtLeast` / `MonthTempAtMost` (parsed out of the
catalog's `avg_temp_monthly` JSON).

**A stated assumption:** the catalog has no price column — only a three-level
`budget_level` enum — so "under $50 a day" is mapped onto
`BudgetAtMost("Budget")` under an assumed scale (Budget ≈ <$50/day,
Mid-range ≈ $50–150, Luxury ≈ >$150). That mapping is a modelling choice the
data does not itself support, so it lives at the top of `constraints.py`
rather than being implied by the code.

## Experiment

Not an algorithm comparison — but the strictness knob is measurable.

**Setup:** `content_filter` top-100 candidates for `batch_test.py`'s negation
and numeric probe queries, each with a hand-labelled constraint.

**Conditions:** no filter (baseline) · hard-fail · soft-penalise (demote by a
fixed margin rather than drop).

**Metrics:** constraint-violation rate in the returned top-10 (primary), and
recall loss against the unfiltered candidate set.

**Interpretation:** filtering trades recall for correctness — plot the two
against penalty strength. The useful number is how much recall a violation
rate near zero costs, and whether soft-penalising gets there more cheaply than
hard-failing.

**Deviation from the pre-specified setup:** `batch_test.py` supplies only four
negation/numeric probes, which makes a violation rate move in steps of 2.5
points — too coarse to read. The labelled set was widened to 15 queries across
the same three failure modes (negation, budget threshold, temperature
threshold) plus combinations. Labels are in `constraints.py`.

## Results

15 labelled queries, top-100 candidates, top-10 returned. `retention` is the
fraction of the *unfiltered* top-10 still present; `mean_score` is the mean
cosine of what was returned (filtering pushes you down the similarity ranking).

| condition | violation rate | retention | mean score | n returned |
|---|---|---|---|---|
| no filter | **0.793** | 1.000 | 0.403 | 10.0 |
| hard-fail | **0.000** | 0.207 | 0.340 | 8.9 |
| soft, p=0.01 | 0.713 | 0.907 | 0.393 | 10.0 |
| soft, p=0.02 | 0.647 | 0.827 | 0.385 | 10.0 |
| soft, p=0.05 | 0.453 | 0.633 | 0.366 | 10.0 |
| soft, p=0.1 | 0.267 | 0.453 | 0.346 | 10.0 |
| soft, p=0.2 | 0.113 | 0.273 | 0.327 | 10.0 |
| soft, p=0.5 | 0.107 | 0.267 | 0.295 | 10.0 |
| soft, p=1.0 | 0.107 | 0.267 | 0.242 | 10.0 |

**The layer is justified: 79% of the unfiltered top-10 violates the query's own
stated constraint.** For *"quiet town, definitely no nightlife"* it is 10 out of
10 — Mumbai, São Paulo, Belgrade, Tampa, Reykjavík. Hard-failing returns
Naypyidaw, Keswick, Chefchaouen, Anchorage instead: actually quiet places, at a
cosine of 0.40 instead of 0.51.

**Soft-penalising cannot reach zero violations, at any penalty strength.** It
plateaus at 0.107 from p=0.2 onward, and raising the penalty to 1.0 — larger
than the entire cosine range in play — does not move it. This is structural,
not a tuning failure: demotion reorders a list, it cannot shorten one. When
fewer than 10 candidates satisfy the constraints, a soft-penalised top-10 has
to fill the remaining slots with violators no matter how far down they are
pushed. Only dropping items can guarantee correctness.

So the README's question — does soft-penalising reach a near-zero violation
rate more cheaply? — has a clean answer: **it never reaches it at all.** At its
floor it retains 0.267 of the unfiltered top-10 against hard-fail's 0.207,
which buys about half an extra original item while still returning roughly one
violator per ten results.

**Reading the retention number correctly.** Hard-fail retaining only 0.207 of
the unfiltered top-10 looks alarming and is not. With a baseline violation rate
of 0.793, the displaced items are overwhelmingly the *violating* ones —
discarding them is the entire job. Retention here measures how much the filter
perturbs the similarity ranking, not how much relevance it destroys. The real
cost shows up in `mean_score`: 0.403 → 0.340, i.e. correctness is bought by
accepting topically weaker matches.

**The finding that changes the design: retrieve-then-filter starves the list.**
Hard-fail returns 8.9 items on average, not 10, because the constraint-
satisfying set inside the top-100 is often tiny — 9/100 for "no nightlife",
7/100 for "budget beach trip with warm July weather", and 2/100 for "adventure
travel on a budget, no beaches". Filtering *after* retrieval can only remove;
it cannot go find the qualifying cities that ranked 300th. A production system
pushes hard predicates down into retrieval (a pre-filtered ANN search, or a
metadata index queried first) rather than applying them as a post-pass. That is
a concrete argument for wiring this layer into `content_filter` rather than
bolting it on afterwards — and it is invisible if you only look at the
violation-rate column.

**Not every negation maps to a predicate.** *"authentic place, avoid touristy
spots"* is left unlabelled and excluded from the metrics: there is no
"touristy" attribute in the schema, and inventing a proxy would be fitting the
label to the available data. It is recorded in `UNMAPPABLE_QUERIES` rather than
quietly dropped. A deterministic filter can only enforce constraints the
catalog actually represents — everything else has to stay with the ranker.

**Standing caveat:** the catalog is real, but the constraint labels are the
author's reading of each query. They validate that the mechanism works, not
that real users phrase constraints this way.

**Status:** done — `constraints.py` + `filter_experiment.py` written and run;
raw results in `data/filter_results.csv`.
