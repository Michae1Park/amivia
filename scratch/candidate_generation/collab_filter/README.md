# Project 2: Collaborative-filtering candidate retrieval

Layer: candidate retrieval (collaborative filtering) — see `docs/architecture.md` §3.

**Problem:** `embed_retrieve` retrieves cities by matching content (descriptions)
to a query. It has no notion of "travelers with taste similar to yours liked
these other cities" — the other major candidate-generation signal production
systems run in parallel with content-based retrieval.

**Data:** `scratch/synthetic_interactions` — synthetic users, personas, and an
impression/click/save interaction log built on top of `embed_retrieve`'s city
catalog. **Built and ready** — 5,000 users, 450,852 impressions.

**Algorithms to compare:**
- Matrix factorization (ALS or SVD) over the user-item interaction matrix
- Item-based neighborhood collaborative filtering (cosine similarity over
  co-interaction patterns)
- Implicit-feedback Bayesian Personalized Ranking (BPR)

**Knobs to tune:** latent dimension, regularization strength, how implicit
feedback (impression/click/save) is weighted vs. treated as explicit rating.

**Usage:**
```
python3 sweep.py                  # the full grid (~20 min), writes data/cf_sweep.csv
python3 sweep.py --quick          # one config per model
python3 verify_vs_implicit.py     # cross-check ALS/BPR against the `implicit` library

cd ../../eval && python3 run_eval.py --cf     # score them beside the baselines
```

**Implementation:** all four are hand-rolled in NumPy/SciPy (`models.py`) —
the point of the project is to feel how they differ, which reading library call
signatures does not teach. Each implements the eval harness's
`.recommend(user_id, k)` contract plus `.scores(user_id)` over the full catalog,
which the popularity diagnostic needs.

Two implementation notes worth keeping:

- **The interaction matrix dedupes to the strongest grade per (user, item)**,
  matching `eval/data.py`'s `max(cur, grade)`. Summing repeat events and
  clipping instead would score two clicks in separate sessions identically to a
  save — inventing signal that isn't in the log.
- **ALS and BPR are cross-checked against `implicit`** (`verify_vs_implicit.py`)
  rather than trusted. A sign error in a BPR gradient or a mis-derived ALS normal
  equation still produces plausible numbers, and this project's deliverable is an
  interpretation — a wrong one is worse than none.

## Experiment

**Setup:** `synthetic_interactions` (5,000 users, 450,852 impressions), per-user
temporal holdout via `eval/`. Graded relevance: save=2, click=1.

**Conditions:** ALS · SVD · item-kNN (cosine over co-interaction) · BPR.
Sweep latent dim {16, 32, 64, 128} and regularisation; for implicit feedback
compare binary (click∪save) against confidence-weighted (save=2, click=1).

**Metrics:** recall@k, NDCG@k, hit-rate@k for k ∈ {5, 10, 20}, against
`random` / `popularity` / `oracle_persona`.

**Interpretation:** the bar is `popularity`, not `random`. Because impressions
are popularity-sampled, CF trained on this log may simply *relearn popularity*
— so also report Spearman correlation between each model's per-user ranking
and the global popularity ranking. High correlation plus a small NDCG gain
means it learned exposure, not taste, and should be reported as such.

## Results

112 configs swept. Best per model by NDCG@10, against the harness baselines
(4,675 test users with ≥1 held-out positive). Raw grid in `data/cf_sweep.csv`;
reproduce the table below with `cd ../../eval && python3 run_eval.py --cf`.

| model | best config | NDCG@5 | NDCG@10 | hit@10 | recall@10 | pop ρ |
|---|---|---|---|---|---|---|
| random | — | 0.0048 | 0.0080 | 0.0456 | 0.0116 | — |
| popularity | — | 0.0457 | 0.0580 | 0.2535 | 0.0796 | — |
| oracle_persona | — | 0.0276 | 0.0380 | 0.1807 | 0.0518 | — |
| **item_knn** | n_neighbors=560, confidence | **0.0774** | **0.0926** | **0.3594** | **0.1180** | 0.637 |
| bpr | dim=64, reg=0.1 | 0.0716 | 0.0859 | 0.3416 | 0.1117 | 0.748 |
| als | dim=16, reg=10, α=1, binary | 0.0628 | 0.0768 | 0.3151 | 0.0990 | 0.347 |
| svd | dim=16, confidence | 0.0581 | 0.0712 | 0.2948 | 0.0931 | 0.288 |

**All four clear the real bar.** `popularity`, not `random`, is what counts here,
and every model beats it — item-kNN by 60% on NDCG@10, BPR by 48%, ALS by 32%,
SVD by 23%. Given the exposure bias documented in `eval/README.md`, that was not
a given: this log's held-out positives track *what was shown*, and a model that
merely relearned exposure would land on top of `popularity`, not well past it.

**They also beat `oracle_persona` — by a lot — and that is the interesting part.**
The oracle ranks by the exact affinity formula that generated the clicks and
still scores 0.0380, less than half item-kNN's 0.0926. There is no contradiction:
the oracle ranks by *latent taste* over the whole catalog, including cities a
user was never shown and therefore could never have clicked. CF learns from the
log, so it inherits the log's exposure shape and predicts taste *within the set
users actually see*. On an exposure-biased test set that is the winning strategy,
and it is a compact illustration of why offline recsys numbers flatter models
trained on logged feedback.

**The simplest model wins.** Item-kNN has no training loop and no latent space —
a 560×560 cosine similarity matrix, built in well under a second — and it beats
all three factorization models. Its best setting is `n_neighbors=560`, i.e. no
truncation at all: pruning to each item's strongest neighbours only ever hurt.
At 560 items and 64k interactions there is not enough data for a learned latent
space to pay for itself, which is worth remembering before reaching for a
two-tower model in Project 6.

**Where the popularity diagnostic earns its keep.** The pre-specified reading —
high correlation plus a small NDCG gain means it learned exposure, not taste —
applies almost perfectly to one config, and it is an ALS one:

| ALS (dim=16, binary) | NDCG@10 | pop ρ |
|---|---|---|
| reg=1 | 0.0718 | 0.282 |
| **reg=10** | **0.0768** | 0.347 |
| reg=100, α=1 | 0.0588 | **0.949** |

At `reg=100` ALS scores 0.0588 against `popularity`'s 0.0580 — a 1.4% "win" — with
a 0.949 rank correlation to the global popularity ordering. It did not learn
taste; over-regularisation collapsed it onto a near-rank-1 solution that *is*
popularity. Judged on NDCG alone it looks like a modest success. The correlation
column is what exposes it, which is the entire reason the experiment
pre-specified that column.

Read the same way, the genuine winners are honest but not innocent: item-kNN at
ρ=0.637 and BPR at ρ=0.748 are substantially popularity-correlated while beating
popularity by 48–60%. They are doing real personalisation *on top of* a strong
popularity prior, not instead of one. ALS at ρ=0.347 is the least
popularity-driven of the four and pays for it in raw score — the same
exposure-bias tax `oracle_persona` pays.

**The confidence-weighting knob does almost nothing.** Graded (save=2, click=1)
vs. binary (click∪save) moves NDCG@10 by 2% for item-kNN (0.0926 vs 0.0906),
0.3% for SVD, and not at all for ALS. That is a clean negative result on one of
the three knobs this project set out to test: with only two grade levels and
saves making up ~14% of clicks, the graded matrix is nearly the binary one.
Distinguishing implicit-feedback strength would need the richer signal
`synthetic_interactions` already logs — `dwell_seconds`, `itinerary_add`, and
the explicit `not_interested` negative — none of which the standard formulations
here consume.

**BPR ignores the knob entirely, by construction.** Standard BPR samples
(user, positive, negative) triples from the *positions* of observed interactions
and never reads their values, so binary and confidence weighting produce a
bit-identical model. The sweep runs it once and labels the weighting `either`
rather than printing duplicate rows that would look like independent evidence.

**Low latent dimension wins throughout** — ALS and SVD both peak at dim=16, BPR
at 64, with 128 worse across the board. 64k nonzeros over 560 items does not
support much capacity.

**What the `implicit` cross-check actually caught.** Ranking agreement with the
reference library is moderate (Spearman ρ 0.45 for ALS, 0.50 for BPR; top-10
overlap 0.46 / 0.31) — two valid solutions differing in the ranking tail, not a
broken gradient, confirmed by NDCG tracking within a few percent across four
hyperparameter settings. But running it surfaced something the sweep alone would
have hidden: `implicit` scored far better at `reg=10` than anything in the
original grid, which topped out at `reg=1`. ALS was simply under-regularised,
and on the first pass it scored *below* `popularity` and would have been written
up as the weakest model. The grid now runs to `reg=100`. **The check paid for
itself not by finding a bug, but by catching a false negative.**

**Standing caveat:** this is synthetic data. It validates that the four
implementations work and that the diagnostics discriminate; it says nothing
about how real travellers behave. In particular the persona structure is
generated, so "CF recovers taste structure here" does not imply it would recover
anything on a real log of this size.

**Status:** done — `models.py`, `sweep.py`, `verify_vs_implicit.py` written,
full grid run, wired into `eval/run_eval.py --cf`.
