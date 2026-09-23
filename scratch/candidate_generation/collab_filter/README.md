# Project 2: Collaborative-filtering candidate retrieval

Layer: candidate retrieval (collaborative filtering) — see `docs/architecture.md` §3.
`content_filter` (Project 1) retrieves by matching a query's text to a city's
description; this project retrieves by "travelers with taste similar to
yours liked these other cities" — no query, no text, just the interaction
log. The two run in parallel as separate retrieval sources.

## Preparation

Check yourself against [`PREPARATION_NOTES.md`](PREPARATION_NOTES.md) — background
concepts, not the lab's findings, so there's nothing to spoil by reading it first.

- [ ] Know the difference between implicit feedback (clicks) and explicit feedback (ratings), and what "confidence" means for the former
- [ ] Know why impressions carry no relevance signal, and what a per-user temporal holdout is
- [ ] Know, at a sketch level, what item-kNN, SVD, ALS, and BPR each optimize (don't need the derivations — just what's different)
- [ ] Know why `popularity`, not `random`, is the bar to beat here, and what a high popularity-correlation on top of a real win actually means
- [ ] Know what matrix density measures, and roughly where real datasets (e.g. MovieLens) sit on it

## Objectives

- Feel how differently four CF families behave on the *same* data — no
  training loop (item-kNN) vs. reconstructing the matrix (SVD/ALS) vs.
  optimizing ranking directly (BPR)
- Feel the difference between "beats popularity" and "learned taste, not
  just exposure" — and the one diagnostic that tells them apart
- Feel how a more realistic, sparse interaction log changes which model wins
- Practice cross-checking a hand-rolled implementation against a reference
  library instead of trusting your own math

## Questions to answer

**Modeling:**
1. Do all four models actually beat `popularity`? Is beating it enough to conclude a model "learned taste"?
2. Which model's ranking correlates most with raw popularity, and is that a red flag or an expected byproduct of exposure-biased training data?
3. Does confidence-weighting (graded save=2/click=1 vs. binary) change anything — for each model individually, and why or why not?
4. A model's score can climb while it's actually collapsing onto a popularity ranking. What in the results would expose that, and what would it look like if you only checked the headline metric?

**Computation & robustness:**
5. Which model needs the least compute, and does the cheapest model also win?
6. Does the winner at this project's default (small, dense) scale stay the winner once the log is realistically sparse?
7. Cross-checking ALS/BPR against a reference library (`implicit`) either confirms your implementation or catches a bug. What would each outcome look like, concretely, in the numbers?

**Before you touch the code, write down a guess for two of these** — which
of the four models you expect to win at the default settings, and whether
you expect that ranking to survive under realistic sparsity (see Part f).
Nothing here checks that guess for you.

## Background

*Definitions and equations live in [`PREPARATION_NOTES.md`](PREPARATION_NOTES.md) — this is just orientation.*

- **Data:** `scratch/synthetic_interactions` — synthetic users, personas, and
  an impression/click/save log over `content_filter`'s 560-city catalog.
  Built and ready: 5,000 users, 449,411 impressions, 2.8% click/save density.
- **A caveat:** every item in that default log gets at least one interaction
  — real logs (MovieLens-25M: ~0.25% density) have a long tail of near-cold
  items this doesn't. See Part f.
- **Four models** (`models.py`, all hand-rolled in NumPy/SciPy):

  | Model | Optimizes | Training |
  |---|---|---|
  | item-kNN | co-interaction cosine similarity | none — one similarity matrix |
  | SVD | matrix reconstruction error | one truncated-SVD solve |
  | ALS | weighted reconstruction error, confidence-scaled | alternating closed-form solves |
  | BPR | pairwise ranking order | SGD over sampled triples |

- **Vocabulary:** `recall@k` / `NDCG@k` / `hit_rate@k` — graded relevance
  (save=2, click=1); `pop_rho` = Spearman correlation between a model's
  ranking and raw popularity (§8 of the prep notes) — the diagnostic that
  separates "learned taste" from "relearned what's popular."

## Procedure

**Setup** (skip the venv creation if you already made one for `content_filter`
or `filter_constraints` — it's shared across all three):
```
python3 -m venv ../.venv               # once, from anywhere in candidate_generation/
source ../.venv/bin/activate           # re-run this in every new shell
pip install -r ../../requirements.txt
```

**Before Part a: generate the data.** This project's data isn't shipped in
the repo (`scratch/**/data/` is gitignored — see `.gitignore`'s comment on
that line). If `../../synthetic_interactions/data/interactions.csv` doesn't
exist yet, `sweep.py` will fail on a plain file-not-found with no other clue
why. Generate it once, with defaults:
```
cd ../../synthetic_interactions && python3 generate.py && cd ../collab_filter
```

**Part a — sanity run.** `python3 sweep.py --quick` (one config per model,
seconds). Do all four already beat `popularity` at a single arbitrary config?

**Part b — the full sweep.** `python3 sweep.py` (~20 min, writes
`data/cf_sweep.csv`). For each model, note its best NDCG@10 config and
whether it clears `popularity`. Which model wins outright?

**Part c — read the popularity diagnostic.** For your Part b winner, check
its `pop_rho` column. High correlation *and* a big margin over `popularity`
means real personalization on top of a popularity prior. Now run
`python3 sweep.py --models als` and scan its printed rows for the highest
`reg` value (the full grid already sweeps ALS's regularization up to 100) —
find the config that "wins" on NDCG while `pop_rho` sits near 1.0. What does
that combination actually mean?

**Part d — the confidence-weighting knob.** Compare `--weightings binary`
against the default (graded) for each model. Which models move, which
don't, and for BPR specifically — why would you expect zero movement before
you even run it (see prep notes §6)?

**Part e — cross-check.** Run `python3 verify_vs_implicit.py`. Does it
confirm your ALS/BPR implementations agree with the reference library's
ranking, or does it surface a discrepancy worth chasing down?

**Part f — realistic sparsity.** The Results so far come from a small, dense
catalog where nothing is ever truly cold. Build a sparser variant and re-run:
```
cd ../../synthetic_interactions
python3 scale_catalog.py --n-items 5000                                    # perturbs the 560 real cities up to 5,000 synthetic ones
python3 generate.py --catalog data/catalog_5000.csv --out-dir data/sparse_5000 \
    --n-users 5000 --popularity-skew 1.5                                   # same user count, ~9x sparser log
cd ../collab_filter
python3 sparsity_sweep.py --k 10 \
    --variant "dense (560)":../../synthetic_interactions/data:../data/cities.csv \
    --variant "sparse (5000)":../../synthetic_interactions/data/sparse_5000:../../synthetic_interactions/data/catalog_5000.csv
```
Try more `--n-items` sizes (2000, 20000, ...) for more points on the curve.
Hyperparameters are held fixed across densities on purpose (see
`sparsity_sweep.py`'s docstring) — this isolates the density effect rather
than re-tuning at each point. Does your Part b winner stay the winner? Does
any model's `pop_rho` climb toward 1.0 as density drops — the same collapse
pattern from Part c, now driven by sparsity instead of over-regularization?

## Deliverables

Fill these in as you go in [`DELIVERABLES.md`](DELIVERABLES.md) — a template with every question and table already laid out, so you're writing answers, not reformatting.

1. Your own answer to each of the seven questions above, in your own numbers
2. The full-grid table from Part b: best config per model, against `random`/`popularity`
3. The config you found in Part c that "wins" on NDCG while its `pop_rho` gives away that it's collapsed toward popularity, and why that combination is misleading if read from NDCG alone
4. Your Part d comparison, and an explanation for BPR's result specifically
5. What Part e's cross-check told you — agreement, or a bug worth fixing
6. The dense-vs-sparse comparison table from Part f, and whether your Part b winner survived
7. One paragraph: which of the four models would you reach for first on a real, sparse, large-catalog dataset, and why that might differ from your Part b answer

## Notes

- `models.py`, `sweep.py`, `verify_vs_implicit.py`, `scale_catalog.py` (in
  `synthetic_interactions`), and `sparsity_sweep.py` are all implemented and
  runnable.
- A prior write-up with actual measured numbers (the dense-catalog sweep,
  predating the sparsity extension) exists in this file's git history, if
  you want to check your Deliverables against it after — not before.
