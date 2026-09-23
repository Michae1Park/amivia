# Shared prerequisite: synthetic user-item interactions

- Three toy projects need user-item interaction history no public dataset
  here provides: `collab_filter`, `rank_two_tower`, `feedback_taste_profile`.
- This project generates that history once, as a shared resource all three
  consume.
- **Items:** the existing city catalog from `content_filter` (560 cities, see
  [`DATA.md`](DATA.md) for columns).
- **Users/interactions:** synthetic, via [`generate.py`](generate.py).

## Personas

Six hidden traveler archetypes, each a preference weight vector over the 9 tag
columns plus a `budget_level` preference:

| Persona | Weighted toward | Budget |
|---|---|---|
| Budget Backpacker | adventure, culture | Budget |
| Luxury Relaxer | wellness, beaches, cuisine | Luxury |
| Nightlife Seeker | nightlife, urban | Mid-range |
| Culture Buff | culture, cuisine, urban | Mid-range |
| Nature & Adventure | nature, adventure, seclusion | Budget/Mid-range |
| Family Beach | beaches, wellness | Mid-range |

- **Ground truth for evaluation only** — e.g. "do the embeddings a model
  learns actually cluster by persona?"
- Never fed to a model as an input feature, or the exercise would be trivial.

## Why it's built this way

Three design choices keep this a real test rather than something a model
could solve by memorizing:

- **Users are persona blends, not pure instances.**
  - Each user leans toward a primary persona, with a smaller weight from a
    second (`persona_mix_purity` knob), plus individual noise.
  - Pure, unmixed users would be trivial pattern matching, not a real test of
    collaborative filtering.
- **Interactions are an exposure funnel, not a list of "liked" items.**
  - Impressions are sampled by item popularity alone.
  - Clicks and saves are then rolled probabilistically from persona fit.
  - This mirrors real logs, where what wasn't shown differs from what was
    shown but ignored.
  - It makes popularity bias a real confound to correct for, not an
    assumption.
  - Exact mechanics (affinity score, sigmoid, thresholds) are in
    [`DATA.md`](DATA.md).
- **The funnel carries graded signal and an explicit negative, not just
  impression/click/save.**
  - `dwell_seconds` on each click — longer for a better-fit item (same idea
    as watch-time on Netflix/YouTube).
  - A save can escalate to `itinerary_add` — a later, stronger commitment
    (cart vs. purchase).
  - A badly-mismatched impression can roll an explicit `not_interested`
    instead of silent non-engagement.
  - Each session logs a templated `query_text` plus any
    `filter_budget_level` / `filter_required_tag` applied — intent signal for
    `filter_constraints` / `intent_parsing`, separate from what was clicked.
  - See `sessions.csv` and the new `interactions.csv` columns in
    [`DATA.md`](DATA.md).

## Knobs (the point of making this synthetic)

| Knob | Effect |
|---|---|
| `n_users`, `sessions_per_user`, `impressions_per_session` | overall data volume / sparsity |
| `persona_mix_purity` | how cleanly separable user taste is — lower it to see how much CF/two-tower quality degrades as preference gets noisier |
| `popularity_skew` | how much popularity dominates over personalization signal — raise it to see whether a model just learns to recommend popular items |
| `click_threshold`, `save_threshold` | overall signal sparsity — how rare are positives |
| `random_seed` | reproducibility |
| catalog size (via `scale_catalog.py`, below) | the knob that actually gets close to real-world sparsity — see next section |

### Catalog scale, and why it matters more than the knobs above

The default catalog is the 560 real cities, and at 5,000 users every one of
them gets at least one click or save — 2.8% matrix density, no cold items at
all. Real interaction logs (MovieLens-25M: ~0.25% density) are sparse mainly
because the **catalog** is huge, not because any of the knobs above were
tuned toward scarcity. `scale_catalog.py` perturbs the 560 real cities up to
a larger synthetic catalog (same idiom `content_filter/ann_benchmark.py` uses
to scale to 1M for its ANN sweep — perturbed real vectors, not uniform random
ones), so `generate.py` can be pointed at it to produce a log with genuinely
cold items and a long popularity tail:

```
python3 scale_catalog.py --n-items 5000
python3 generate.py --catalog data/catalog_5000.csv --out-dir data/sparse_5000 \
    --n-users 5000 --popularity-skew 1.5
```

At 5,000 items this drops density to ~0.29% (MovieLens-25M range) with ~38%
of items getting zero training interactions — `collab_filter/sparsity_sweep.py`
is what consumes a variant like this; see that project's README.

## Status

Done — `generate.py` written and run with defaults (5,000 users,
`data/users.csv` + `data/interactions.csv` + `data/sessions.csv`): 449,411
impressions, 20.0% CTR (mean dwell 24s), 13.6% of clicks saved, 7.6% of saves
escalate to `itinerary_add`, 1.0% of impressions get an explicit
`not_interested`, 47.7% of sessions apply a query filter.
