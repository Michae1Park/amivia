# Shared prerequisite: synthetic user-item interactions

- Three toy projects need user-item interaction history that no public
  dataset here provides: `collab_filter`, `rank_two_tower`, and
  `feedback_taste_profile`
- This project generates that history once, as a shared resource all three
  consume

**Items:** the existing city catalog from
`scratch/candidate_generation/embed_retrieve/data/Worldwide Travel Cities Dataset (Ratings and Climate).csv`
(560 cities, each already tagged 1-5 on `culture`, `adventure`, `nature`,
`beaches`, `nightlife`, `cuisine`, `wellness`, `urban`, `seclusion`, plus a
`budget_level`).

**Users/interactions:** synthetic, generated to have:
- Real latent structure (so collaborative filtering / two-tower models have
  something genuine to recover)
- Realistic noise, sparsity, and popularity bias (so a naive approach
  visibly underperforms and tuning knobs actually matter)

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

- Persona labels are **ground truth for evaluation only** (e.g. "do the
  embeddings a model learns actually cluster by persona?")
- Never fed to a model as an input feature — that would defeat the point of
  the exercise

## User generation

Each synthetic user:
- Gets one primary persona (uniform random) and a small blend of a second persona
  (`persona_mix_purity` controls how dominant the primary is, e.g. 0.7-0.9)
- Gets individual idiosyncratic noise added to their effective weight vector, so
  users aren't perfectly separable by persona alone

- Deliberate: if users were pure, unmixed persona instances, recovering them
  would be trivial pattern matching, not a real test of collaborative
  filtering or two-tower training

## Interaction generation

Real interaction logs aren't "a list of things the user liked" — they're an
exposure funnel, and what wasn't shown is different from what was shown but
ignored. Modeling that distinction is the point:

1. **Item popularity**: each item gets a popularity multiplier drawn from a power-law
   distribution (`popularity_skew` knob controls the exponent) — some cities are just
   more shown/visited regardless of fit, mirroring real popularity bias.
2. **Impressions**: per session, sample ~10-20 items weighted by popularity (not
   persona fit) — this is what a naive, non-personalized surface would have shown.
3. **Clicks**: for each impression, roll a click with probability from
   `sigmoid(affinity_score - threshold)`, where `affinity_score` is the dot product of
   the user's effective weight vector and the item's tag vector, plus a budget-match
   bonus.
4. **Saves**: conditional on a click, roll a save with a smaller, stronger-signal
   probability — gives a graded implicit-feedback strength (impression < click <
   save) rather than flat binary labels.
5. **Timestamps**: sessions are spread across a synthetic 2-year window per user, so
   recency-decay experiments (`feedback_taste_profile`) have a real timeline to decay
   over.

## Output schema

- `data/users.csv` — `user_id, primary_persona, secondary_persona, mix_weight`
  (persona columns are eval-only ground truth, see above)
- `data/interactions.csv` — `interaction_id, user_id, item_id, event_type
  (impression|click|save), session_id, timestamp`

`item_id` matches the `id` column in `embed_retrieve`'s catalog CSV — no item table
is duplicated here.

## Knobs (the point of making this synthetic)

| Knob | Effect |
|---|---|
| `n_users`, `sessions_per_user`, `impressions_per_session` | overall data volume / sparsity |
| `persona_mix_purity` | how cleanly separable user taste is — lower it to see how much CF/two-tower quality degrades as preference gets noisier |
| `popularity_skew` | how much popularity dominates over personalization signal — raise it to see whether a model just learns to recommend popular items |
| `click_threshold`, `save_threshold` | overall signal sparsity — how rare are positives |
| `random_seed` | reproducibility |

## Status

Design only — `generate.py` not yet written.
