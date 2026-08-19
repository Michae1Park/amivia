# Data dictionary

Column-by-column reference for the three CSVs the recsys projects consume.
`cities.csv` is real (sourced) data; `users.csv` and `interactions.csv` are
synthetic, produced by [`generate.py`](generate.py) — see
[`README.md`](README.md) for *how* and *why* they're generated. This
file is just "what does each field mean."

## `candidate_generation/data/cities.csv` — item catalog (560 rows)

Sourced from the [Worldwide Travel Cities](https://www.kaggle.com/datasets/furkanima/worldwide-travel-cities-ratings-and-climate)
Kaggle dataset. Every other file below joins to this one via `id`.

| Column | Type | Meaning |
|---|---|---|
| `id` | UUID string | Primary key. This is what `interactions.csv.item_id` refers to. |
| `city`, `country` | string | Display name. |
| `region` | enum | One of `europe, asia, north_america, south_america, africa, oceania, middle_east`. |
| `short_description` | text | Free-text blurb — the field `embed_retrieve` embeds. |
| `latitude`, `longitude` | float | Standard decimal degrees. |
| `avg_temp_monthly` | JSON string | `{"1": {"avg": 3.7, "max": 7.8, "min": 0.4}, ..., "12": {...}}` — keys `"1"`-`"12"` are calendar months, values are °C. |
| `ideal_durations` | JSON string (list) | Subset of `["Day trip", "Weekend", "Short trip", "One week", "Long trip"]` — a city can have several. |
| `budget_level` | enum | `Budget` (145 cities) · `Mid-range` (339) · `Luxury` (76). |
| `culture, adventure, nature, beaches, nightlife, cuisine, wellness, urban, seclusion` | int 1-5 | Tag ratings, 1 = not at all, 5 = defining trait of the city. This is the 9-dim vector personas and interactions score against. |

## `synthetic_interactions/data/users.csv` — synthetic users (5,000 rows)

One row per synthetic user. Persona columns are **eval-only ground truth**
— never fed to a model as a feature, only used to check whether a model's
learned structure lines up with the hidden persona.

| Column | Type | Meaning |
|---|---|---|
| `user_id` | string | `u000000`-`u004999`. |
| `primary_persona` | enum | One of the 6 personas below (uniform random, ~830-850 users each). |
| `secondary_persona` | enum | A different persona, also uniform random. |
| `mix_weight` | float, [0.5, 1.0] | How much the user's true taste leans toward `primary_persona` vs. `secondary_persona`. `1.0` = behaves exactly like the primary persona; `0.5` = an even 50/50 blend. Drawn per-user as `clip(normal(persona_mix_purity, 0.07), 0.5, 1.0)`, where `persona_mix_purity` (default 0.8) is a generation-time knob. Does **not** capture the extra per-tag Gaussian noise also added to the user's preference vector — that noise isn't persisted anywhere, only baked into which items they click. |

**The 6 personas** (each a weighted-tag archetype, defined in `generate.py`):

| Persona | Weighted toward (tags) | Preferred `budget_level` |
|---|---|---|
| Budget Backpacker | adventure, culture | Budget |
| Luxury Relaxer | wellness, beaches, cuisine | Luxury |
| Nightlife Seeker | nightlife, urban | Mid-range |
| Culture Buff | culture, cuisine, urban | Mid-range |
| Nature & Adventure | nature, adventure, seclusion | Budget, Mid-range |
| Family Beach | beaches, wellness | Mid-range |

## `synthetic_interactions/data/interactions.csv` — event log (529,343 rows)

One row per event. Events nest: every `click` has a parent `impression`;
every `save` has a parent `click`. This grading (impression < click < save)
is the implicit-feedback strength signal.

| Column | Type | Meaning |
|---|---|---|
| `interaction_id` | int | Row sequence number, not meaningful beyond uniqueness. |
| `user_id` | string | Foreign key into `users.csv`. |
| `item_id` | UUID string | Foreign key into `cities.csv.id`. |
| `event_type` | enum | `impression` (450,852 rows, 85.2%) · `click` (70,441, 13.3%, = 15.6% of impressions) · `save` (8,050, 1.5%, = 11.4% of clicks). |
| `session_id` | string | `{user_id}-s{NNN}` — groups impressions shown together in one sitting. |
| `timestamp` | ISO 8601 | Spread across a synthetic 2-year window (2024-01-01 onward) per user, so recency-decay experiments have a real timeline. |

Whether an impression becomes a click (and a click a save) is driven by an
**affinity score** — the dot product of the user's persona-blended tag
weights and the item's tag vector, plus a bonus if the item's `budget_level`
matches the user's preferred budgets — passed through a sigmoid. Impressed
items themselves are sampled by item popularity (power-law), independent of
the user's persona, so popularity bias is present in what's *shown*, and
persona fit only affects what's *clicked*.

## Regenerating

```
python3 generate.py --n-users 5000 --persona-mix-purity 0.8 --random-seed 42
```
All the counts above are from the checked-in default run; see `README.md`
for the full knob list (`sessions_per_user`, `popularity_skew`,
`click_threshold`, `save_threshold`, etc.).
