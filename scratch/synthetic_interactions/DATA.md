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
| `short_description` | text | Free-text blurb — the field `content_filter` embeds. |
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

The 6 personas (weighted-tag archetypes) are defined in [`README.md`](README.md#personas).

## `synthetic_interactions/data/interactions.csv` — event log (556,536 rows)

One row per event. Events nest:

- `click` → parent `impression`
- `save` → parent `click`
- `itinerary_add` → parent `save`
- `not_interested` → sibling of `click` (rolled only for impressions that
  weren't clicked)

This grading (impression < click < save < itinerary_add, plus the
`not_interested` explicit negative) is the implicit-feedback strength signal.

| Column | Type | Meaning |
|---|---|---|
| `interaction_id` | int | Row sequence number, not meaningful beyond uniqueness. |
| `user_id` | string | Foreign key into `users.csv`. |
| `item_id` | UUID string | Foreign key into `cities.csv.id`. |
| `event_type` | enum | `impression` (449,411, 80.8%) · `click` (89,672, 16.1%, = 20.0% of impressions) · `save` (12,173, 2.2%, = 13.6% of clicks) · `itinerary_add` (930, 0.2%, = 7.6% of saves) · `not_interested` (4,350, 0.8%, = 1.0% of impressions). |
| `session_id` | string | `{user_id}-s{NNN}` — groups impressions shown together in one sitting; foreign key into `sessions.csv`. |
| `timestamp` | ISO 8601 | Spread across a synthetic 2-year window (2024-01-01 onward) per user, so recency-decay experiments have a real timeline. |
| `dwell_seconds` | float, or blank | Only set on `click` rows — time spent on the item's detail page, drawn from a log-normal that scales with how far the item clears the click-affinity threshold (a better-fit click dwells longer, same idea as Netflix/YouTube watch-time). Median ~20s, p95 ~54s, capped at 900s. Blank for every other event type. |

Whether an impression becomes a click, save, or `itinerary_add` is driven by
an **affinity score**:

- **Score** = dot product of the user's persona-blended tag weights and the
  item's tag vector, plus a bonus if `budget_level` matches the user's
  preferred budgets.
- Passed through a sigmoid; each event type needs a higher threshold than the
  last (`click_threshold < save_threshold < itinerary_threshold`).
- `not_interested` uses the *opposite* tail of the same score — a low, capped
  rate for strongly *negative* affinity, modeling explicit rejection rather
  than mere disinterest.
- Impressions are sampled by item popularity (power-law), independent of
  persona — except in filtered sessions (see `sessions.csv` below), which
  sample only from the filter-matching subset.
- Net effect: popularity bias shows up in what's *shown*; persona fit only
  affects what's *clicked*.

## `synthetic_interactions/data/sessions.csv` — session log (29,944 rows)

One row per session — the query/filter context an impression batch was shown
under. `session_id` is the join key into `interactions.csv`.

| Column | Type | Meaning |
|---|---|---|
| `session_id` | string | `{user_id}-s{NNN}`, matches `interactions.csv.session_id`. |
| `user_id` | string | Foreign key into `users.csv`. |
| `timestamp` | ISO 8601 | Session start time — same instant as that session's `impression` rows. |
| `query_text` | text | A templated free-text query built from 1-2 of the user's *primary persona*'s tags (e.g. "somewhere good for beaches and coastline and a wellness retreat"). Not fed into affinity scoring — it's a stated-intent signal for `filter_constraints`/`intent_parsing` to consume, generated from persona but not a literal restatement of it. |
| `filter_budget_level` | enum, or blank | Set ~30% of the time, to one of the user's preferred `budget_level`s. When set, that session's impressions are drawn only from items matching it. |
| `filter_required_tag` | enum, or blank | Set ~25% of the time, to one of the tags mentioned in `query_text`. When set, impressions are drawn only from items rated ≥4 on that tag. If a filter combination leaves too few matching items for the session's impression count, generation falls back to the unfiltered catalog for that session (the field still records what was asked for). |

47.7% of sessions have at least one filter set.

## Regenerating

```
python3 generate.py --n-users 5000 --persona-mix-purity 0.8 --random-seed 42
```
All the counts above are from the checked-in default run; see `README.md`
for the full knob list (`sessions_per_user`, `popularity_skew`,
`click_threshold`, `save_threshold`, etc.). `not_interested` rate, dwell-time
distribution, `itinerary_add` threshold, and filter rates are internal
constants in `generate.py` (not CLI flags), same as `BUDGET_MATCH_BONUS`.
