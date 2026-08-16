# Offline eval harness

Layer: cross-cutting — scores any recommender against held-out ground truth,
not tied to one funnel stage.

**Problem:** every project that consumes `synthetic_interactions` needs a
consistent way to answer "is this actually better," instead of eyeballing
results per project. This harness is the shared scorer.

**Scope:** ground truth here is keyed by `user_id` (from
`synthetic_interactions`), so it fits `collab_filter`, `rank_two_tower`, and
`feedback_taste_profile` directly. `embed_retrieve` takes a free-text query,
not a user, so it isn't scored here — it stays on `batch_test.py`'s
qualitative checks until there's a query-to-user mapping worth designing.

**Data / split:** per-user temporal holdout over
`synthetic_interactions/data/interactions.csv` — each user's most recent
`test_fraction` of sessions become the test set, the rest is train.
Held-out relevance is graded: `save=2`, `click=1`, impressions carry no
signal. Any item a user saw in train (any event type) is excluded from that
user's recommendable candidates, same as standard leave-one-out CF eval.

**Metrics:** `hit_rate_at_k`, `recall_at_k`, `ndcg_at_k` — see
`metrics.py`. Recall/NDCG are computed only for users with at least one
held-out click/save; users with none are skipped (nothing to measure
against).

**Interface:** any recommender is just an object with
`.recommend(user_id: str, k: int) -> list[item_id]`. `run_eval.py` scores
whatever's registered in its `recommenders` dict — future projects
(`collab_filter`, `rank_two_tower`, `feedback_taste_profile`) plug in by
implementing that method and adding themselves there.

**Baselines (`baselines.py`)** — built to sanity-check the harness itself,
not to represent real candidates:
- `RandomRecommender` — floor.
- `PopularityRecommender` — same ranked list for everyone, by train
  click/save counts.
- `OraclePersonaRecommender` — ranks by the exact affinity formula
  `generate.py` used to produce clicks/saves, reconstructed from each
  user's ground-truth persona + `mix_weight` (idiosyncratic noise isn't
  persisted, so this is an approximation, not a true ceiling). Persona
  labels are eval-only ground truth and never a real model's input — this
  exists purely to prove the harness can detect "near-ideal" behavior.

## A finding worth keeping in mind, not a bug

Running the three baselines (`python3 run_eval.py`), `popularity` beats
`oracle_persona` on every metric, despite `oracle_persona` using the exact
formula that generated the labels:

```
model              k  hit_rate    recall      ndcg
random             5    0.0173    0.0052    0.0043
popularity         5    0.1975    0.0676    0.0619
oracle_persona     5    0.0466    0.0150    0.0131
```

This is exposure bias, not a broken harness. In `synthetic_interactions`,
*impressions* are sampled by popularity, not affinity — an item only gets a
chance to be clicked/saved if it was shown in the first place. So the
held-out positive set is itself popularity-skewed, and a pure-affinity
ranker (`oracle_persona`) gets "punished" for recommending well-matched but
rarely-impressed items that had near-zero chance of ever appearing in
ground truth. This is the same logged-bandit-feedback pitfall real offline
recsys eval has to account for (Booking.com/Netflix-style off-policy
evaluation) — and it's exactly what `synthetic_interactions`'s
`popularity_skew` knob is designed to let you feel.

Takeaway for future projects here: a model that beats `popularity` on this
harness is doing something real. A model that merely *ties* popularity may
just be relearning "recommend what's popular" — worth checking the actual
recommended lists, not just the metric, before declaring a win.

## Status

Done — `metrics.py`, `data.py`, `baselines.py`, `run_eval.py` written and
validated: `random` < `oracle_persona`/`popularity` as expected (harness
discriminates), with the popularity-vs-oracle ordering above worth reading
before trusting any future project's numbers at face value.
