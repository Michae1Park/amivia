# Offline eval harness

Layer: cross-cutting — scores any recommender against held-out ground truth,
not tied to one funnel stage.

**Problem:** every project that consumes `synthetic_interactions` needs a
consistent way to answer "is this actually better," instead of eyeballing
results per project. This harness is the shared scorer.

**Scope:** ground truth here is keyed by `user_id` (from
`synthetic_interactions`), so it fits `collab_filter`, `rank_two_tower`, and
`feedback_taste_profile` directly. `content_filter` takes a free-text query,
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
(`rank_two_tower`, `feedback_taste_profile`) plug in by implementing that
method and adding themselves there.

`collab_filter` is the first project wired in this way:

```
python3 run_eval.py --cf                          # + item_knn, svd, als, bpr
python3 run_eval.py --cf --cf-weighting binary    # binary instead of graded input
```

It loads that project's four models at their swept-best settings
(`CF_BEST_PARAMS` in `run_eval.py`); the full grid and the interpretation live
in `candidate_generation/collab_filter/README.md`. The import is lazy, so a
bare `run_eval.py` run stays independent of it.

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
- `LLMRecommender` *(opt-in, `--llm`)* — the whole 560-city catalog in the
  prompt, the user's train history in the user turn, indices back out. No
  training, no embeddings, no interaction matrix.

## The LLM baseline

**Why it's here:** at 560 items an LLM with the full catalog in context is a
genuinely viable recommender, so it's the honest bar for the classic funnel
to clear. It also marks where that stops being true — the approach dies as
soon as the catalog outgrows the context window, which is the constraint the
funnel exists to solve (see `docs/roadmap-to-service.md`).

```
pip install anthropic          # not currently in the venv
export ANTHROPIC_API_KEY=...   # or: ant auth login

python3 run_eval.py --llm --sample-users 200
python3 run_eval.py --llm --sample-users 200 --llm-model claude-sonnet-5 --llm-effort medium
```

**Costs money per user**, so it is opt-in and effectively requires
`--sample-users`. `--sample-users` subsamples *every* model identically —
otherwise the LLM's numbers aren't comparable to the baselines it's meant to
be measured against. Three things keep the bill down:

- the catalog block (~7k tokens, identical for every user) is a cached prompt
  prefix, so per-user input bills at ~10% of list rate
- responses are memoised to `data/llm_cache.json` — re-runs are free
- `prefetch()` makes one call to warm the cache, then fans out concurrently
  (parallel requests sharing a prefix would all miss it otherwise)

`run_eval.py` prints a token/cache summary after the metrics table so the
cache hit rate is visible rather than assumed.

**Read its score against `popularity`, not `random`** — the LLM ranks by
inferred taste, so the exposure bias documented below applies to it in full.

## A finding worth keeping in mind, not a bug

Running the three baselines (`python3 run_eval.py`), `popularity` beats
`oracle_persona` on every metric, despite `oracle_persona` using the exact
formula that generated the labels:

```
model              k  hit_rate    recall      ndcg
random             5    0.0207    0.0053    0.0048
popularity         5    0.1585    0.0473    0.0457
oracle_persona     5    0.1029    0.0270    0.0276
```

*(Numbers refreshed after `interactions.csv` was regenerated with dwell,
`itinerary_add`, `not_interested` and session filters — the earlier table in
this README predated that log and no longer reproduced. The ordering, and
therefore the finding below, is unchanged.)*

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

`LLMRecommender` added and exercised offline (prompt construction, index
validation, seen/duplicate filtering) — **not yet run against the API**: the
`anthropic` SDK isn't installed in the venv and no credentials are set, so it
has no scores in this README yet.
