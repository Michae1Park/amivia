# Project 5: Ranking over structured features

Layer: precise ranking (GBDT / learning-to-rank) — see `docs/architecture.md`
§5b.

**Problem:** retrieval (Project 1/2) answers "which destinations are
relevant at all" — ranking answers "in what order should they be shown,"
scoring candidates on richer structured attributes (rating, cost, climate,
popularity) than embedding similarity alone gives.

**Data:** [Popular Tourist Destinations and Their Features](https://www.kaggle.com/datasets/cosmox23/popular-tourist-destinations-and-their-features).
Download the CSV and place it at: `data/Tourist_Destinations.csv`

**Algorithms to compare** (LightGBM/XGBoost, same features, different
objective):
- Pointwise regression — predict a relevance/rating target directly
- Pairwise (LambdaMART) — trained on preference pairs instead of raw scores
- Listwise (LambdaRank / ListNet) — optimizes the whole ranked list's order
  directly, closest to what production rankers actually use

**Knobs to tune:** `num_leaves`, learning rate, boosting rounds, objective
function.

## Experiment

**Setup:** LightGBM over `synthetic_interactions`, grouped by user, same
`eval/` split. Features: item tag vector, budget match, item popularity,
taste-profile dot product.

**Conditions:** pointwise regression · pairwise (LambdaMART) · listwise
(LambdaRank). Sweep `num_leaves`, learning rate, boosting rounds.

**Metrics:** NDCG@10 (primary — it is what the pairwise/listwise objectives
optimise), recall@k secondary.

**Interpretation:** pairwise/listwise should beat pointwise on NDCG; if they
don't, check positives-per-user first, since ranking losses need several
positives per group to have anything to order. Report that distribution
alongside the result — "too few positives per user" is the likely explanation
and a legitimate finding.

**Results:** *not yet run.*

**Status:** scaffolded — dataset chosen, no code yet.
