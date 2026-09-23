"""Out-of-distribution (OOD) hand-labelled eval set.

`filter_constraints.constraints.LABELLED_QUERIES` (15 queries) is the seed —
reused via `schema.Intent.from_predicates` rather than re-typed, so the two
copies can't drift apart. This file adds 35 more in the same style to reach
~50, split across the same failure categories plus one this project needs
that `filter_constraints` didn't: queries with NO hard constraint at all, to
catch a model that over-eagerly invents one from soft descriptive language
(the same trap `content_filter/batch_test.py`'s paraphrase probes are named
for, applied here to extraction instead of retrieval).

None of this — negation, numeric thresholds, or the deliberate absence of a
constraint — ever appears in `synthetic_interactions/sessions.csv`'s templated
`query_text` (see `schema.weak_label_from_session`'s stated limitation). That
is what makes this the generalisation test, not a second in-distribution
sample.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "candidate_generation" / "filter_constraints"))
from constraints import (  # noqa: E402
    BudgetAtMost, MonthTempAtLeast, MonthTempAtMost, TagAtLeast, TagAtMost,
)

# -- negation: tags LABELLED_QUERIES never excludes on --
NEGATION = [
    ("cultural getaway, but nothing to do with adventure sports",
     (TagAtLeast("culture", 4), TagAtMost("adventure", 2))),
    ("relaxed trip, no wild outdoor adventures please",
     (TagAtMost("adventure", 2),)),
    ("city break, not interested in nature or hiking",
     (TagAtMost("nature", 2),)),
    ("food-focused trip, skip the wellness/spa stuff",
     (TagAtLeast("cuisine", 4), TagAtMost("wellness", 2))),
    ("lively destination, don't want anywhere secluded or remote",
     (TagAtMost("seclusion", 2),)),
    ("urban trip, but no all-night party scene",
     (TagAtLeast("urban", 4), TagAtMost("nightlife", 2))),
    ("somewhere with great food, definitely not a wellness retreat",
     (TagAtLeast("cuisine", 4), TagAtMost("wellness", 2))),
    ("beach trip that isn't a nightlife hotspot",
     (TagAtLeast("beaches", 4), TagAtMost("nightlife", 2))),
    ("outdoorsy trip, not a big city",
     (TagAtLeast("adventure", 4), TagAtMost("urban", 2))),
    ("historic city, no beach resort vibe",
     (TagAtLeast("culture", 4), TagAtMost("beaches", 2))),
]

# -- numeric: budget thresholds --
BUDGET = [
    ("shoestring trip, as cheap as possible",
     (BudgetAtMost("Budget"),)),
    ("keep it mid-range, nothing too expensive",
     (BudgetAtMost("Mid-range"),)),
    ("budget-conscious city break",
     (BudgetAtMost("Budget"),)),
    ("won't spend more than a mid-range budget",
     (BudgetAtMost("Mid-range"),)),
    ("backpacker budget only",
     (BudgetAtMost("Budget"),)),
]

# -- numeric: temperature thresholds, months LABELLED_QUERIES doesn't use --
TEMPERATURE = [
    ("somewhere mild in March, around 15 degrees or warmer",
     (MonthTempAtLeast(3, 15.0),)),
    ("looking for cool weather in September, below 18 degrees",
     (MonthTempAtMost(9, 18.0),)),
    ("hot destination in April, at least 28 degrees",
     (MonthTempAtLeast(4, 28.0),)),
    ("mild October trip, no hotter than 22 degrees",
     (MonthTempAtMost(10, 22.0),)),
    ("warm February getaway, above 24 degrees",
     (MonthTempAtLeast(2, 24.0),)),
    ("cool November escape, under 15 degrees",
     (MonthTempAtMost(11, 15.0),)),
]

# -- combined: multiple hard constraints at once --
COMBINED = [
    ("cheap cultural trip, no nightlife, warm in April above 25 degrees",
     (BudgetAtMost("Budget"), TagAtLeast("culture", 4), TagAtMost("nightlife", 2), MonthTempAtLeast(4, 25.0))),
    ("mid-range beach holiday, avoid crowded party towns",
     (BudgetAtMost("Mid-range"), TagAtLeast("beaches", 4), TagAtMost("nightlife", 2))),
    ("secluded wellness retreat, nothing urban, budget-friendly",
     (TagAtLeast("wellness", 4), TagAtLeast("seclusion", 4), TagAtMost("urban", 2), BudgetAtMost("Budget"))),
    ("foodie city trip, no beaches, mid-range budget",
     (TagAtLeast("cuisine", 4), TagAtMost("beaches", 2), BudgetAtMost("Mid-range"))),
    ("adventure trip in a cool climate, September under 20 degrees, no cities",
     (TagAtLeast("adventure", 4), MonthTempAtMost(9, 20.0), TagAtMost("urban", 2))),
    ("warm cultural escape in February above 20 degrees, avoid nightlife",
     (TagAtLeast("culture", 4), MonthTempAtLeast(2, 20.0), TagAtMost("nightlife", 2))),
    ("cheap nature trip, no adventure sports, cool in October below 18 degrees",
     (BudgetAtMost("Budget"), TagAtLeast("nature", 4), TagAtMost("adventure", 2), MonthTempAtMost(10, 18.0))),
    ("luxury beach vacation, quiet, no nightlife at all",
     (BudgetAtMost("Luxury"), TagAtLeast("beaches", 4), TagAtMost("nightlife", 2))),
]

# -- no hard constraint: soft/descriptive language only --
# The over-extraction trap: a model biased toward "always fill every field"
# will hallucinate a required_tag or budget_at_most here that the query never
# stated. Empty Intent (all fields empty/null) is the correct label.
NO_CONSTRAINT = [
    ("relaxing beach vacation somewhere nice", ()),
    ("chill seaside getaway", ()),
    ("somewhere with a vibrant nightlife scene", ()),
    ("off-the-beaten-path adventure destination", ()),
    ("a romantic city break", ()),
]

ADDITIONAL_OOD_QUERIES = NEGATION + BUDGET + TEMPERATURE + COMBINED + NO_CONSTRAINT
