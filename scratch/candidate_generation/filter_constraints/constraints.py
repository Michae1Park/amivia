"""Hard-constraint predicates over the city catalog, plus the hand-labelled probe set.

Deliberately boring: structured predicate logic, no ML. The point of this layer is
that `embed_retrieve` cannot express "no nightlife" or "under $50/day" at all —
those failures are representational, not a ranking wobble, so they get fixed
outside the retriever rather than by tuning it.

Labels live here, next to the queries, so the mapping from free text to predicate
stays auditable rather than buried in the experiment script.
"""
import json
from dataclasses import dataclass

TAGS = ["culture", "adventure", "nature", "beaches", "nightlife",
        "cuisine", "wellness", "urban", "seclusion"]

BUDGET_ORDER = {"Budget": 0, "Mid-range": 1, "Luxury": 2}

# STATED ASSUMPTION: the catalog has no price column — only this three-level enum.
# Mapping dollar amounts onto it is a modelling choice, not something the data
# supports, so it is written down here rather than implied by the code.
BUDGET_DOLLARS = {"Budget": "under ~$50/day", "Mid-range": "~$50-150/day", "Luxury": "over ~$150/day"}

MONTH_NAMES = {1: "January", 7: "July", 12: "December"}


# ---- predicates ------------------------------------------------------------


@dataclass(frozen=True)
class TagAtMost:
    """Negation: 'no nightlife' -> nightlife rated at most 2."""
    tag: str
    max_rating: int

    def satisfied(self, city: dict) -> bool:
        return int(city[self.tag]) <= self.max_rating

    def describe(self) -> str:
        return f"{self.tag} <= {self.max_rating}"


@dataclass(frozen=True)
class TagAtLeast:
    tag: str
    min_rating: int

    def satisfied(self, city: dict) -> bool:
        return int(city[self.tag]) >= self.min_rating

    def describe(self) -> str:
        return f"{self.tag} >= {self.min_rating}"


@dataclass(frozen=True)
class BudgetAtMost:
    level: str

    def satisfied(self, city: dict) -> bool:
        return BUDGET_ORDER[city["budget_level"]] <= BUDGET_ORDER[self.level]

    def describe(self) -> str:
        return f"budget <= {self.level} ({BUDGET_DOLLARS[self.level]})"


@dataclass(frozen=True)
class MonthTempAtLeast:
    month: int
    celsius: float

    def satisfied(self, city: dict) -> bool:
        return month_avg_temp(city, self.month) >= self.celsius

    def describe(self) -> str:
        return f"{MONTH_NAMES.get(self.month, self.month)} avg >= {self.celsius}C"


@dataclass(frozen=True)
class MonthTempAtMost:
    month: int
    celsius: float

    def satisfied(self, city: dict) -> bool:
        return month_avg_temp(city, self.month) <= self.celsius

    def describe(self) -> str:
        return f"{MONTH_NAMES.get(self.month, self.month)} avg <= {self.celsius}C"


def month_avg_temp(city: dict, month: int) -> float:
    return float(json.loads(city["avg_temp_monthly"])[str(month)]["avg"])


def violations(city: dict, predicates: tuple) -> int:
    return sum(1 for p in predicates if not p.satisfied(city))


def satisfies_all(city: dict, predicates: tuple) -> bool:
    return violations(city, predicates) == 0


# ---- the labelled probe set ------------------------------------------------

# Each query carries the constraints a human reads out of it. These extend
# batch_test.py's negation/numeric probes: four queries is far too few for a
# violation rate to mean anything, so the set is widened to 15 across the same
# three failure modes plus combinations.
LABELLED_QUERIES = [
    # -- negation: the failure embeddings cannot express --
    ("quiet town, definitely no nightlife",
     (TagAtMost("nightlife", 2),)),
    ("beach getaway with no crowds and no nightlife",
     (TagAtLeast("beaches", 4), TagAtMost("nightlife", 2))),
    ("cultural city trip, not a beach destination",
     (TagAtLeast("culture", 4), TagAtMost("beaches", 2))),
    ("nature escape, nothing urban about it",
     (TagAtLeast("nature", 4), TagAtMost("urban", 2))),
    ("somewhere secluded, definitely not a party town",
     (TagAtLeast("seclusion", 4), TagAtMost("nightlife", 2))),

    # -- numeric: budget thresholds --
    ("under $50 a day",
     (BudgetAtMost("Budget"),)),
    ("cheap backpacking trip, under $50 per day",
     (BudgetAtMost("Budget"),)),
    ("affordable city break, mid-range at most",
     (BudgetAtMost("Mid-range"),)),

    # -- numeric: temperature thresholds --
    ("average July temperature above 30 degrees Celsius",
     (MonthTempAtLeast(7, 30.0),)),
    ("somewhere warm in December, at least 25 degrees",
     (MonthTempAtLeast(12, 25.0),)),
    ("cool summer escape, July below 20 degrees",
     (MonthTempAtMost(7, 20.0),)),

    # -- combined: several hard constraints at once --
    ("budget beach trip under $50 a day with warm July weather",
     (BudgetAtMost("Budget"), TagAtLeast("beaches", 4), MonthTempAtLeast(7, 25.0))),
    ("quiet wellness retreat, no nightlife",
     (TagAtLeast("wellness", 4), TagAtMost("nightlife", 2))),
    ("adventure travel on a budget, no beaches",
     (TagAtLeast("adventure", 4), BudgetAtMost("Budget"), TagAtMost("beaches", 2))),
    ("warm cultural city in January, above 20 degrees",
     (TagAtLeast("culture", 4), MonthTempAtLeast(1, 20.0))),
]

# Queries whose negation has no honest predicate in this schema. Kept visible
# rather than quietly dropped: "touristy" is not a column, and inventing a proxy
# for it would be fitting the label to the data. A real system would need either
# a new attribute or a soft signal here — which is exactly the boundary of what
# a deterministic filter layer can do.
UNMAPPABLE_QUERIES = [
    ("authentic place, avoid touristy spots", "no 'touristy' attribute in the catalog schema"),
]
