"""Structured intent schema for the conversation layer, and the two ways this
project derives labels for it:

- weak, at-scale, from `synthetic_interactions/data/sessions.csv`'s templated
  `query_text` + `filter_budget_level` + `filter_required_tag` columns
- hand-labelled, small, from `filter_constraints`'s `LABELLED_QUERIES` (negation,
  numeric thresholds, combinations) plus this project's own additions in
  `ood_queries.py` — the patterns the templated session log never produces

Deliberately mirrors `filter_constraints.constraints`'s predicate classes
field-for-field: `to_predicates()` turns a parsed intent directly into the
tuple `filter_experiment.py` already knows how to consume, so this project's
output is literally that layer's input, not a standalone JSON blob that
happens to look similar.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "candidate_generation" / "filter_constraints"))
import constraints as fc  # noqa: E402

TAGS = fc.TAGS
BUDGET_LEVELS = tuple(fc.BUDGET_ORDER)

# Matches synthetic_interactions/generate.py's FILTER_TAG_MIN_RATING — the
# rating threshold a session's filter_required_tag was actually generated
# under. Weak labels reuse the same number so they stay consistent with how
# the data was made, not an independently guessed threshold.
SESSION_FILTER_TAG_MIN_RATING = 4


@dataclass
class Intent:
    required_tags: list[dict] = field(default_factory=list)   # [{"tag": str, "min_rating": int}]
    excluded_tags: list[dict] = field(default_factory=list)   # [{"tag": str, "max_rating": int}]
    budget_at_most: str | None = None
    month_temp_at_least: dict | None = None                    # {"month": int, "celsius": float}
    month_temp_at_most: dict | None = None

    def to_dict(self) -> dict:
        return {
            "required_tags": self.required_tags,
            "excluded_tags": self.excluded_tags,
            "budget_at_most": self.budget_at_most,
            "month_temp_at_least": self.month_temp_at_least,
            "month_temp_at_most": self.month_temp_at_most,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Intent":
        return cls(
            required_tags=[{"tag": t["tag"], "min_rating": int(t["min_rating"])} for t in (d.get("required_tags") or [])],
            excluded_tags=[{"tag": t["tag"], "max_rating": int(t["max_rating"])} for t in (d.get("excluded_tags") or [])],
            budget_at_most=d.get("budget_at_most"),
            month_temp_at_least=d.get("month_temp_at_least"),
            month_temp_at_most=d.get("month_temp_at_most"),
        )

    def to_predicates(self) -> tuple:
        """The whole point of the schema: hand straight to
        filter_constraints/filter_experiment.py, no translation layer needed."""
        preds = []
        for t in self.required_tags:
            preds.append(fc.TagAtLeast(t["tag"], int(t["min_rating"])))
        for t in self.excluded_tags:
            preds.append(fc.TagAtMost(t["tag"], int(t["max_rating"])))
        if self.budget_at_most:
            preds.append(fc.BudgetAtMost(self.budget_at_most))
        if self.month_temp_at_least:
            preds.append(fc.MonthTempAtLeast(int(self.month_temp_at_least["month"]), float(self.month_temp_at_least["celsius"])))
        if self.month_temp_at_most:
            preds.append(fc.MonthTempAtMost(int(self.month_temp_at_most["month"]), float(self.month_temp_at_most["celsius"])))
        return tuple(preds)

    @classmethod
    def from_predicates(cls, predicates: tuple) -> "Intent":
        """Inverse of to_predicates — lets the OOD eval set be derived straight
        from filter_constraints.LABELLED_QUERIES instead of re-typing 15 labels
        by hand and risking the two copies drifting apart."""
        intent = cls()
        for p in predicates:
            if isinstance(p, fc.TagAtLeast):
                intent.required_tags.append({"tag": p.tag, "min_rating": p.min_rating})
            elif isinstance(p, fc.TagAtMost):
                intent.excluded_tags.append({"tag": p.tag, "max_rating": p.max_rating})
            elif isinstance(p, fc.BudgetAtMost):
                intent.budget_at_most = p.level
            elif isinstance(p, fc.MonthTempAtLeast):
                intent.month_temp_at_least = {"month": p.month, "celsius": p.celsius}
            elif isinstance(p, fc.MonthTempAtMost):
                intent.month_temp_at_most = {"month": p.month, "celsius": p.celsius}
            else:
                raise ValueError(f"unhandled predicate type: {type(p)}")
        return intent


def weak_label_from_session(filter_budget_level: str, filter_required_tag: str) -> Intent:
    """`sessions.csv` row -> Intent, for the in-distribution (ID) train/val split.

    STATED ASSUMPTION: the session log's `filter_budget_level` is an EXACT match
    (generate.py: `item_budgets_arr == filter_budget`), not an upper bound. This
    schema only has `budget_at_most` (matching filter_constraints), so an exact
    "Mid-range" filter gets mapped onto `budget_at_most="Mid-range"` — a looser
    constraint than what actually generated the session. That is a modelling
    choice the data does not itself support, written down here rather than left
    implicit, the same way filter_constraints documents its own budget-dollar
    mapping.

    STATED LIMITATION: the session log never encodes negation or a temperature
    threshold — `build_session_query` in generate.py only ever emits a required
    tag and/or a budget filter. A model trained only on this function's output
    has literally never seen a labelled excluded_tags or month_temp_* example
    during training. That gap is intentional, not an oversight — it's what
    makes the OOD hand-labelled set (ood_queries.py) a real generalisation
    test rather than a second sample from the same distribution.
    """
    intent = Intent()
    if filter_required_tag:
        intent.required_tags.append({"tag": filter_required_tag, "min_rating": SESSION_FILTER_TAG_MIN_RATING})
    if filter_budget_level:
        intent.budget_at_most = filter_budget_level
    return intent


# ---- JSON Schema for structured-output / tool-calling prompting -----------

INTENT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "required_tags": {
            "type": "array",
            "description": "Tags the destination MUST have, at or above min_rating (1-5).",
            "items": {
                "type": "object",
                "properties": {
                    "tag": {"type": "string", "enum": list(TAGS)},
                    "min_rating": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["tag", "min_rating"],
                "additionalProperties": False,
            },
        },
        "excluded_tags": {
            "type": "array",
            "description": "Tags the destination must NOT have above max_rating. This is how "
                            "negation ('no nightlife', 'not a beach destination') is expressed.",
            "items": {
                "type": "object",
                "properties": {
                    "tag": {"type": "string", "enum": list(TAGS)},
                    "max_rating": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["tag", "max_rating"],
                "additionalProperties": False,
            },
        },
        "budget_at_most": {
            "type": ["string", "null"],
            "enum": list(BUDGET_LEVELS) + [None],
            "description": "Upper bound on budget level, or null if the query states no budget constraint.",
        },
        "month_temp_at_least": {
            "type": ["object", "null"],
            "description": "A stated lower-bound temperature constraint for one month, or null.",
            "properties": {
                "month": {"type": "integer", "minimum": 1, "maximum": 12},
                "celsius": {"type": "number"},
            },
        },
        "month_temp_at_most": {
            "type": ["object", "null"],
            "description": "A stated upper-bound temperature constraint for one month, or null.",
            "properties": {
                "month": {"type": "integer", "minimum": 1, "maximum": 12},
                "celsius": {"type": "number"},
            },
        },
    },
    "required": ["required_tags", "excluded_tags", "budget_at_most", "month_temp_at_least", "month_temp_at_most"],
    "additionalProperties": False,
}


def is_schema_valid(d: dict) -> bool:
    """Cheap structural check, not full JSON-Schema validation — answers 'did the
    model produce something well-formed at all,' the floor metric before any
    field is scored for correctness."""
    try:
        intent = Intent.from_dict(d)
        for t in intent.required_tags:
            assert t["tag"] in TAGS and 1 <= int(t["min_rating"]) <= 5
        for t in intent.excluded_tags:
            assert t["tag"] in TAGS and 1 <= int(t["max_rating"]) <= 5
        if intent.budget_at_most is not None:
            assert intent.budget_at_most in BUDGET_LEVELS
        for temp in (intent.month_temp_at_least, intent.month_temp_at_most):
            if temp is not None:
                assert 1 <= int(temp["month"]) <= 12
                float(temp["celsius"])
        return True
    except (KeyError, ValueError, AssertionError, TypeError):
        return False
