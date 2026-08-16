import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "synthetic_interactions"))
from generate import load_catalog, build_persona_weight_vectors, PERSONAS, CATALOG_PATH  # noqa: E402

SYN_DATA_DIR = Path(__file__).resolve().parent.parent / "synthetic_interactions" / "data"
USERS_PATH = SYN_DATA_DIR / "users.csv"
INTERACTIONS_PATH = SYN_DATA_DIR / "interactions.csv"

GRADE = {"save": 2, "click": 1}  # impressions carry no positive signal


def load_users(path: Path = USERS_PATH) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_interactions(path: Path = INTERACTIONS_PATH) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_train_test_split(interactions: list[dict], test_fraction: float = 0.2):
    """Per-user temporal holdout: each user's most recent sessions become the test set."""
    by_user = defaultdict(list)
    for row in interactions:
        by_user[row["user_id"]].append(row)

    train, test = [], []
    for rows in by_user.values():
        rows.sort(key=lambda r: r["timestamp"])
        sessions = list(dict.fromkeys(r["session_id"] for r in rows))  # de-dup, preserves order
        n_test_sessions = max(1, round(len(sessions) * test_fraction))
        test_sessions = set(sessions[-n_test_sessions:])
        for r in rows:
            (test if r["session_id"] in test_sessions else train).append(r)
    return train, test


def relevance_and_seen(rows: list[dict]):
    """rows -> (user_id -> {item_id: grade}, user_id -> {item_id seen}). Grade = max(save=2, click=1)."""
    relevance = defaultdict(dict)
    seen = defaultdict(set)
    for r in rows:
        seen[r["user_id"]].add(r["item_id"])
        grade = GRADE.get(r["event_type"])
        if grade is not None:
            cur = relevance[r["user_id"]].get(r["item_id"], 0)
            relevance[r["user_id"]][r["item_id"]] = max(cur, grade)
    return relevance, seen
