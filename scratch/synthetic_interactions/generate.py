#!/usr/bin/env python3
"""Generate synthetic users and an impression/click/save interaction log
over the embed_retrieve city catalog. See README.md for the full design.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

CATALOG_PATH = (
    Path(__file__).parent.parent
    / "candidate_generation" / "data"
    / "Worldwide Travel Cities Dataset (Ratings and Climate).csv"
)
DATA_DIR = Path(__file__).parent / "data"

TAGS = ["culture", "adventure", "nature", "beaches", "nightlife",
        "cuisine", "wellness", "urban", "seclusion"]

# tags: what this persona is weighted toward. budgets: acceptable budget_level values.
PERSONAS = {
    "Budget Backpacker":  {"tags": ["adventure", "culture"], "budgets": ["Budget"]},
    "Luxury Relaxer":     {"tags": ["wellness", "beaches", "cuisine"], "budgets": ["Luxury"]},
    "Nightlife Seeker":   {"tags": ["nightlife", "urban"], "budgets": ["Mid-range"]},
    "Culture Buff":       {"tags": ["culture", "cuisine", "urban"], "budgets": ["Mid-range"]},
    "Nature & Adventure": {"tags": ["nature", "adventure", "seclusion"], "budgets": ["Budget", "Mid-range"]},
    "Family Beach":       {"tags": ["beaches", "wellness"], "budgets": ["Mid-range"]},
}
PERSONA_NAMES = list(PERSONAS.keys())

EMPHASIS_WEIGHT = 1.0
BASELINE_WEIGHT = -0.2  # mild negative so mismatched tags actively hurt affinity, not just fail to help
BUDGET_MATCH_BONUS = 1.5


def build_persona_weight_vectors() -> dict[str, np.ndarray]:
    vectors = {}
    for name, spec in PERSONAS.items():
        w = np.full(len(TAGS), BASELINE_WEIGHT)
        for tag in spec["tags"]:
            w[TAGS.index(tag)] = EMPHASIS_WEIGHT
        vectors[name] = w / np.linalg.norm(w)
    return vectors


def load_catalog(path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    ids, tag_rows, budgets = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.append(row["id"])
            tag_rows.append([int(row[t]) for t in TAGS])
            budgets.append(row["budget_level"])
    tag_matrix = np.array(tag_rows, dtype=float)
    centered = (tag_matrix - 3.0) / 2.0  # 1-5 -> [-1, 1] so mismatch can pull affinity negative
    return ids, centered, budgets


def item_popularity(n_items: int, skew: float, rng: np.random.Generator) -> np.ndarray:
    """Power-law popularity, decoupled from persona fit — some cities are just more shown."""
    ranks = rng.permutation(n_items) + 1
    weights = 1.0 / (ranks ** skew)
    return weights / weights.sum()


def generate_users(n_users: int, persona_mix_purity: float, rng: np.random.Generator) -> list[dict]:
    persona_vecs = build_persona_weight_vectors()
    users = []
    for i in range(n_users):
        primary = rng.choice(PERSONA_NAMES)
        secondary = rng.choice([p for p in PERSONA_NAMES if p != primary])
        purity = float(np.clip(rng.normal(persona_mix_purity, 0.07), 0.5, 1.0))
        noise = rng.normal(0, 0.15, size=len(TAGS))
        effective = purity * persona_vecs[primary] + (1 - purity) * persona_vecs[secondary] + noise
        users.append({
            "user_id": f"u{i:06d}",
            "primary_persona": primary,
            "secondary_persona": secondary,
            "mix_weight": round(purity, 4),
            "effective_weights": effective,
            "preferred_budgets": set(PERSONAS[primary]["budgets"]),
        })
    return users


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_interactions(
    users: list[dict],
    item_ids: list[str],
    item_tags: np.ndarray,
    item_budgets: list[str],
    popularity: np.ndarray,
    sessions_per_user: float,
    impressions_range: tuple[int, int],
    click_threshold: float,
    save_threshold: float,
    rng: np.random.Generator,
) -> list[tuple]:
    n_items = len(item_ids)
    two_years_seconds = 2 * 365 * 24 * 3600
    base_time = np.datetime64("2024-01-01T00:00:00")

    rows = []
    interaction_id = 0
    for user in users:
        n_sessions = max(1, rng.poisson(sessions_per_user))
        session_offsets = np.sort(rng.integers(0, two_years_seconds, size=n_sessions))
        for s, offset in enumerate(session_offsets):
            session_id = f"{user['user_id']}-s{s:03d}"
            session_time = base_time + np.timedelta64(int(offset), "s")
            n_impressions = int(rng.integers(impressions_range[0], impressions_range[1] + 1))
            item_idx = rng.choice(n_items, size=n_impressions, replace=False, p=popularity)

            affinity = item_tags[item_idx] @ user["effective_weights"]
            budget_bonus = np.array([
                BUDGET_MATCH_BONUS if item_budgets[i] in user["preferred_budgets"] else 0.0
                for i in item_idx
            ])
            affinity = affinity + budget_bonus

            clicked = rng.random(n_impressions) < sigmoid(affinity - click_threshold)
            saved = clicked & (rng.random(n_impressions) < sigmoid(affinity - save_threshold))

            for k, idx in enumerate(item_idx):
                rows.append((interaction_id, user["user_id"], item_ids[idx], "impression", session_id, session_time))
                interaction_id += 1
                if clicked[k]:
                    t_click = session_time + np.timedelta64(int(rng.integers(1, 120)), "s")
                    rows.append((interaction_id, user["user_id"], item_ids[idx], "click", session_id, t_click))
                    interaction_id += 1
                    if saved[k]:
                        t_save = t_click + np.timedelta64(int(rng.integers(1, 300)), "s")
                        rows.append((interaction_id, user["user_id"], item_ids[idx], "save", session_id, t_save))
                        interaction_id += 1
    return rows


def write_users_csv(path: Path, users: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "primary_persona", "secondary_persona", "mix_weight"])
        for u in users:
            w.writerow([u["user_id"], u["primary_persona"], u["secondary_persona"], u["mix_weight"]])


def write_interactions_csv(path: Path, rows: list[tuple]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["interaction_id", "user_id", "item_id", "event_type", "session_id", "timestamp"])
        for interaction_id, user_id, item_id, event_type, session_id, timestamp in rows:
            w.writerow([interaction_id, user_id, item_id, event_type, session_id, timestamp])


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic user-item interaction data")
    parser.add_argument("--n-users", type=int, default=5000)
    parser.add_argument("--sessions-per-user", type=float, default=6.0, help="Poisson mean")
    parser.add_argument("--impressions-min", type=int, default=10)
    parser.add_argument("--impressions-max", type=int, default=20)
    parser.add_argument("--persona-mix-purity", type=float, default=0.8)
    parser.add_argument("--popularity-skew", type=float, default=1.0)
    parser.add_argument("--click-threshold", type=float, default=2.8)
    parser.add_argument("--save-threshold", type=float, default=3.8)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    rng = np.random.default_rng(args.random_seed)

    item_ids, item_tags, item_budgets = load_catalog(args.catalog)
    popularity = item_popularity(len(item_ids), args.popularity_skew, rng)
    users = generate_users(args.n_users, args.persona_mix_purity, rng)
    rows = generate_interactions(
        users, item_ids, item_tags, item_budgets, popularity,
        args.sessions_per_user, (args.impressions_min, args.impressions_max),
        args.click_threshold, args.save_threshold, rng,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_users_csv(args.out_dir / "users.csv", users)
    write_interactions_csv(args.out_dir / "interactions.csv", rows)

    n_impressions = sum(1 for r in rows if r[3] == "impression")
    n_clicks = sum(1 for r in rows if r[3] == "click")
    n_saves = sum(1 for r in rows if r[3] == "save")
    print(f"users:       {len(users)}")
    print(f"interactions: {len(rows)} total")
    print(f"  impressions: {n_impressions}")
    print(f"  clicks:      {n_clicks}  (CTR {n_clicks / n_impressions:.1%})")
    print(f"  saves:       {n_saves}  (save rate of clicks {n_saves / n_clicks:.1%})")
    print(f"wrote {args.out_dir / 'users.csv'}")
    print(f"wrote {args.out_dir / 'interactions.csv'}")


if __name__ == "__main__":
    main()
