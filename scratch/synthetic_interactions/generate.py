#!/usr/bin/env python3
"""Generate synthetic users and an interaction log (impression, click, save,
itinerary_add, not_interested) plus a per-session query/filter log, over the
content_filter city catalog. See README.md for the full design.
"""
import argparse
import csv
from collections import Counter
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

# Explicit "not interested": rare relative to silent non-clicks, and only likely for a
# genuine mismatch — this is what distinguishes "ignored" from "actively rejected."
NOT_INTERESTED_RATE = 0.08
NOT_INTERESTED_THRESHOLD = -1.0

# Dwell time on a clicked item's detail page — a graded engagement signal, not just
# binary click/no-click. Scales with how far affinity clears the click threshold.
DWELL_BASE_SECONDS = 20.0
DWELL_AFFINITY_LOG_SCALE = 0.35
DWELL_LOGNORMAL_SIGMA = 0.6
DWELL_MIN_SECONDS, DWELL_MAX_SECONDS = 3, 900

# Itinerary add: a stronger, later commitment than a save (cart -> purchase, not just wishlist).
ITINERARY_THRESHOLD_DELTA = 1.0
ITINERARY_DELAY_RANGE = (60, 3 * 24 * 3600)  # itinerary-building isn't instant

# Session-level query text + filters, templated from the user's (hidden) primary persona.
FILTER_BUDGET_RATE = 0.3
FILTER_TAG_RATE = 0.25
FILTER_TAG_MIN_RATING = 4

QUERY_TAG_PHRASES = {
    "culture": ["museums and historic architecture", "local culture and history", "art and heritage"],
    "adventure": ["outdoor adventure and hiking", "adrenaline and outdoor activities", "trekking"],
    "nature": ["scenic nature and landscapes", "national parks and wildlife", "unspoiled natural beauty"],
    "beaches": ["beaches and coastline", "sun, sand and sea", "beachfront relaxation"],
    "nightlife": ["nightlife and bars", "clubs and evening entertainment", "a vibrant nightlife scene"],
    "cuisine": ["great local food", "restaurants and street food", "a strong food and cuisine scene"],
    "wellness": ["spas and relaxation", "a wellness retreat", "a peaceful, restorative trip"],
    "urban": ["city life and urban exploring", "big-city energy", "urban sightseeing"],
    "seclusion": ["a quiet, secluded getaway", "off-the-beaten-path spots", "somewhere remote and uncrowded"],
}
QUERY_TEMPLATES_ONE_TAG = [
    "looking for a trip with {a}",
    "planning a trip around {a}",
    "want a destination known for {a}",
]
QUERY_TEMPLATES_TWO_TAG = [
    "somewhere good for {a} and {b}",
    "{a}, ideally with {b} too",
    "a trip that combines {a} and {b}",
]


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


def build_session_query(user: dict, rng: np.random.Generator) -> tuple[str, str, str]:
    """Template-derived query text + filters a session's user would plausibly type/apply,
    drawn from their (hidden) primary persona. Not fed back into affinity scoring — real
    query text wouldn't be either, it's a signal for filter_constraints/intent_parsing
    to consume, not for the generator's own ground truth."""
    persona_tags = PERSONAS[user["primary_persona"]]["tags"]
    n_tags = 2 if len(persona_tags) > 1 and rng.random() >= 0.4 else 1
    tags = list(rng.choice(persona_tags, size=min(n_tags, len(persona_tags)), replace=False))

    if len(tags) > 1:
        template = rng.choice(QUERY_TEMPLATES_TWO_TAG)
        query = template.format(a=rng.choice(QUERY_TAG_PHRASES[tags[0]]), b=rng.choice(QUERY_TAG_PHRASES[tags[1]]))
    else:
        template = rng.choice(QUERY_TEMPLATES_ONE_TAG)
        query = template.format(a=rng.choice(QUERY_TAG_PHRASES[tags[0]]))

    filter_budget = str(rng.choice(sorted(user["preferred_budgets"]))) if rng.random() < FILTER_BUDGET_RATE else ""
    filter_tag = str(tags[0]) if rng.random() < FILTER_TAG_RATE else ""
    return query, filter_budget, filter_tag


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
) -> tuple[list[tuple], list[tuple]]:
    n_items = len(item_ids)
    two_years_seconds = 2 * 365 * 24 * 3600
    base_time = np.datetime64("2024-01-01T00:00:00")
    item_budgets_arr = np.array(item_budgets)
    raw_tags = item_tags * 2.0 + 3.0  # undo load_catalog's 1-5 -> [-1,1] centering, for filtering
    itinerary_threshold = save_threshold + ITINERARY_THRESHOLD_DELTA

    rows = []
    session_rows = []
    interaction_id = 0
    for user in users:
        n_sessions = max(1, rng.poisson(sessions_per_user))
        session_offsets = np.sort(rng.integers(0, two_years_seconds, size=n_sessions))
        for s, offset in enumerate(session_offsets):
            session_id = f"{user['user_id']}-s{s:03d}"
            session_time = base_time + np.timedelta64(int(offset), "s")

            query_text, filter_budget, filter_tag = build_session_query(user, rng)
            session_rows.append((session_id, user["user_id"], session_time, query_text, filter_budget, filter_tag))

            mask = np.ones(n_items, dtype=bool)
            if filter_budget:
                mask &= item_budgets_arr == filter_budget
            if filter_tag:
                mask &= raw_tags[:, TAGS.index(filter_tag)] >= FILTER_TAG_MIN_RATING
            eligible = np.nonzero(mask)[0]

            n_impressions = int(rng.integers(impressions_range[0], impressions_range[1] + 1))
            if eligible.size < n_impressions:
                eligible = np.arange(n_items)  # filter too strict for this session, fall back
            pop_sub = popularity[eligible]
            pop_sub = pop_sub / pop_sub.sum()
            item_idx = rng.choice(eligible, size=n_impressions, replace=False, p=pop_sub)

            affinity = item_tags[item_idx] @ user["effective_weights"]
            budget_bonus = np.array([
                BUDGET_MATCH_BONUS if item_budgets[i] in user["preferred_budgets"] else 0.0
                for i in item_idx
            ])
            affinity = affinity + budget_bonus

            clicked = rng.random(n_impressions) < sigmoid(affinity - click_threshold)
            saved = clicked & (rng.random(n_impressions) < sigmoid(affinity - save_threshold))
            itinerary_added = saved & (rng.random(n_impressions) < sigmoid(affinity - itinerary_threshold))
            not_interested = ~clicked & (
                rng.random(n_impressions) < NOT_INTERESTED_RATE * sigmoid(NOT_INTERESTED_THRESHOLD - affinity)
            )

            dwell_mu_log = np.log(DWELL_BASE_SECONDS) + DWELL_AFFINITY_LOG_SCALE * np.clip(
                affinity - click_threshold, 0, None
            )
            dwell = np.clip(
                np.exp(rng.normal(dwell_mu_log, DWELL_LOGNORMAL_SIGMA)), DWELL_MIN_SECONDS, DWELL_MAX_SECONDS
            )

            for k, idx in enumerate(item_idx):
                item_id = item_ids[idx]
                rows.append((interaction_id, user["user_id"], item_id, "impression", session_id, session_time, ""))
                interaction_id += 1

                if clicked[k]:
                    t_click = session_time + np.timedelta64(int(rng.integers(1, 120)), "s")
                    rows.append((
                        interaction_id, user["user_id"], item_id, "click", session_id, t_click,
                        round(float(dwell[k]), 1),
                    ))
                    interaction_id += 1
                    if saved[k]:
                        t_save = t_click + np.timedelta64(int(rng.integers(1, 300)), "s")
                        rows.append((interaction_id, user["user_id"], item_id, "save", session_id, t_save, ""))
                        interaction_id += 1
                        if itinerary_added[k]:
                            t_itin = t_save + np.timedelta64(int(rng.integers(*ITINERARY_DELAY_RANGE)), "s")
                            rows.append((
                                interaction_id, user["user_id"], item_id, "itinerary_add", session_id, t_itin, "",
                            ))
                            interaction_id += 1
                elif not_interested[k]:
                    t_ni = session_time + np.timedelta64(int(rng.integers(1, 120)), "s")
                    rows.append((interaction_id, user["user_id"], item_id, "not_interested", session_id, t_ni, ""))
                    interaction_id += 1
    return rows, session_rows


def write_users_csv(path: Path, users: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "primary_persona", "secondary_persona", "mix_weight"])
        for u in users:
            w.writerow([u["user_id"], u["primary_persona"], u["secondary_persona"], u["mix_weight"]])


def write_interactions_csv(path: Path, rows: list[tuple]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["interaction_id", "user_id", "item_id", "event_type", "session_id", "timestamp", "dwell_seconds"])
        w.writerows(rows)


def write_sessions_csv(path: Path, session_rows: list[tuple]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["session_id", "user_id", "timestamp", "query_text", "filter_budget_level", "filter_required_tag"])
        w.writerows(session_rows)


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
    rows, session_rows = generate_interactions(
        users, item_ids, item_tags, item_budgets, popularity,
        args.sessions_per_user, (args.impressions_min, args.impressions_max),
        args.click_threshold, args.save_threshold, rng,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_users_csv(args.out_dir / "users.csv", users)
    write_interactions_csv(args.out_dir / "interactions.csv", rows)
    write_sessions_csv(args.out_dir / "sessions.csv", session_rows)

    counts = Counter(r[3] for r in rows)
    n_impressions, n_clicks, n_saves = counts["impression"], counts["click"], counts["save"]
    n_itinerary, n_not_interested = counts["itinerary_add"], counts["not_interested"]
    dwell_vals = [r[6] for r in rows if r[3] == "click"]
    filtered_sessions = sum(1 for r in session_rows if r[4] or r[5])

    print(f"users:       {len(users)}")
    print(f"sessions:    {len(session_rows)}  ({filtered_sessions} with a query filter applied)")
    print(f"interactions: {len(rows)} total")
    print(f"  impressions:    {n_impressions}")
    print(f"  clicks:         {n_clicks}  (CTR {n_clicks / n_impressions:.1%}, "
          f"mean dwell {sum(dwell_vals) / len(dwell_vals):.0f}s)")
    print(f"  saves:          {n_saves}  (save rate of clicks {n_saves / n_clicks:.1%})")
    print(f"  itinerary_add:  {n_itinerary}  ({n_itinerary / n_saves:.1%} of saves)")
    print(f"  not_interested: {n_not_interested}  ({n_not_interested / n_impressions:.1%} of impressions)")
    print(f"wrote {args.out_dir / 'users.csv'}")
    print(f"wrote {args.out_dir / 'interactions.csv'}")
    print(f"wrote {args.out_dir / 'sessions.csv'}")


if __name__ == "__main__":
    main()
