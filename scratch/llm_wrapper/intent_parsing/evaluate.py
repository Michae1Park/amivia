"""Shared scorer for both arms' prediction files (prompt_arm.py / finetune_arm.py
write the same {"query", "gold_intent", "predicted_intent", ...} JSONL shape).

Reports per-field, not one blended accuracy number — per the original design
note, dates/budget/tag failures would otherwise hide inside an average. The
ID/OOD split is itself the headline result: a gap between the two is the
generalisation question this whole project is testing, so it is never
collapsed into a single "overall" row.

Usage:
    python3 evaluate.py data/predictions_prompt_claude-sonnet-5_described_ood_eval.jsonl
    python3 evaluate.py data/predictions_finetune_Qwen2.5-1.5B-Instruct_ood_eval.jsonl --label "finetune (OOD)"
"""
import argparse
import json
import sys
from pathlib import Path

from schema import TAGS, Intent, is_schema_valid


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tag_set(entries: list[dict]) -> set:
    return {e["tag"] for e in entries if isinstance(e, dict) and "tag" in e}


def set_prf1(gold: set, pred: set) -> tuple[float, float, float]:
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    if not pred:
        return 0.0, 0.0, 0.0
    if not gold:
        return 0.0, 0.0, 0.0
    tp = len(gold & pred)
    precision = tp / len(pred)
    recall = tp / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def score(predictions: list[dict]) -> dict:
    n = len(predictions)
    schema_valid = 0
    required_f1s, excluded_f1s = [], []
    budget_correct = 0
    temp_at_least_correct = 0
    temp_at_most_correct = 0
    exact_match = 0
    latencies = []

    for p in predictions:
        gold = Intent.from_dict(p["gold_intent"])
        pred_raw = p.get("predicted_intent") or {}
        valid = is_schema_valid(pred_raw)
        schema_valid += valid
        pred = Intent.from_dict(pred_raw) if valid else Intent()

        _, _, f1 = set_prf1(tag_set(gold.required_tags), tag_set(pred.required_tags))
        required_f1s.append(f1)
        _, _, f1 = set_prf1(tag_set(gold.excluded_tags), tag_set(pred.excluded_tags))
        excluded_f1s.append(f1)

        budget_correct += gold.budget_at_most == pred.budget_at_most
        temp_at_least_correct += gold.month_temp_at_least == pred.month_temp_at_least
        temp_at_most_correct += gold.month_temp_at_most == pred.month_temp_at_most
        exact_match += gold.to_dict() == pred.to_dict()

        if "latency_s" in p:
            latencies.append(p["latency_s"])

    return {
        "n": n,
        "schema_valid_rate": schema_valid / n,
        "required_tags_f1": sum(required_f1s) / n,
        "excluded_tags_f1": sum(excluded_f1s) / n,
        "budget_at_most_acc": budget_correct / n,
        "month_temp_at_least_acc": temp_at_least_correct / n,
        "month_temp_at_most_acc": temp_at_most_correct / n,
        "exact_match_rate": exact_match / n,
        "mean_latency_s": sum(latencies) / len(latencies) if latencies else None,
    }


def print_report(label: str, metrics: dict) -> None:
    print(f"\n== {label} (n={metrics['n']}) ==")
    for key in ("schema_valid_rate", "required_tags_f1", "excluded_tags_f1",
                "budget_at_most_acc", "month_temp_at_least_acc", "month_temp_at_most_acc",
                "exact_match_rate"):
        print(f"  {key:<28} {metrics[key]:.3f}")
    if metrics["mean_latency_s"] is not None:
        print(f"  {'mean_latency_s':<28} {metrics['mean_latency_s']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prediction_files", nargs="+", type=Path)
    parser.add_argument("--label", action="append", help="one per prediction file, in order")
    args = parser.parse_args()

    labels = args.label or [f.stem for f in args.prediction_files]
    if len(labels) != len(args.prediction_files):
        sys.exit("--label must be passed once per prediction file, or not at all")

    for path, label in zip(args.prediction_files, labels):
        predictions = load_jsonl(path)
        print_report(label, score(predictions))


if __name__ == "__main__":
    main()
