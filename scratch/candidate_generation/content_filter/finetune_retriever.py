#!/usr/bin/env python3
"""Contrastive fine-tune of the bi-encoder on synthetic_interactions clicks/saves,
instead of only ever using an off-the-shelf sentence-transformers checkpoint.

Positive pairs: (session's query_text, description of an item that session
clicked or saved). Trained with MultipleNegativesRankingLoss — the standard,
no-hard-negative-mining way to fine-tune a bi-encoder: every other positive's
item in the batch is used as an implicit negative for this one. STATED
SIMPLIFICATION: at 560 catalog items and a training batch of 16-32, the same
item landing twice in one batch (as two different sessions' positive) creates
a false negative — the loss briefly, incorrectly penalizes it. Left uncorrected
here (dataset dedup or a larger batch would reduce it) because it's a training
efficiency wrinkle, not a validity bug in the eval below.

Same in-distribution/out-of-distribution split `llm_wrapper/intent_parsing`
uses, and for the same reason: `sessions.csv`'s query_text is templated from
1-2 tags per query (see synthetic_interactions/generate.py's
build_session_query) and never contains negation or a numeric threshold —
`filter_constraints.LABELLED_QUERIES` does. Fine-tuning only ever sees the
ID shape; scoring on both tells you whether it generalized past that shape or
just overfit to the template.

Usage:
    python3 finetune_retriever.py                       # full run
    python3 finetune_retriever.py --max-pairs 500 --epochs 1   # quick smoke test
    python3 finetune_retriever.py --model all-MiniLM-L6-v2      # fast on CPU (default)
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "filter_constraints"))
from constraints import LABELLED_QUERIES, satisfies_all  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eval"))
from metrics import hit_rate_at_k, mrr_at_k  # noqa: E402

from content_filter import DATA_PATH, load_cities  # noqa: E402
from hybrid_search import dense_rank_all  # noqa: E402

SYNTHETIC_DIR = Path(__file__).resolve().parent.parent.parent / "synthetic_interactions" / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "finetuned_retriever"
CLICK_EVENTS = {"click", "save"}


def load_pairs(catalog_by_id: dict[str, dict]) -> list[dict]:
    """(session_id -> query_text) joined with (session_id -> clicked/saved item),
    keeping only sessions where the clicked/saved item is still in the catalog."""
    with open(SYNTHETIC_DIR / "sessions.csv", newline="", encoding="utf-8") as f:
        query_by_session = {row["session_id"]: row["query_text"] for row in csv.DictReader(f)}

    pairs = []
    with open(SYNTHETIC_DIR / "interactions.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["event_type"] not in CLICK_EVENTS:
                continue
            query = query_by_session.get(row["session_id"])
            item = catalog_by_id.get(row["item_id"])
            if query and item:
                pairs.append({"query": query, "item_id": row["item_id"], "description": item["short_description"]})
    return pairs


def train(model_name: str, pairs: list[dict], epochs: float, batch_size: int, lr: float, output_dir: Path):
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

    model = SentenceTransformer(model_name)
    loss = MultipleNegativesRankingLoss(model)
    dataset = Dataset.from_dict({
        "anchor": [p["query"] for p in pairs],
        "positive": [p["description"] for p in pairs],
    })
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=loss)
    trainer.train()
    model.save_pretrained(str(output_dir / "model"))
    return model


def id_eval(model, val_pairs: list[dict], k_values: list[int]) -> dict:
    """Held-out session queries: is the single known clicked/saved item ranked
    highly among all 560 candidates? Sparse ground truth (one known positive,
    everything else merely 'unknown', not confirmed negative) — same convention
    eval/'s per-user recsys metrics use."""
    cities = load_cities(DATA_PATH)
    descriptions = [c["short_description"] for c in cities]
    embeddings = model.encode(descriptions, normalize_embeddings=True)
    query_vecs = model.encode([p["query"] for p in val_pairs], normalize_embeddings=True)

    results = {k: {"hit_rate": [], "mrr": []} for k in k_values}
    for p, qvec in zip(val_pairs, query_vecs):
        ranking = dense_rank_all(qvec, cities, embeddings)
        relevant = {p["item_id"]: 1}
        for k in k_values:
            results[k]["hit_rate"].append(hit_rate_at_k(ranking, relevant, k))
            results[k]["mrr"].append(mrr_at_k(ranking, relevant, k))
    return {k: {m: float(np.mean(v)) for m, v in metrics.items()} for k, metrics in results.items()}


def ood_eval(model, k_values: list[int]) -> dict:
    """filter_constraints.LABELLED_QUERIES — negation/numeric patterns the
    training pairs above never contain."""
    cities = load_cities(DATA_PATH)
    descriptions = [c["short_description"] for c in cities]
    embeddings = model.encode(descriptions, normalize_embeddings=True)
    queries = [q for q, _ in LABELLED_QUERIES]
    query_vecs = model.encode(queries, normalize_embeddings=True)

    results = {k: {"hit_rate": [], "mrr": []} for k in k_values}
    for (_, predicates), qvec in zip(LABELLED_QUERIES, query_vecs):
        ranking = dense_rank_all(qvec, cities, embeddings)
        relevant = {c["id"]: 1 for c in cities if satisfies_all(c, predicates)}
        for k in k_values:
            results[k]["hit_rate"].append(hit_rate_at_k(ranking, relevant, k))
            results[k]["mrr"].append(mrr_at_k(ranking, relevant, k))
    return {k: {m: float(np.mean(v)) for m, v in metrics.items()} for k, metrics in results.items()}


def print_table(label: str, scores: dict) -> None:
    print(f"\n{label}")
    for k, metrics in scores.items():
        print(f"  k={k:<4}" + "  ".join(f"{m}={v:.3f}" for m, v in metrics.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="fast on CPU; swap for the bge model with a GPU")
    parser.add_argument("--max-pairs", type=int, default=None, help="subsample training pairs for a quick run")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cities = load_cities(DATA_PATH)
    catalog_by_id = {c["id"]: c for c in cities}
    pairs = load_pairs(catalog_by_id)

    rng = np.random.default_rng(args.seed)
    rng.shuffle(pairs)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    n_val = max(1, int(len(pairs) * args.val_fraction))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    print(f"{len(train_pairs)} train pairs, {len(val_pairs)} val pairs, "
          f"{len(LABELLED_QUERIES)} OOD queries")

    from sentence_transformers import SentenceTransformer
    base_model = SentenceTransformer(args.model)
    print("\n=== before fine-tuning ===")
    print_table("ID (held-out session queries)", id_eval(base_model, val_pairs, args.k_values))
    print_table("OOD (LABELLED_QUERIES)", ood_eval(base_model, args.k_values))

    tuned_model = train(args.model, train_pairs, args.epochs, args.batch_size, args.lr, OUTPUT_DIR)
    print("\n=== after fine-tuning ===")
    print_table("ID (held-out session queries)", id_eval(tuned_model, val_pairs, args.k_values))
    print_table("OOD (LABELLED_QUERIES)", ood_eval(tuned_model, args.k_values))
    print(f"\nfine-tuned model saved to {OUTPUT_DIR / 'model'}")


if __name__ == "__main__":
    main()
