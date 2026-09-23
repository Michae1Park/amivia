"""Arm B: LoRA fine-tune of a small open model on the same JSON-extraction task
Arm A prompts a hosted model for.

Needs a GPU (or a slow CPU/MPS run) and packages not in the project venv:
    pip install torch transformers peft accelerate

Trains on data/train.jsonl only — which, per schema.weak_label_from_session's
stated limitation, contains zero labelled negation/temperature examples. That
gap is deliberate: it's what makes data/ood_eval.jsonl a real test of whether
fine-tuning on cheap templated data generalises past the template, rather than
a second in-distribution sample dressed up as a harder one.

Loss is computed on the assistant completion only — the prompt tokens (system
schema + user query) are masked to -100, standard SFT practice. Skipping that
mask would let the model get credit for predicting tokens it was simply shown,
inflating the loss curve without teaching it anything about the task.

Usage:
    python3 finetune_arm.py                                   # full run, default settings
    python3 finetune_arm.py --max-train 500 --epochs 1         # quick smoke test
    python3 finetune_arm.py --model Qwen/Qwen2.5-0.5B-Instruct  # smaller/faster base
"""
import argparse
import json
import time
from pathlib import Path

from schema import is_schema_valid

DATA_DIR = Path(__file__).resolve().parent / "data"

SYSTEM_PROMPT = (
    "Extract a travel query's constraints into the given JSON schema. "
    "Return only fields the query actually states; leave everything else empty/null.\n\n"
    "Schema fields: required_tags (list of {tag, min_rating}), excluded_tags (list of "
    "{tag, max_rating} — this is how negation is expressed), budget_at_most "
    "(Budget/Mid-range/Luxury/null), month_temp_at_least ({month, celsius}/null), "
    "month_temp_at_most ({month, celsius}/null).\n"
    "Respond with the JSON object only, no other text."
)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_messages(query: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]


def tokenize_example(tokenizer, query: str, intent: dict, max_length: int):
    """Full (prompt + completion) input_ids, with labels masked to -100 over the
    prompt span so loss is only ever computed on the JSON completion."""
    messages = build_messages(query)
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    completion = json.dumps(intent, separators=(",", ":")) + tokenizer.eos_token
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

    input_ids = (prompt_ids + completion_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + completion_ids)[:max_length]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


class IntentSFTDataset:
    def __init__(self, examples: list[dict], tokenizer, max_length: int = 512):
        self.rows = [tokenize_example(tokenizer, ex["query"], ex["intent"], max_length) for ex in examples]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def make_collator(tokenizer):
    import torch

    def collate(batch):
        max_len = max(len(r["input_ids"]) for r in batch)
        pad_id = tokenizer.pad_token_id
        input_ids, attention_mask, labels = [], [], []
        for r in batch:
            pad = max_len - len(r["input_ids"])
            input_ids.append(r["input_ids"] + [pad_id] * pad)
            attention_mask.append(r["attention_mask"] + [0] * pad)
            labels.append(r["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }
    return collate


def train(model_name: str, output_dir: Path, train_examples: list[dict],
          epochs: float, lr: float, lora_r: int, lora_alpha: int):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = IntentSFTDataset(train_examples, tokenizer)
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        bf16=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=make_collator(tokenizer))
    trainer.train()

    model.save_pretrained(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))
    return model, tokenizer


def generate_predictions(model, tokenizer, examples: list[dict]) -> list[dict]:
    import torch

    model.eval()
    predictions = []
    for ex in examples:
        messages = build_messages(ex["query"])
        prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        prompt_ids = prompt_ids.to(model.device)

        start = time.monotonic()
        with torch.no_grad():
            output_ids = model.generate(
                prompt_ids, max_new_tokens=256, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        latency = time.monotonic() - start

        raw_text = tokenizer.decode(output_ids[0][prompt_ids.shape[1]:], skip_special_tokens=True)
        try:
            # Models occasionally wrap the JSON in prose despite the instruction; take
            # the first {...} block rather than failing the whole parse on that.
            start_brace, end_brace = raw_text.index("{"), raw_text.rindex("}") + 1
            intent = json.loads(raw_text[start_brace:end_brace])
        except (ValueError, json.JSONDecodeError):
            intent = {}

        predictions.append({
            "query": ex["query"],
            "gold_intent": ex["intent"],
            "predicted_intent": intent,
            "raw_text": raw_text,
            "latency_s": latency,
        })
    return predictions


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "finetune_run")
    parser.add_argument("--max-train", type=int, default=None, help="subsample train.jsonl for a quick run")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    train_examples = load_jsonl(DATA_DIR / "train.jsonl")
    if args.max_train:
        train_examples = train_examples[: args.max_train]

    model, tokenizer = train(args.model, args.output_dir, train_examples, args.epochs, args.lr,
                              args.lora_r, args.lora_alpha)

    for split in ("val", "ood_eval"):
        examples = load_jsonl(DATA_DIR / f"{split}.jsonl")
        predictions = generate_predictions(model, tokenizer, examples)
        n_valid = sum(is_schema_valid(p["predicted_intent"]) for p in predictions)
        out_path = DATA_DIR / f"predictions_finetune_{args.model.split('/')[-1]}_{split}.jsonl"
        write_jsonl(out_path, predictions)
        print(f"{split}: {len(predictions)} predictions, {n_valid} schema-valid -> {out_path}")


if __name__ == "__main__":
    main()
