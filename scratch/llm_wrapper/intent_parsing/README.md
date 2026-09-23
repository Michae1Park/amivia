# Project 10: Intent parsing (optional, LLM)

Layer: conversation layer / user understanding — see `docs/architecture.md`
§1 and §2. First of the optional LLM-wrapper projects; the backbone
(Projects 1-9) works without this.

**Problem:** turn a free-text query ("cheap beach trip in March, no crowds")
into the structured session intent the backbone actually consumes (budget,
tag constraints) — the same slot-filling job a classic search UI's
form/filters do, done by an LLM instead.

**Scope change from the original sketch:** this project now compares two
different ways to do that extraction, not just one:

- **Arm A — prompting.** A hosted LLM (Claude), given the schema in the
  system prompt, no training.
- **Arm B — fine-tuning.** A small open model (Qwen2.5-0.5B/1.5B-Instruct),
  LoRA-tuned on labelled (query, intent) pairs via HF PEFT.

Reason for the change: prompting alone only demonstrates "structured JSON
output" and "prompting strategy." It says nothing about fine-tuning or about
training a model at all. Arm B closes that gap without inventing an
unrelated side project — it's the same task, the same schema, the same eval,
scored against the same OOD set, so the two arms produce one comparison
instead of two disconnected demos.

**Schema:** deliberately mirrors `filter_constraints.constraints`'s predicate
classes field-for-field (`schema.py`) — `required_tags` -> `TagAtLeast`,
`excluded_tags` -> `TagAtMost`, `budget_at_most` -> `BudgetAtMost`,
`month_temp_at_least`/`month_temp_at_most` -> their same-named predicates.
`Intent.to_predicates()` converts straight into the tuple
`filter_experiment.py` already knows how to consume, so this project's
output is layer 3's input, not a standalone JSON blob that happens to look
similar. `schema.INTENT_JSON_SCHEMA` is the same schema as a JSON-Schema dict,
for structured-output prompting.

**Data — two sources, deliberately different in kind:**

- **In-distribution (ID), free, at scale:** `synthetic_interactions/data/sessions.csv`'s
  29,944 rows already carry `query_text` (templated free text) plus
  `filter_budget_level` / `filter_required_tag` (what was actually applied) —
  see that project's README. `schema.weak_label_from_session` turns each row
  into a weak intent label with zero hand-labelling. `build_dataset.py`
  samples 4,000 train / 500 val from it.
- **Out-of-distribution (OOD), hand-labelled, small:**
  `filter_constraints.LABELLED_QUERIES` (15 queries) plus this project's own
  `ood_queries.py` (34 more) — 49 total, covering negation, numeric budget,
  numeric temperature, combinations, and queries with **no** hard constraint
  at all (testing over-extraction, the "invents a constraint from vague
  language" failure mode).

**Why two sources instead of one bigger hand-labelled set:** `sessions.csv`'s
generator (`build_session_query` in `synthetic_interactions/generate.py`)
only ever emits a required tag and/or a budget filter — never negation, never
a temperature threshold (see `schema.py`'s stated limitation). That's not a
sampling gap that a bigger pull would fix; the template mechanically cannot
produce those patterns. So the ID set is free but structurally incomplete,
and the OOD set is small but is the only place negation/numeric/no-constraint
queries exist at all. Training on one and testing on the other is the point,
not a compromise.

**Knobs to tune:** Arm A — prompt condition (bare schema / schema + field
descriptions / schema + few-shot), model tier. Arm B — LoRA rank/alpha,
learning rate, base model size, number of fine-tuning examples.

**Usage:**
```
python3 build_dataset.py                            # writes data/{train,val,ood_eval}.jsonl

# Arm A — prompting (needs: pip install anthropic; export ANTHROPIC_API_KEY)
python3 prompt_arm.py --split ood_eval --condition bare
python3 prompt_arm.py --split ood_eval --condition described
python3 prompt_arm.py --split ood_eval --condition few_shot --model claude-opus-5

# Arm B — fine-tuning (needs: pip install torch transformers peft accelerate; a GPU)
python3 finetune_arm.py --max-train 500 --epochs 1    # quick smoke test
python3 finetune_arm.py                                # full run

# scoring (stdlib only, no extra deps)
python3 evaluate.py data/predictions_prompt_claude-sonnet-5_described_ood_eval.jsonl
python3 evaluate.py data/predictions_finetune_Qwen2.5-1.5B-Instruct_ood_eval.jsonl
```

## Experiment

**Setup:** `data/train.jsonl` (4,000, weak-labelled) / `data/val.jsonl` (500,
weak-labelled, same distribution as train) / `data/ood_eval.jsonl` (49,
hand-labelled: negation, numeric budget/temperature, combinations, and
no-constraint queries). Arm B trains only on `train.jsonl`.

**Conditions:**
- Arm A: prompt condition {bare, described, few_shot (k=8)} × model tier
  {e.g. `claude-sonnet-5`, `claude-opus-5`}.
- Arm B: base model {Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct} ×
  LoRA rank {8, 16}.
- Both scored on `val.jsonl` (ID) and `ood_eval.jsonl` (OOD) separately.

**Metrics** (`evaluate.py`, all reported per-field, ID and OOD never
averaged together): schema-validity rate; set-F1 over tag identity for
`required_tags` and `excluded_tags` separately; exact-match accuracy for
`budget_at_most`, `month_temp_at_least`, `month_temp_at_most`; whole-intent
exact-match rate; latency per parse. Cost per parse for Arm A (token usage is
in `prompt_arm.py`'s cache entries); training time and hardware for Arm B.

**Interpretation — the pre-specified prediction:** Arm B should match or beat
Arm A on the ID val split — it's fitting the exact template family the
training data was generated from. On the OOD set it should lag specifically
on `excluded_tags` and the two `month_temp_*` fields, since it never saw a
single labelled example of either during training; a large prompted model
should generalise better there because it's applying general-purpose
language understanding, not pattern-matching a narrow template. If the
fine-tuned model instead holds up on OOD negation/numeric fields, that's the
more interesting result and the write-up should say so plainly — a small
LoRA-tuned model generalising past its training template on the specific
fields the codebase relies on `filter_constraints` to catch would be worth
flagging loudly, not burying.

The no-constraint OOD queries are a separate diagnostic: a model that always
gets a high `required_tags_f1` by over-triggering will show it here as a low
score specifically on those five examples, which a blended F1 would hide.

**Results:** *not yet run — dataset built, both arms scaffolded, no
prediction files yet.*

**Status:**
- `schema.py`: done — schema, weak-labeller, and the round-trip through
  `filter_constraints`'s predicates, spot-checked.
- `ood_queries.py`: done — 34 hand-labelled queries extending
  `LABELLED_QUERIES`'s 15.
- `build_dataset.py`: done and run — `data/train.jsonl` (4,000),
  `data/val.jsonl` (500), `data/ood_eval.jsonl` (49) written.
- `evaluate.py`: done and sanity-checked (perfect-prediction and
  always-empty-prediction synthetic inputs score as expected).
- `prompt_arm.py`: written, not yet run against the API (needs
  `anthropic` + credentials, same gap as `eval/`'s LLM baseline).
- `finetune_arm.py`: written, not yet run (needs `torch`/`transformers`/`peft`
  + a GPU, none available in this environment).
