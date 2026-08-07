# GPU evaluation runbook

The repository supports `Baseline → SFT → GRPO → Evaluation` under ShopSimulator
Environment v2.1, Observation v2, Tool v2 and terminal Reward v3.

This runbook measures whether Terminal-GRPO improves over SFT. Model selection uses
only `data/grpo/validation.jsonl`. The frozen `data/evaluation/tasks.jsonl` is not
opened for repeated sampling or tuning.

The protocol has been completed once on the current branch. Its compact result,
configuration, and hashes are stored in the
[Validation-50×3 experiment card](../experiments/validation-50x3/README.md).

## 1. Preflight and fixed-seed replay

```bash
bash scripts/setup.sh
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src scripts environments/ShopSimulator/shop_env
.venv/bin/shopping-grpo smoke --json
```

Before the full comparison, run the same SFT service twice on 5 tasks × 2 attempts
with protocol `seed-replay`, then verify:

```bash
.venv/bin/python scripts/verify_seed_replay.py \
  --first outputs/eval/seed-a/raw.jsonl \
  --second outputs/eval/seed-b/raw.jsonl \
  --output outputs/eval/seed-replay.json
```

The verifier must report `exact_model_and_action_replay=true`. Otherwise results are
task-paired only and must not be described as common-random-number estimates.

## 2. Matched SFT and GRPO evaluation

Serve `outputs/models/sft-merged`, then run:

```bash
.venv/bin/python scripts/evaluate_shop_benchmark.py \
  --protocol dev50x3 --benchmark data/grpo/validation.jsonl --limit 50 \
  --output outputs/eval/sft-50x3/raw.jsonl \
  --summary outputs/eval/sft-50x3/summary.json \
  --model shopping-agent --llm-base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY --attempts-per-task 3 --temperature 0.7 --top-p 0.9 \
  --max-steps 35 --seed 2026
```

Stop the SFT server, serve `outputs/models/terminal-grpo-30-merged`, and repeat with
output directory `outputs/eval/terminal-grpo-50x3`. The task order, attempt indices,
seed, temperature, top-p, context limits and ShopSimulator process must remain fixed.

## 3. Paired report

```bash
.venv/bin/python scripts/compare_repeated_evaluations.py \
  --benchmark data/grpo/validation.jsonl --limit 50 \
  --baseline outputs/eval/sft-50x3/raw.jsonl \
  --candidate outputs/eval/terminal-grpo-50x3/raw.jsonl \
  --attempts-per-task 3 --bootstrap-samples 10000 --seed 2026 \
  --baseline-label SFT --candidate-label Terminal-GRPO \
  --output outputs/eval/sft-vs-terminal-50x3/comparison.json \
  --markdown-output outputs/eval/sft-vs-terminal-50x3/report.md \
  --csv-output outputs/eval/sft-vs-terminal-50x3/report.csv
```

Headline claims require 100% attempt coverage, no infrastructure-invalid attempts,
no critical footer failures, and a paired bootstrap interval. Report the observed
effect and interval even when McNemar is not significant. Difficulty strata are
exploratory; do not select a model from a favorable subgroup.

## 4. Optional training variance

Only if budget permits, train short Terminal-GRPO runs with three seeds and evaluate
each checkpoint with the same 50-task protocol. Report inference variance within a
checkpoint separately from variance across training seeds. This optional experiment
must not delay or alter the primary matched SFT-vs-GRPO comparison.

## 5. Final-200 boundary

Final-200 remains one deterministic attempt per task after configuration and model
freeze. It is not used for setting selection and is never repeated after results are
visible. Starting Final-200 still requires explicit user authorization.
