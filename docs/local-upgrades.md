# Local upgrades for the next GPU run

This branch adds three opt-in experiment capabilities without changing the frozen
ShopSimulator Environment v2.1 or Reward v3 contract.

## 1. Public Evidence Memory

`shopping-evidence-memory-v1` keeps a bounded task-local ledger of search queries,
candidates, prices, attributes, selected options, and visited information subpages.
It accepts only `shopping-observation-v2`; the existing renderer rejects hidden
goal, reward, answer, and target fields before they can enter memory. The current
observation remains the only action-validity boundary.

Evaluation ablation:

```bash
python scripts/evaluate_shop_benchmark.py \
  --benchmark data/evaluation/tasks.jsonl \
  --output outputs/eval/sft-memory/raw.jsonl \
  --summary outputs/eval/sft-memory/summary.json \
  --model outputs/models/sft-merged \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --attempts-per-task 5 \
  --temperature 0.7 \
  --top-p 0.9 \
  --evidence-memory
```

Run the control with identical arguments and without `--evidence-memory`. For GRPO
training, set `SHOPPING_EVIDENCE_MEMORY_ENABLE=true`; the default is `false`, so
existing runs retain their original prompt distribution.

## 2. Repeated-run statistics

The evaluator uses every `(task_id, attempt_index)` as a resumable key. Missing
attempts stay in the fixed denominator. Compare two complete JSONL files with:

```bash
python scripts/compare_repeated_evaluations.py \
  --benchmark data/evaluation/tasks.jsonl \
  --baseline outputs/eval/sft-control/raw.jsonl \
  --candidate outputs/eval/sft-memory/raw.jsonl \
  --attempts-per-task 5 \
  --bootstrap-samples 10000 \
  --seed 2026 \
  --baseline-label sft-control \
  --candidate-label sft-evidence-memory \
  --output outputs/eval/sft-memory-vs-control.json
```

The JSON report includes attempt coverage, strict success with a Wilson 95%
interval, empirical `pass@k` and `pass^k`, task-level wins/ties/losses, a paired
bootstrap confidence interval, an exact McNemar test, and SHA-256 input provenance.

## 3. Length curriculum preparation

The curriculum planner uses only teacher probe step counts already checked into
`data/grpo/train.jsonl`. It does not inspect hidden goals. The default cumulative
stages contain 695 short tasks, 896 short+medium tasks, and all 1000 tasks.

```bash
python scripts/prepare_grpo_curriculum.py \
  --metadata data/grpo/train.jsonl \
  --source-parquet data/grpo/train.parquet \
  --output-dir outputs/curriculum \
  --dry-run
```

Run this inside the pinned GRPO environment (which includes PyArrow via veRL) and
remove `--dry-run` to materialize three Parquet files and a hash manifest. Staged
checkpoint resume has not yet been validated
against the pinned veRL 0.8 runtime, so curriculum data is experimental and must
not replace the primary GRPO recipe until a resume smoke test passes.

## Local acceptance

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m compileall -q src scripts
shopping-grpo smoke --json
```

veRL/PyTorch patch tests skip in a CPU-only environment and run automatically when
the pinned GPU training packages are installed.
