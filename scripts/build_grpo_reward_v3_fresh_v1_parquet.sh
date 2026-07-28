#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv-grpo-v080/bin/python"
TRAIN_TASKS="$PROJECT_ROOT/data/splits/grpo_reward_v3_fresh_v1_train.jsonl"
VAL_TASKS="$PROJECT_ROOT/data/splits/grpo_reward_v3_fresh_v1_val.jsonl"
PROBE="$PROJECT_ROOT/outputs/grpo_reward_v3_fresh_v1_probe/raw.jsonl"
TRAIN_OUTPUT="$PROJECT_ROOT/data/verl/grpo_reward_v3_fresh_v1_train.parquet"
VAL_OUTPUT="$PROJECT_ROOT/data/verl/grpo_reward_v3_fresh_v1_val.parquet"
BASE_URL="${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}"

export PYTHONPATH="$PROJECT_ROOT/src"
export HF_HOME=/root/autodl-tmp/.cache/huggingface

for required in "$PYTHON_BIN" "$TRAIN_TASKS" "$VAL_TASKS" "$PROBE"; do
  if [[ ! -e "$required" ]]; then
    echo "missing Reward v3 parquet input: $required" >&2
    exit 1
  fi
done
for output in "$TRAIN_OUTPUT" "$VAL_OUTPUT"; do
  if [[ -e "$output" ]]; then
    echo "refusing to overwrite frozen Reward v3 parquet: $output" >&2
    exit 1
  fi
done

cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/prepare_verl_grpo_dataset.py \
  --tasks "$TRAIN_TASKS" \
  --output "$TRAIN_OUTPUT" \
  --metadata "$PROJECT_ROOT/data/verl/grpo_reward_v3_fresh_v1_train.metadata.json" \
  --base-url "$BASE_URL" \
  --split train \
  --required-environment-version shopsimulator-environment-v2.1 \
  --reward-contract shopsimulator-reward-v3 \
  --source-probe "$PROBE"

"$PYTHON_BIN" scripts/prepare_verl_grpo_dataset.py \
  --tasks "$VAL_TASKS" \
  --output "$VAL_OUTPUT" \
  --metadata "$PROJECT_ROOT/data/verl/grpo_reward_v3_fresh_v1_val.metadata.json" \
  --base-url "$BASE_URL" \
  --split validation \
  --required-environment-version shopsimulator-environment-v2.1 \
  --reward-contract shopsimulator-reward-v3 \
  --source-probe "$PROBE"
