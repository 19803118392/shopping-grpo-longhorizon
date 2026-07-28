#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$PROJECT_ROOT/.venv-grpo-v080/bin/python"
EXPERIMENT="${1:-a0}"
if [[ "$EXPERIMENT" == "a0" || "$EXPERIMENT" == "a1" ]]; then
  shift || true
else
  echo "expected first argument a0 or a1" >&2
  exit 2
fi

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi

export PYTHONPATH="$PROJECT_ROOT/src"
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export OMP_NUM_THREADS=1
export SHOPPING_GRPO_ROOT="$PROJECT_ROOT"
export GRPO_CONFIG_NAME=vanilla_grpo_reward_v3_fresh_v1
export SHOPPING_ENVIRONMENT_VERSION=shopsimulator-environment-v2.1
export SHOPPING_ENV_MANIFEST="$PROJECT_ROOT/data/manifests/environment_v2_1_reward_v3_fresh_v1.json"
export SHOPPING_TOOL_CONFIG="$PROJECT_ROOT/configs/verl/shop_tools_v2.json"
export SHOPSIM_BASE_URL="${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}"
export GRPO_MODEL_PATH="$PROJECT_ROOT/../checkpoints/qwen35-2b-sft-v1-fresh-merged"
export GRPO_TRAIN_FILE="$PROJECT_ROOT/data/verl/grpo_reward_v3_fresh_v1_train.parquet"
export GRPO_VAL_FILE="$PROJECT_ROOT/data/verl/grpo_reward_v3_fresh_v1_val.parquet"

COMMON_OVERRIDES=(
  actor_rollout_ref.actor.use_kl_loss=false
  algorithm.use_kl_in_reward=false
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  actor_rollout_ref.actor.calculate_entropy=false
  actor_rollout_ref.actor.entropy_coeff=0.0
  actor_rollout_ref.actor.clip_ratio_low=0.20
  actor_rollout_ref.actor.clip_ratio_high=0.20
)
EXPERIMENT_OVERRIDES=()
case "$EXPERIMENT" in
  a0)
    export SHOPPING_REWARD_MODE=native
    EXPERIMENT_OVERRIDES=(
      "${COMMON_OVERRIDES[@]}"
      algorithm.norm_adv_by_std_in_grpo=true
      shopping_dynamic_sampling.enable=false
      trainer.experiment_name=reward-v3-fresh-v1-a0-native
    )
    ;;
  a1)
    export SHOPPING_REWARD_MODE=constraint_aware
    EXPERIMENT_OVERRIDES=(
      "${COMMON_OVERRIDES[@]}"
      algorithm.norm_adv_by_std_in_grpo=false
      shopping_dynamic_sampling.enable=true
      shopping_dynamic_sampling.max_num_gen_batches=3
      shopping_dynamic_sampling.max_consecutive_skipped_updates=10
      shopping_dynamic_sampling.reward_tolerance=1.0e-8
      trainer.experiment_name=reward-v3-fresh-v1-a1-constraint
    )
    ;;
esac

if [[ "$DRY_RUN" == true ]]; then
  echo "contract=Environment-v2.1/Reward-v3/fresh-v1"
  echo "config=$GRPO_CONFIG_NAME"
  echo "experiment=$EXPERIMENT"
  echo "reward_mode=$SHOPPING_REWARD_MODE"
  echo "model=$GRPO_MODEL_PATH"
  echo "train=$GRPO_TRAIN_FILE"
  echo "validation=$GRPO_VAL_FILE"
  echo "manifest=$SHOPPING_ENV_MANIFEST"
  echo "tools=$SHOPPING_TOOL_CONFIG"
  printf 'hydra_overrides:'
  printf ' %s' "${EXPERIMENT_OVERRIDES[@]}" "$@"
  printf '\n'
  exit 0
fi

: "${GRPO_OUTPUT_DIR:?set GRPO_OUTPUT_DIR to a new Reward v3 checkpoint directory}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "missing GRPO Python: $PYTHON_BIN" >&2
  exit 1
fi
for required in \
  "$SHOPPING_ENV_MANIFEST" \
  "$SHOPPING_TOOL_CONFIG" \
  "$GRPO_MODEL_PATH/config.json" \
  "$GRPO_MODEL_PATH/model.safetensors" \
  "$GRPO_TRAIN_FILE" \
  "$GRPO_VAL_FILE"; do
  if [[ ! -f "$required" ]]; then
    echo "missing Reward v3 runtime asset: $required" >&2
    exit 1
  fi
done
if [[ -e "$GRPO_OUTPUT_DIR/latest_checkpointed_iteration.txt" ]]; then
  echo "refusing to reuse an existing GRPO checkpoint directory: $GRPO_OUTPUT_DIR" >&2
  exit 1
fi

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$GRPO_OUTPUT_DIR/wandb}"
mkdir -p "$WANDB_DIR"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/check_grpo_runtime.py "${EXPERIMENT_OVERRIDES[@]}" "$@"

exec "$PYTHON_BIN" -m verl.trainer.main_ppo \
  --config-path="$PROJECT_ROOT/configs/verl" \
  --config-name="$GRPO_CONFIG_NAME" \
  "${EXPERIMENT_OVERRIDES[@]}" \
  "$@"
