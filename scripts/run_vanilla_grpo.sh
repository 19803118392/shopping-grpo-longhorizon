#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXPERIMENT="${SHOPPING_GRPO_EXPERIMENT:-legacy}"
if [[ "${1:-}" == "a0" || "${1:-}" == "a1" ]]; then
  EXPERIMENT="$1"
  shift
fi

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi

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
      trainer.experiment_name=shopping-grpo-a0-native
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
      trainer.experiment_name=shopping-grpo-a1-constraint
    )
    ;;
  legacy)
    export SHOPPING_REWARD_MODE="${SHOPPING_REWARD_MODE:-native}"
    ;;
  *)
    echo "unknown experiment '$EXPERIMENT'; expected a0 or a1" >&2
    exit 2
    ;;
esac

if [[ "$DRY_RUN" == true ]]; then
  echo "SHOPPING_GRPO_EXPERIMENT=$EXPERIMENT"
  echo "SHOPPING_REWARD_MODE=$SHOPPING_REWARD_MODE"
  printf 'hydra_overrides:'
  printf ' %s' "${EXPERIMENT_OVERRIDES[@]}" "$@"
  printf '\n'
  exit 0
fi

: "${GRPO_MODEL_PATH:?set GRPO_MODEL_PATH to the merged SFT checkpoint or model path}"
: "${GRPO_TRAIN_FILE:?set GRPO_TRAIN_FILE to the training parquet}"
: "${GRPO_VAL_FILE:?set GRPO_VAL_FILE to the validation parquet}"
: "${GRPO_OUTPUT_DIR:?set GRPO_OUTPUT_DIR to a new checkpoint directory}"

export SHOPPING_GRPO_ROOT="$PROJECT_ROOT"
# 不继承旧 shell 中可能指向 reference fork 的 PYTHONPATH。
export PYTHONPATH="$PROJECT_ROOT/src"
export SHOPSIM_BASE_URL="${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}"
# veRL's W&B backend receives the exact same metrics dictionary as the console
# logger. Keep online monitoring explicit while allowing an intentional
# WANDB_MODE=offline override for disconnected debugging.
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$GRPO_OUTPUT_DIR/wandb}"
mkdir -p "$WANDB_DIR"

cd "$PROJECT_ROOT"

python3 "$PROJECT_ROOT/scripts/generate_verl_shop_configs.py" \
  --tool-output "$PROJECT_ROOT/configs/verl/shop_tools.json"

python3 "$PROJECT_ROOT/scripts/check_grpo_runtime.py" "${EXPERIMENT_OVERRIDES[@]}" "$@"

exec python3 -m verl.trainer.main_ppo \
  --config-path="$PROJECT_ROOT/configs/verl" \
  --config-name=vanilla_grpo \
  "${EXPERIMENT_OVERRIDES[@]}" \
  "$@"
