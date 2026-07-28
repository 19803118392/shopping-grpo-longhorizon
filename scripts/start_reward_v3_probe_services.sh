#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR=/root/autodl-tmp/outputs/grpo_reward_v3_fresh_v1_probe
MODEL=/root/autodl-tmp/checkpoints/qwen35-2b-sft-v1-fresh-merged

mkdir -p "$RUNTIME_DIR"

if screen -ls | grep -q '[.]shopsim-rv3'; then
  echo "screen shopsim-rv3 already exists" >&2
  exit 1
fi
if screen -ls | grep -q '[.]vllm-rv3-fresh'; then
  echo "screen vllm-rv3-fresh already exists" >&2
  exit 1
fi

screen -dmS shopsim-rv3 bash -lc "
  set -euo pipefail
  source '$PROJECT_ROOT/environments/ShopSimulator/.venv-shopsim-v2/bin/activate'
  cd '$PROJECT_ROOT/environments/ShopSimulator/shop_env'
  SHOPSIM_ENV_SLOTS=8 SHOPSIM_PORT=5700 ./run_environment_v2_1.sh \
    >> '$RUNTIME_DIR/shopsim.log' 2>&1
"

screen -dmS vllm-rv3-fresh bash -lc "
  set -euo pipefail
  cd '$PROJECT_ROOT'
  export HF_HOME=/root/autodl-tmp/.cache/huggingface
  export OMP_NUM_THREADS=1
  export VLLM_USE_FLASHINFER_SAMPLER=0
  .venv-grpo-v080/bin/vllm serve '$MODEL' \
    --host 127.0.0.1 \
    --port 8000 \
    --served-model-name qwen35-2b-sft-v1-fresh \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 24576 \
    --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    >> '$RUNTIME_DIR/vllm.log' 2>&1
"

echo "started shopsim-rv3 and vllm-rv3-fresh"
echo "ShopSimulator log: $RUNTIME_DIR/shopsim.log"
echo "vLLM log: $RUNTIME_DIR/vllm.log"
