#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-Qwen/Qwen3.5-2B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-shopping-agent}"
LLM_PORT="${LLM_PORT:-8000}"

# FlashInfer 0.6.13 can mis-detect Blackwell SM12 when paired with the
# CUDA 13 wheels pinned by vLLM 0.25.1, then crash during sampler warmup.
# vLLM's native sampler implements the same top-k/top-p contract and is the
# documented fallback.  Keep an explicit caller override for future runtimes.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

if [[ ! -x "$ROOT/.venv/bin/vllm" ]]; then
  echo "vLLM is not installed. Run: bash scripts/setup.sh" >&2
  exit 1
fi

exec "$ROOT/.venv/bin/vllm" serve "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --port "$LLM_PORT" \
  --max-model-len 24576 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
