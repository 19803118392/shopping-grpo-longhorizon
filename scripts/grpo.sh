#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Keep standalone serving and veRL's embedded vLLM on the same sampler path.
# See scripts/serve_model.sh for the pinned-runtime compatibility rationale.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
cd "$ROOT"
exec "$ROOT/.venv/bin/python" scripts/train_grpo.py "$@"
