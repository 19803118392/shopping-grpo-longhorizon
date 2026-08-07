#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${1:-model}"
if [[ $# -gt 0 ]]; then
  shift
fi
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$ROOT/outputs/evaluation/$LABEL}"
SHOPSIM_BASE_URL="${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-shopping-agent}"
FINAL_ARTIFACT_MANIFEST="${FINAL_ARTIFACT_MANIFEST:-}"

if [[ -z "$FINAL_ARTIFACT_MANIFEST" ]]; then
  echo "Final-200 requires FINAL_ARTIFACT_MANIFEST from freeze_final_candidate.py" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
cd "$ROOT"
exec "$ROOT/.venv/bin/python" scripts/evaluate_shop_benchmark.py \
  --benchmark data/evaluation/tasks.jsonl \
  --final-200 \
  --frozen-artifact-manifest "$FINAL_ARTIFACT_MANIFEST" \
  --output "$OUTPUT_DIR/trajectories.jsonl" \
  --summary "$OUTPUT_DIR/summary.json" \
  --base-url "$SHOPSIM_BASE_URL" \
  --model "$SERVED_MODEL_NAME" \
  --llm-base-url "$LLM_BASE_URL" \
  --api-key "$LLM_API_KEY" \
  --attempts-per-task 1 \
  --temperature 0.0 \
  --top-p 1.0 \
  --max-steps 35 \
  "$@"
