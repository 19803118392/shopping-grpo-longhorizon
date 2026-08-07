#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTOR_CHECKPOINT="${1:?usage: bash scripts/export_grpo.sh <global_step_*/actor> [output_dir]}"
OUTPUT_DIR="${2:-$ROOT/outputs/models/grpo-merged}"

if [[ ! -d "$ACTOR_CHECKPOINT" ]]; then
  echo "Actor checkpoint does not exist: $ACTOR_CHECKPOINT" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite existing export: $OUTPUT_DIR" >&2
  exit 2
fi

OUTPUT_PARENT="$(dirname "$OUTPUT_DIR")"
mkdir -p "$OUTPUT_PARENT"
STAGING_ROOT="$(mktemp -d "$OUTPUT_PARENT/.grpo-export.XXXXXX")"
RAW_EXPORT="$STAGING_ROOT/verl-export"
MERGED_EXPORT="$STAGING_ROOT/merged"
trap 'rm -rf -- "$STAGING_ROOT"' EXIT

"$ROOT/.venv/bin/python" -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$ACTOR_CHECKPOINT" \
  --target_dir "$RAW_EXPORT" \
  --trust-remote-code

if [[ ! -s "$RAW_EXPORT/lora_adapter/adapter_model.safetensors" ]]; then
  echo "veRL export did not produce a LoRA adapter: $RAW_EXPORT/lora_adapter" >&2
  exit 2
fi

"$ROOT/.venv/bin/python" "$ROOT/scripts/merge_lora_adapter.py" \
  --base-model "$RAW_EXPORT" \
  --adapter "$RAW_EXPORT/lora_adapter" \
  --output "$MERGED_EXPORT" \
  --source-checkpoint "$ACTOR_CHECKPOINT" \
  --bf16

if [[ -e "$MERGED_EXPORT/lora_adapter" ]]; then
  echo "Final export still contains an unmerged LoRA adapter" >&2
  exit 2
fi
if [[ ! -s "$MERGED_EXPORT/merge_manifest.json" ]]; then
  echo "Final export is missing merge_manifest.json" >&2
  exit 2
fi

mv "$MERGED_EXPORT" "$OUTPUT_DIR"
echo "Exported fully merged GRPO model to $OUTPUT_DIR"
