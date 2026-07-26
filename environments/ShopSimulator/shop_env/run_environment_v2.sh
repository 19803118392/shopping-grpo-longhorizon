#!/usr/bin/env bash
set -euo pipefail

SHOP_ENV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INDEX_PATH="${SHOP_SEARCH_INDEX:-${SHOP_ENV_ROOT}/search_engine/environment_v2.sqlite3}"

if [[ ! -f "${INDEX_PATH}" ]]; then
  echo "Environment v2 index is missing: ${INDEX_PATH}" >&2
  echo "Run: python scripts/build_environment_v2_index.py" >&2
  exit 1
fi

export SHOP_ENVIRONMENT_VERSION=shopsimulator-environment-v2
export SHOP_ENV_CONFIG="${SHOP_ENV_CONFIG:-${SHOP_ENV_ROOT}/configs/environment_v2.json}"
export SHOP_SEARCH_BACKEND=multifield_bm25_v2
export SHOP_SEARCH_INDEX="${INDEX_PATH}"
export SHOP_MAX_STEPS="${SHOP_MAX_STEPS:-35}"
export SHOPSIM_ENV_SLOTS="${SHOPSIM_ENV_SLOTS:-8}"
export SHOPSIM_PORT="${SHOPSIM_PORT:-5700}"

cd "${SHOP_ENV_ROOT}/shop_env"
exec python pack_api.py
