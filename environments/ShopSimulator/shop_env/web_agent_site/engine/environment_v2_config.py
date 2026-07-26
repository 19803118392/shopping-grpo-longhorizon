"""Load and validate the frozen ShopSimulator Environment v2 contract."""

from __future__ import annotations

import json
from pathlib import Path

from web_agent_site.engine.observation_v2 import OBSERVATION_VERSION
from web_agent_site.engine.reward_v2 import DEFAULT_REWARDS, REWARD_VERSION
from web_agent_site.engine.search_v2 import DEFAULT_FIELD_WEIGHTS, SEARCH_VERSION


ENVIRONMENT_VERSION = "shopsimulator-environment-v2"
TOOL_VERSION = "shopping-tools-v2"
SEARCH_TOP_K = 150
SEARCH_PAGE_SIZE = 20


def load_environment_v2_config(path):
    config_path = Path(path).resolve()
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot load Environment v2 config {config_path}: {exc}"
        ) from exc
    validate_environment_v2_config(config)
    return config


def validate_environment_v2_config(config):
    if not isinstance(config, dict):
        raise ValueError("Environment v2 config must be an object")
    if config.get("environment_version") != ENVIRONMENT_VERSION:
        raise ValueError("Environment v2 config has the wrong environment_version")

    search = config.get("search")
    if not isinstance(search, dict) or search.get("version") != SEARCH_VERSION:
        raise ValueError("Environment v2 config has the wrong search version")
    if int(search.get("top_k", 0)) != SEARCH_TOP_K:
        raise ValueError(f"Environment v2 search top_k must equal {SEARCH_TOP_K}")
    if int(search.get("page_size", 0)) != SEARCH_PAGE_SIZE:
        raise ValueError(
            f"Environment v2 search page_size must equal {SEARCH_PAGE_SIZE}"
        )
    if search.get("field_weights") != DEFAULT_FIELD_WEIGHTS:
        raise ValueError(
            "Environment v2 search field weights differ from the index contract"
        )

    reward = config.get("reward")
    if not isinstance(reward, dict) or reward.get("version") != REWARD_VERSION:
        raise ValueError("Environment v2 config has the wrong reward version")
    reward_values = {
        key: float(reward.get(key)) for key in DEFAULT_REWARDS if key in reward
    }
    if reward_values != DEFAULT_REWARDS:
        raise ValueError(
            "Environment v2 reward values differ from the runtime contract"
        )

    termination = config.get("termination")
    if not isinstance(termination, dict):
        raise ValueError("Environment v2 config is missing termination settings")
    for name in ("exact_repeat_limit", "no_new_asin_limit", "max_steps"):
        if int(termination.get(name, 0)) <= 0:
            raise ValueError(f"Environment v2 termination.{name} must be positive")

    if config.get("observation_version") != OBSERVATION_VERSION:
        raise ValueError("Environment v2 config has the wrong observation version")
    if config.get("tool_version") != TOOL_VERSION:
        raise ValueError("Environment v2 config has the wrong tool version")
    return config
