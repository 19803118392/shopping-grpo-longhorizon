"""每条 veRL trajectory 的轻量运行状态；不保存 ShopSimulator 隐藏 goal。"""

from __future__ import annotations

from contextvars import ContextVar
import hashlib
import json
import math
from collections.abc import Mapping


current_environment: ContextVar = ContextVar("shopsimulator_environment", default=None)
current_runtime_state: ContextVar = ContextVar("shopsimulator_runtime_state", default=None)
REWARD_COMPONENT_NAMES = ("r_type", "r_att", "r_option", "r_price")


def make_runtime_state(task_id: int, max_steps: int) -> dict:
    """创建只含公共运行诊断的状态，reward 仅在环境正常终局后写入。"""
    return {
        "task_id": int(task_id),
        "max_steps": int(max_steps),
        "steps": [],
        "done": False,
        "terminate": False,
        "termination_reason": None,
        "consecutive_guard_rejections": 0,
        "action_attempt_count": 0,
        "repeat_action_count": 0,
        "recent_action_signatures": [],
        "terminal_result": {},
        "final_reward": 0.0,
        "reward_components": None,
        "infrastructure_invalid": False,
        "error": None,
        "context_compactions": 0,
        "context_tokens_removed": 0,
    }


def record_action_attempt(state: dict, tool_name: str, parameters: dict, observation: str) -> None:
    """记录环境动作尝试，并统计最近三次中的重复签名。"""
    if tool_name == "think":
        return
    canonical_parameters = json.dumps(
        parameters,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    observation_fingerprint = hashlib.sha256(str(observation).encode("utf-8")).hexdigest()
    signature = (str(tool_name), canonical_parameters, observation_fingerprint)
    recent = state["recent_action_signatures"]
    state["action_attempt_count"] += 1
    if signature in recent:
        state["repeat_action_count"] += 1
    recent.append(signature)
    del recent[:-3]


def validate_reward_components(raw_components: object) -> dict[str, float]:
    """校验并复制 ShopSimulator 终局评分，不保留其他隐藏字段。"""
    if not isinstance(raw_components, Mapping):
        raise ValueError("reward_detail must be an object")
    missing = [name for name in REWARD_COMPONENT_NAMES if name not in raw_components]
    if missing:
        raise ValueError("reward_detail missing components: " + ", ".join(missing))

    components = {}
    for name in REWARD_COMPONENT_NAMES:
        try:
            value = float(raw_components[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"reward_detail {name} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"reward_detail {name} must be finite")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"reward_detail {name} must be in [0, 1]")
        components[name] = value
    return components


def _normal_terminal(state: dict) -> bool:
    terminal = state.get("terminal_result") or {}
    return (
        state.get("done") is True
        and terminal.get("done") is True
        and terminal.get("over") is True
    )


def reward_breakdown(state: dict) -> dict[str, float | bool]:
    """计算约束感知终局奖励；基础设施无效轨迹只返回诊断，不制造学习信号。"""
    invalid = bool(state.get("infrastructure_invalid"))
    normal_terminal = _normal_terminal(state)
    native = float(state.get("final_reward", 0.0)) if normal_terminal else 0.0
    if not math.isfinite(native):
        invalid = True
        native = 0.0

    raw_components = state.get("reward_components")
    if normal_terminal and raw_components is None:
        invalid = True
    if raw_components is None:
        components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
    else:
        try:
            components = validate_reward_components(raw_components)
        except ValueError:
            invalid = True
            components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
    action_attempts = max(int(state.get("action_attempt_count", 0)), 1)
    repeat_action_rate = int(state.get("repeat_action_count", 0)) / action_attempts

    if invalid:
        return {
            **components,
            "full": 0.0,
            "strict": 0.0,
            "native": native,
            "semantic": 0.0,
            "efficiency": 0.0,
            "penalty_overlong": 0.0,
            "penalty_unfinished": 0.0,
            "penalty_repeat": 0.0,
            "repeat_action_rate": repeat_action_rate,
            "total": 0.0,
            "infrastructure_invalid": True,
        }

    full = float(all(components[name] == 1.0 for name in REWARD_COMPONENT_NAMES))
    strict = math.prod(components[name] for name in REWARD_COMPONENT_NAMES)
    semantic = full + 0.5 * strict + 0.2 * native
    steps = min(max(len(state.get("steps") or []), 0), max(int(state.get("max_steps", 35)), 1))
    max_steps = max(int(state.get("max_steps", 35)), 1)
    efficiency = 0.05 * full * max(0.0, 1.0 - steps / max_steps)
    penalty_overlong = (
        0.05 * (1.0 - full) * (steps - 28) / max(max_steps - 28, 1)
        if steps > 28
        else 0.0
    )
    penalty_overlong = min(max(penalty_overlong, 0.0), 0.05)
    penalty_unfinished = (
        0.05
        if state.get("termination_reason") == "assistant_finished_without_environment_done"
        else 0.0
    )
    penalty_repeat = 0.03 * repeat_action_rate
    total = semantic + efficiency - penalty_overlong - penalty_unfinished - penalty_repeat
    return {
        **components,
        "full": full,
        "strict": strict,
        "native": native,
        "semantic": semantic,
        "efficiency": efficiency,
        "penalty_overlong": penalty_overlong,
        "penalty_unfinished": penalty_unfinished,
        "penalty_repeat": penalty_repeat,
        "repeat_action_rate": repeat_action_rate,
        "total": total,
        "infrastructure_invalid": False,
    }


def terminal_reward(state: dict, mode: str = "native") -> float:
    """按实验模式返回原生或约束感知奖励。"""
    if mode == "constraint_aware":
        return float(reward_breakdown(state)["total"])
    if mode != "native":
        raise ValueError(f"unknown shopping reward mode: {mode!r}")
    if state.get("infrastructure_invalid") or state.get("error") or not _normal_terminal(state):
        return 0.0
    return float(state.get("final_reward", 0.0))


def task_id_from_kwargs(kwargs: dict) -> int:
    """从 veRL parquet 的 extra_info 读取当前任务，缺失时立即失败。"""
    extra_info = kwargs.get("extra_info")
    if hasattr(extra_info, "item"):
        extra_info = extra_info.item()
    if not isinstance(extra_info, dict) or "task_id" not in extra_info:
        raise ValueError("veRL sample extra_info is missing task_id")
    return int(extra_info["task_id"])
