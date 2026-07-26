"""Pure reward-group selection used by the bounded veRL sampling patch."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from typing import Any


def aggregate_shopping_metrics(shopping_infos: Sequence[object]) -> dict[str, float]:
    """把 AgentLoop 轨迹诊断聚合为 veRL 每步指标。"""
    if not shopping_infos:
        return {}

    reward_keys = (
        "full",
        "strict",
        "native",
        "semantic",
        "total",
        "efficiency",
        "penalty_overlong",
        "penalty_unfinished",
        "penalty_repeat",
        "repeat_action_rate",
        "r_type",
        "r_att",
        "r_option",
        "r_price",
    )
    rewards = {key: [] for key in reward_keys}
    steps = []
    done = []
    max_steps = []
    infrastructure_invalid = []
    reward_unverifiable = []
    for index, info in enumerate(shopping_infos):
        if not isinstance(info, Mapping) or not isinstance(info.get("reward"), Mapping):
            raise ValueError(f"shopping extra field at index {index} is missing reward diagnostics")
        reward = info["reward"]
        for key in reward_keys:
            try:
                value = float(reward[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"shopping reward at index {index} is missing numeric {key}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"shopping reward {key} at index {index} is not finite")
            rewards[key].append(value)
        steps.append(float(info.get("steps", 0)))
        done.append(float(info.get("done") is True))
        max_steps.append(float(info.get("termination_reason") == "max_steps"))
        infrastructure_invalid.append(float(bool(info.get("infrastructure_invalid"))))
        reward_unverifiable.append(float(bool(info.get("reward_unverifiable"))))

    def mean(values):
        return sum(values) / len(values)

    return {
        "reward/full_mean": mean(rewards["full"]),
        "reward/strict_mean": mean(rewards["strict"]),
        "reward/native_mean": mean(rewards["native"]),
        "reward/semantic_mean": mean(rewards["semantic"]),
        "reward/shaped_min": min(rewards["total"]),
        "reward/shaped_mean": mean(rewards["total"]),
        "reward/shaped_max": max(rewards["total"]),
        "reward/efficiency_mean": mean(rewards["efficiency"]),
        "penalty/overlong_mean": mean(rewards["penalty_overlong"]),
        "penalty/unfinished_mean": mean(rewards["penalty_unfinished"]),
        "penalty/repeat_mean": mean(rewards["penalty_repeat"]),
        "component/r_type_mean": mean(rewards["r_type"]),
        "component/r_att_mean": mean(rewards["r_att"]),
        "component/r_option_mean": mean(rewards["r_option"]),
        "component/r_price_mean": mean(rewards["r_price"]),
        "trajectory/average_steps": mean(steps),
        "trajectory/done_rate": mean(done),
        "trajectory/max_steps_rate": mean(max_steps),
        "trajectory/repeat_action_rate": mean(rewards["repeat_action_rate"]),
        "trajectory/infrastructure_invalid_rate": mean(infrastructure_invalid),
        "trajectory/reward_unverifiable_rate": mean(reward_unverifiable),
    }


def extract_shopping_group_signals(shopping_infos: Sequence[object]) -> tuple[list[float], list[bool]]:
    """Return semantic reward and combined sampling-invalid flags.

    Infrastructure failures and unverifiable rewards remain separate in
    trajectory metrics, but either one makes a complete rollout group unsafe
    for an optimizer update.
    """
    semantic_rewards = []
    infrastructure_invalid = []
    for index, info in enumerate(shopping_infos):
        if not isinstance(info, Mapping) or not isinstance(info.get("reward"), Mapping):
            raise ValueError(f"shopping extra field at index {index} is missing reward diagnostics")
        try:
            semantic = float(info["reward"]["semantic"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"shopping extra field at index {index} is missing semantic reward"
            ) from exc
        if not math.isfinite(semantic):
            raise ValueError(f"shopping semantic reward at index {index} is not finite")
        if "infrastructure_invalid" not in info:
            raise ValueError(
                f"shopping extra field at index {index} is missing infrastructure_invalid"
            )
        semantic_rewards.append(semantic)
        infrastructure_invalid.append(
            bool(info["infrastructure_invalid"])
            or bool(info.get("reward_unverifiable"))
        )
    return semantic_rewards, infrastructure_invalid


def select_reward_varying_groups(
    uids: Sequence[Hashable],
    seq_rewards: Sequence[float],
    *,
    semantic_rewards: Sequence[float] | None = None,
    infrastructure_invalid: Sequence[bool] | None = None,
    tolerance: float = 1.0e-8,
) -> tuple[list[int], dict[str, Any]]:
    """Return trajectory indices belonging to groups with non-constant reward.

    Group order follows the first occurrence of each uid. Returned trajectory
    indices preserve their original order, so callers can safely apply the same
    selection to every aligned tensor and non-tensor batch field.
    """

    if len(uids) != len(seq_rewards):
        raise ValueError(
            f"uids and seq_rewards must have equal length, got {len(uids)} and {len(seq_rewards)}"
        )
    if semantic_rewards is not None and len(semantic_rewards) != len(uids):
        raise ValueError("semantic_rewards must have the same length as uids")
    if infrastructure_invalid is not None and len(infrastructure_invalid) != len(uids):
        raise ValueError("infrastructure_invalid must have the same length as uids")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError(f"tolerance must be a finite non-negative number, got {tolerance!r}")

    semantic_values = semantic_rewards if semantic_rewards is not None else seq_rewards
    invalid_values = infrastructure_invalid if infrastructure_invalid is not None else [False] * len(uids)
    grouped: dict[Hashable, dict[str, Any]] = {}
    for index, (uid, raw_reward, raw_semantic, raw_invalid) in enumerate(
        zip(uids, seq_rewards, semantic_values, invalid_values, strict=True)
    ):
        try:
            hash(uid)
        except TypeError as exc:
            raise ValueError(f"uid at index {index} is not hashable: {uid!r}") from exc

        reward = float(raw_reward)
        if not math.isfinite(reward):
            raise ValueError(f"seq_reward at index {index} is not finite: {raw_reward!r}")
        semantic = float(raw_semantic)
        if not math.isfinite(semantic):
            raise ValueError(f"semantic_reward at index {index} is not finite: {raw_semantic!r}")

        group = grouped.setdefault(
            uid,
            {
                "uid": uid,
                "indices": [],
                "rewards": [],
                "semantic_rewards": [],
                "infrastructure_invalid": [],
            },
        )
        group["indices"].append(index)
        group["rewards"].append(reward)
        group["semantic_rewards"].append(semantic)
        group["infrastructure_invalid"].append(bool(raw_invalid))

    kept_uids: list[Hashable] = []
    dropped_uids: list[Hashable] = []
    groups: list[dict[str, Any]] = []
    for uid, group in grouped.items():
        rewards = group["rewards"]
        reward_min = min(rewards)
        reward_max = max(rewards)
        reward_varying = reward_max - reward_min > tolerance
        semantic_positive = max(group["semantic_rewards"]) > 0.0
        has_infrastructure_invalid = any(group["infrastructure_invalid"])
        if has_infrastructure_invalid:
            drop_reason = "infrastructure_invalid"
        elif not reward_varying:
            drop_reason = "constant_reward"
        elif semantic_rewards is not None and not semantic_positive:
            drop_reason = "no_semantic_signal"
        else:
            drop_reason = None
        keep = drop_reason is None
        if keep:
            kept_uids.append(uid)
        else:
            dropped_uids.append(uid)
        groups.append(
            {
                "uid": uid,
                "indices": tuple(group["indices"]),
                "rewards": tuple(rewards),
                "semantic_rewards": tuple(group["semantic_rewards"]),
                "reward_min": reward_min,
                "reward_max": reward_max,
                "reward_varying": reward_varying,
                "semantic_positive": semantic_positive,
                "infrastructure_invalid": has_infrastructure_invalid,
                "drop_reason": drop_reason,
                "kept": keep,
            }
        )

    kept_uid_set = set(kept_uids)
    trajectory_indices = [index for index, uid in enumerate(uids) if uid in kept_uid_set]
    stats = {
        "num_trajectories": len(uids),
        "num_groups": len(grouped),
        "kept_group_count": len(kept_uids),
        "dropped_group_count": len(dropped_uids),
        "kept_uids": tuple(kept_uids),
        "dropped_uids": tuple(dropped_uids),
        "all_equal_group_count": sum(not group["reward_varying"] for group in groups),
        "all_zero_semantic_group_count": sum(
            max(abs(value) for value in group["semantic_rewards"]) <= tolerance
            for group in groups
        ),
        "all_full_success_group_count": sum(
            all(value >= 1.0 - tolerance for value in group["semantic_rewards"])
            for group in groups
        ),
        "no_semantic_signal_group_count": sum(
            not group["semantic_positive"] for group in groups
        ),
        "infrastructure_invalid_group_count": sum(
            group["infrastructure_invalid"] for group in groups
        ),
        "groups": tuple(groups),
    }
    return trajectory_indices, stats
