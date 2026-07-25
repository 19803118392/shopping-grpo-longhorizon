"""固定 ShopSimulator benchmark 的任务划分与确定性统计。"""

import random
from collections import Counter


REWARD_KEYS = ("r_type", "r_att", "r_option", "r_price")


def build_benchmark_manifest(all_task_ids, excluded_task_ids, size, seed):
    """从未用于 SFT 的 task 中稳定抽取 benchmark 清单。"""
    candidates = sorted({int(task_id) for task_id in all_task_ids} - {int(task_id) for task_id in excluded_task_ids})
    size = int(size)
    if size < 1:
        raise ValueError("benchmark size must be positive")
    if size > len(candidates):
        raise ValueError("benchmark size exceeds available held-out tasks")
    random.Random(int(seed)).shuffle(candidates)
    return [{"task_id": task_id} for task_id in candidates[:size]]


def summarize_trajectories(expected_task_ids, trajectories):
    """以固定 benchmark 全体 task 为分母汇总严格购物成功率。"""
    expected_ids = [int(task_id) for task_id in expected_task_ids]
    expected_set = set(expected_ids)
    by_task = {}
    for trajectory in trajectories:
        task_id = trajectory.get("task_id")
        if task_id is not None and int(task_id) in expected_set:
            by_task[int(task_id)] = trajectory

    completed_ids = sorted(by_task)
    missing_ids = sorted(expected_set - set(completed_ids))
    strict_successes = [task_id for task_id, item in by_task.items() if _is_strict_success(item)]
    done_tasks = [task_id for task_id, item in by_task.items() if item.get("done")]
    component_successes = {
        key: sum(_reward_detail(item).get(key) == 1 or _reward_detail(item).get(key) is True for item in by_task.values())
        for key in REWARD_KEYS
    }
    steps = [len(item.get("steps") or []) for item in by_task.values()]
    guard_reasons = Counter(
        blocked.get("reason", "unknown")
        for item in by_task.values()
        for blocked in item.get("blocked_tool_calls") or []
    )
    statuses = Counter(item.get("status", "unknown") for item in by_task.values())
    projections = [
        step["projection"]
        for item in by_task.values()
        for step in item.get("steps") or []
        if isinstance(step.get("projection"), dict)
    ]
    truncated_task_ids = {
        task_id
        for task_id, item in by_task.items()
        if any(
            bool((step.get("projection") or {}).get("truncated"))
            for step in item.get("steps") or []
        )
    }
    guard_rejections = [
        blocked
        for item in by_task.values()
        for blocked in item.get("blocked_tool_calls") or []
    ]
    executed_after_truncation = sum(
        bool(((steps[index - 1].get("projection") or {}).get("truncated")))
        for item in by_task.values()
        for steps in [item.get("steps") or []]
        for index in range(1, len(steps))
    )
    guard_rejections_after_truncation = sum(
        bool(item.get("latest_observation_truncated")) for item in guard_rejections
    )
    context_overflows = sum(
        (item.get("error") or {}).get("type") == "ContextBudgetError"
        for item in by_task.values()
    )
    denominator = len(expected_ids)
    return {
        "expected_tasks": denominator,
        "completed_tasks": len(completed_ids),
        "missing_tasks": missing_ids,
        "done_tasks": len(done_tasks),
        "done_rate": len(done_tasks) / denominator if denominator else 0.0,
        "strict_successes": len(strict_successes),
        "strict_success_task_ids": sorted(strict_successes),
        "strict_success_rate": len(strict_successes) / denominator if denominator else 0.0,
        "reward_component_rates": {
            key: component_successes[key] / denominator if denominator else 0.0
            for key in REWARD_KEYS
        },
        "mean_final_reward": (
            sum(float(item.get("final_reward", 0.0)) for item in by_task.values()) / len(by_task)
            if by_task
            else 0.0
        ),
        "average_steps": sum(steps) / len(steps) if steps else 0.0,
        "status_counts": dict(sorted(statuses.items())),
        "guard_reason_counts": dict(sorted(guard_reasons.items())),
        "context_projection": {
            "projected_tool_observations": len(projections),
            "truncated_tool_observations": sum(
                bool(item.get("truncated")) for item in projections
            ),
            "raw_tokens": sum(int(item.get("raw_tokens", 0)) for item in projections),
            "visible_tokens": sum(
                int(item.get("visible_tokens", 0)) for item in projections
            ),
            "critical_footer_failures": sum(
                not bool(item.get("critical_footer_preserved", False))
                for item in projections
            ),
            "visible_asin_count": sum(
                int(item.get("visible_asin_count", 0)) for item in projections
            ),
            "visible_button_count": sum(
                int(item.get("visible_button_count", 0)) for item in projections
            ),
            "guard_rejections": len(guard_rejections),
            "guard_rejections_after_truncation": guard_rejections_after_truncation,
            "action_attempts_after_truncation": (
                executed_after_truncation + guard_rejections_after_truncation
            ),
            "guard_rejection_rate_after_truncation": (
                guard_rejections_after_truncation
                / (executed_after_truncation + guard_rejections_after_truncation)
                if executed_after_truncation + guard_rejections_after_truncation
                else 0.0
            ),
            "context_overflow_tasks": context_overflows,
            "max_context_input_tokens": max(
                (
                    int(turn.get("input_tokens", 0))
                    for item in by_task.values()
                    for turn in item.get("context_turn_tokens") or []
                ),
                default=0,
            ),
            "success_by_truncation_bucket": {
                "none": {
                    "tasks": len(set(by_task) - truncated_task_ids),
                    "strict_successes": sum(
                        _is_strict_success(item)
                        for task_id, item in by_task.items()
                        if task_id not in truncated_task_ids
                    ),
                },
                "any": {
                    "tasks": len(truncated_task_ids),
                    "strict_successes": sum(
                        _is_strict_success(item)
                        for task_id, item in by_task.items()
                        if task_id in truncated_task_ids
                    ),
                },
            },
        },
    }


def _reward_detail(trajectory):
    terminal = trajectory.get("terminal_result") or {}
    detail = terminal.get("reward_detail") or {}
    return detail if isinstance(detail, dict) else {}


def _is_strict_success(trajectory):
    terminal = trajectory.get("terminal_result") or {}
    detail = _reward_detail(trajectory)
    return (
        trajectory.get("status") == "done"
        and trajectory.get("done") is True
        and terminal.get("done") is True
        and terminal.get("over") is True
        and all(detail.get(key) == 1 or detail.get(key) is True for key in REWARD_KEYS)
    )
