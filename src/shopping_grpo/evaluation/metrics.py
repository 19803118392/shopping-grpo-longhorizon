"""Deterministic trajectory metrics; no LLM inference is allowed here."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import json
import math
import re

from shopping_grpo.evaluation.contracts import CONTRACT_VERSION
from shopping_grpo.evaluation.trajectory import NORMALIZED_TRAJECTORY_VERSION


DETERMINISTIC_METRICS_VERSION = "shopping-deterministic-metrics-v1"
REWARD_V3 = "shopsimulator-reward-v3"
_ASIN = re.compile(r"(?<!\d)\d{11,12}(?!\d)")
_INFRASTRUCTURE_ERROR_TYPES = {
    "ContextBudgetError",
    "RemoteDisconnected",
    "ShopHttpError",
    "TimeoutError",
    "URLError",
}


def _canonical_parameters(parameters: object) -> dict:
    if not isinstance(parameters, Mapping):
        return {}
    return {
        str(key): value
        for key, value in parameters.items()
        if str(key) != "note"
    }


def _action_signature(event: Mapping) -> str:
    env_action = event.get("env_action")
    if isinstance(env_action, str) and env_action.strip():
        return f"env:{' '.join(env_action.split()).casefold()}"
    payload = [
        str(event.get("tool_name") or "").casefold(),
        _canonical_parameters(event.get("parameters")),
    ]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _duplicate_counts(signatures: list[str]) -> tuple[int, int]:
    seen = set()
    duplicate = 0
    consecutive = 0
    previous = None
    for signature in signatures:
        if signature in seen:
            duplicate += 1
        seen.add(signature)
        if previous is not None and signature == previous:
            consecutive += 1
        previous = signature
    return duplicate, consecutive


def _input_token_values(context: Mapping) -> list[int]:
    result = []
    turns = context.get("turn_tokens")
    if not isinstance(turns, list):
        return result
    for turn in turns:
        if not isinstance(turn, Mapping):
            continue
        value = turn.get("input_tokens")
        if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
            result.append(int(value))
    return result


def _error_type(value: object) -> str:
    if isinstance(value, Mapping):
        return str(value.get("type") or "")
    return ""


def _strict_success(normalized: Mapping, reward_detail: Mapping) -> bool:
    terminal = normalized.get("terminal")
    terminal = terminal if isinstance(terminal, Mapping) else {}
    return (
        reward_detail.get("reward_version") == REWARD_V3
        and normalized.get("status") == "done"
        and normalized.get("done") is True
        and terminal.get("done") is True
        and terminal.get("over") is True
        and reward_detail.get("reward_type") == "gold_purchase"
        and reward_detail.get("reward_valid") is True
        and reward_detail.get("purchase_success") is True
        and reward_detail.get("termination_reason") == "gold_purchase"
    )


def compute_deterministic_metrics(normalized: object) -> dict:
    """Compute one trajectory's code-owned metrics from normalized events."""

    if not isinstance(normalized, Mapping):
        raise TypeError("normalized trajectory must be an object")
    if normalized.get("schema_version") != NORMALIZED_TRAJECTORY_VERSION:
        raise ValueError(
            "normalized trajectory has an unsupported schema_version"
        )
    events = normalized.get("events")
    events = events if isinstance(events, list) else []
    attempts = [
        event for event in events if isinstance(event, Mapping)
    ]
    executed = [
        event for event in attempts if event.get("event_type") == "tool_step"
    ]
    guards = [
        event
        for event in attempts
        if event.get("event_type") == "guard_rejection"
    ]
    tool_counts = Counter(
        str(event.get("tool_name") or "unknown") for event in executed
    )
    action_signatures = [_action_signature(event) for event in attempts]
    duplicate_actions, consecutive_actions = _duplicate_counts(
        action_signatures
    )

    search_events = [
        event for event in executed if event.get("tool_name") == "search_products"
    ]
    search_queries = [
        " ".join(
            str(
                (event.get("parameters") or {}).get("query", "")
                if isinstance(event.get("parameters"), Mapping)
                else ""
            ).split()
        ).casefold()
        for event in search_events
    ]
    duplicate_searches, consecutive_searches = _duplicate_counts(search_queries)

    visible_asins = set()
    for event in search_events:
        visible_asins.update(
            _ASIN.findall(str(event.get("actor_visible_observation") or ""))
        )
    opened_asins = {
        str((event.get("parameters") or {}).get("asin"))
        for event in executed
        if event.get("tool_name") == "open_product"
        and isinstance(event.get("parameters"), Mapping)
        and (event.get("parameters") or {}).get("asin") is not None
    }

    projections = [
        event.get("observation_projection")
        for event in executed
        if isinstance(event.get("observation_projection"), Mapping)
        and event.get("observation_projection")
    ]
    input_tokens = _input_token_values(normalized.get("context") or {})
    guard_reasons = Counter(
        str(event.get("guard_reason") or "unknown") for event in guards
    )
    malformed_calls = sum(
        bool(event.get("tool_call_parse_error")) for event in attempts
    )
    schema_rejections = sum(
        str(event.get("guard_reason") or "").startswith("schema_")
        for event in guards
    )
    step_errors = sum(bool(event.get("step_error")) for event in executed)

    terminal = normalized.get("terminal")
    terminal = terminal if isinstance(terminal, Mapping) else {}
    reward_detail = terminal.get("reward_detail")
    reward_detail = (
        reward_detail if isinstance(reward_detail, Mapping) else {}
    )
    reward_type = str(reward_detail.get("reward_type") or "unknown")
    termination_reason = str(
        reward_detail.get("termination_reason")
        or terminal.get("termination_reason")
        or normalized.get("status")
        or "unknown"
    )

    errors = normalized.get("errors")
    errors = errors if isinstance(errors, Mapping) else {}
    trajectory_error = errors.get("trajectory_error")
    release_error = errors.get("release_error")
    trajectory_error_type = _error_type(trajectory_error)
    release_error_type = _error_type(release_error)
    infrastructure_invalid = bool(
        normalized.get("infrastructure_invalid")
        or release_error
        or trajectory_error_type in _INFRASTRUCTURE_ERROR_TYPES
        or release_error_type in _INFRASTRUCTURE_ERROR_TYPES
    )
    contract_issues = []
    if normalized.get("status") == "done":
        if normalized.get("done") is not True:
            contract_issues.append("done_status_without_trajectory_done")
        if terminal.get("done") is not True:
            contract_issues.append("done_status_without_terminal_done")
        if terminal.get("over") is not True:
            contract_issues.append("done_status_without_terminal_over")
        if not reward_detail:
            contract_issues.append("done_status_without_reward_detail")
    if reward_detail and reward_detail.get("reward_version") not in {
        None,
        REWARD_V3,
    }:
        contract_issues.append("unexpected_reward_version")
    normalization = normalized.get("normalization")
    if isinstance(normalization, Mapping) and normalization.get("warnings"):
        contract_issues.append("normalization_warnings")

    context = normalized.get("context")
    context = context if isinstance(context, Mapping) else {}
    return {
        "schema_version": DETERMINISTIC_METRICS_VERSION,
        "evaluation_contract": CONTRACT_VERSION,
        "trajectory_id": normalized.get("trajectory_id"),
        "task_id": normalized.get("task_id"),
        "reward_and_outcome": {
            "reward_version": reward_detail.get("reward_version"),
            "reward_type": reward_type,
            "reward_valid": reward_detail.get("reward_valid"),
            "final_reward": float(normalized.get("final_reward", 0.0) or 0.0),
            "terminal_utility": float(
                reward_detail.get(
                    "terminal_utility",
                    normalized.get("final_reward", 0.0),
                )
                or 0.0
            ),
            "weighted_score": float(
                reward_detail.get("weighted_score", 0.0) or 0.0
            ),
            "purchase_success": reward_detail.get("purchase_success") is True,
            "strict_gold_success": _strict_success(normalized, reward_detail),
            "done": normalized.get("done") is True,
            "terminal_done": terminal.get("done") is True,
            "terminal_over": terminal.get("over") is True,
            "termination_reason": termination_reason,
            "max_steps_termination": (
                reward_type == "max_steps"
                or termination_reason == "max_steps"
                or normalized.get("status") == "max_steps"
            ),
            "repeat_loop_termination": (
                reward_type == "repeat_loop"
                or termination_reason == "repeat_loop"
            ),
        },
        "actions_and_efficiency": {
            "executed_tool_steps": len(executed),
            "action_attempts": len(attempts),
            "tool_counts": dict(sorted(tool_counts.items())),
            "search_count": tool_counts.get("search_products", 0),
            "open_product_count": tool_counts.get("open_product", 0),
            "information_page_count": sum(
                tool_counts.get(name, 0)
                for name in (
                    "view_attributes",
                    "view_description",
                    "view_features",
                    "view_reviews",
                )
            ),
            "select_option_count": tool_counts.get("select_option", 0),
            "buy_count": tool_counts.get("buy_now", 0),
            "finish_without_purchase_count": tool_counts.get(
                "finish_without_purchase", 0
            ),
            "visible_search_candidate_count": len(visible_asins),
            "opened_candidate_count": len(opened_asins),
        },
        "repetition": {
            "duplicate_search_query_count": duplicate_searches,
            "consecutive_duplicate_search_count": consecutive_searches,
            "duplicate_canonical_action_count": duplicate_actions,
            "consecutive_duplicate_action_count": consecutive_actions,
            "environment_repeat_loop": (
                reward_type == "repeat_loop"
                or termination_reason == "repeat_loop"
            ),
        },
        "legality": {
            "guard_rejection_count": len(guards),
            "guard_reason_counts": dict(sorted(guard_reasons.items())),
            "malformed_tool_call_count": malformed_calls,
            "schema_rejection_count": schema_rejections,
            "step_error_count": step_errors,
            "invalid_action_limit": normalized.get("status")
            == "invalid_action_limit",
            "tool_call_truncation_count": len(
                context.get("tool_call_truncations") or []
            ),
        },
        "context": {
            "projection_count": len(projections),
            "truncated_observation_count": sum(
                bool(projection.get("truncated")) for projection in projections
            ),
            "any_observation_truncated": any(
                bool(projection.get("truncated")) for projection in projections
            ),
            "raw_observation_tokens": sum(
                int(projection.get("raw_tokens", 0) or 0)
                for projection in projections
            ),
            "visible_observation_tokens": sum(
                int(projection.get("visible_tokens", 0) or 0)
                for projection in projections
            ),
            "critical_footer_failures": sum(
                not bool(projection.get("critical_footer_preserved", False))
                for projection in projections
            ),
            "context_compaction_count": len(context.get("compactions") or []),
            "input_token_turn_count": len(input_tokens),
            "input_tokens_sum": sum(input_tokens),
            "max_input_tokens": max(input_tokens, default=0),
            "completion_tokens": None,
            "total_tokens": None,
            "cached_tokens": None,
        },
        "timing": {
            "trajectory_duration_seconds": None,
            "model_latency_seconds": None,
            "tool_latency_seconds": None,
        },
        "validity": {
            "infrastructure_invalid": infrastructure_invalid,
            "trajectory_error_type": trajectory_error_type or None,
            "release_error_type": release_error_type or None,
            "context_hard_limit": trajectory_error_type
            == "ContextBudgetError",
            "trajectory_contract_valid": not contract_issues,
            "trajectory_contract_issues": contract_issues,
            "judge_eligible": not infrastructure_invalid,
        },
    }
