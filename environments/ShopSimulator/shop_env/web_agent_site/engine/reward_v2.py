"""Deterministic terminal reward for ShopSimulator Environment v2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
import unicodedata


REWARD_VERSION = "shopsimulator-reward-v2"
DEFAULT_REWARDS = {
    "gold_purchase": 1.0,
    "valid_alternative_purchase": 0.4,
    "graceful_stop": -0.1,
    "early_abstain": -0.25,
    "wrong_purchase": -0.4,
    "repeat_loop": -0.6,
    "max_steps": -0.7,
}


@dataclass(frozen=True)
class RewardResult:
    reward: float
    reward_type: str
    reward_valid: bool
    termination_reason: str
    target_asin_match: bool
    hard_gates: dict
    weighted_score: float
    evidence: dict

    def to_dict(self):
        payload = asdict(self)
        payload["reward_version"] = REWARD_VERSION
        return payload


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", "", text)


def _flatten(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        result = []
        for key in sorted(value, key=str):
            result.append(str(key))
            result.extend(_flatten(value[key]))
        return result
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return [str(value)]


def _contains(product_text: str, expected: object) -> bool:
    expected_text = _normalize(expected)
    return bool(expected_text) and expected_text in product_text


def _selected_values(options: object) -> list[str]:
    if isinstance(options, dict):
        return [str(value) for value in options.values()]
    return _flatten(options)


def _required_option_values(goal: dict) -> list[str]:
    raw = goal.get("goal_options") or []
    if isinstance(raw, dict):
        return [str(value) for value in raw.values()]
    return _flatten(raw)


def _option_gate(selected_options: object, goal: dict) -> dict:
    required = _required_option_values(goal)
    selected = _selected_values(selected_options)
    normalized_selected = [_normalize(value) for value in selected]
    missing = [
        value
        for value in required
        if not any(
            _normalize(value) == candidate
            or _normalize(value) in candidate
            or candidate in _normalize(value)
            for candidate in normalized_selected
            if candidate
        )
    ]
    return {
        "passed": not missing,
        "required": required,
        "selected": selected,
        "missing": missing,
        "verifiable": True,
    }


def _category_gate(product: dict, goal: dict) -> dict:
    product_parts = {
        _normalize(part)
        for part in str(product.get("category") or "").split("›")
        if _normalize(part)
    }
    goal_parts = [
        _normalize(part)
        for part in str(goal.get("category") or "").split("›")
        if _normalize(part)
    ]
    passed = bool(goal_parts) and bool(product_parts.intersection(goal_parts[-2:]))
    return {
        "passed": passed,
        "required": goal.get("category"),
        "actual": product.get("category"),
        "verifiable": bool(goal_parts and product_parts),
    }


def _price_gate(price: object, goal: dict) -> dict:
    upper = goal.get("price_upper")
    try:
        actual = float(price)
        limit = float(upper)
    except (TypeError, ValueError):
        return {
            "passed": False,
            "required_max": upper,
            "actual": price,
            "verifiable": False,
        }
    verifiable = math.isfinite(actual) and math.isfinite(limit) and limit > 0
    return {
        "passed": bool(verifiable and actual <= limit),
        "required_max": limit,
        "actual": actual,
        "verifiable": verifiable,
    }


def _declared_constraint_gates(product: dict, goal: dict) -> tuple[dict, bool]:
    """Evaluate explicit hard constraints; never infer missing declarations."""
    constraints = goal.get("hard_constraints")
    if not isinstance(constraints, dict):
        return {}, False
    fields = {
        "brand": product.get("brand") or product.get("shop_name"),
        "model": product.get("model") or product.get("title"),
        "core_functions": " ".join(
            _flatten(
                product.get("attribute")
                or product.get("Attributes")
                or product.get("small_description")
            )
        ),
        "key_specs": " ".join(
            _flatten(product.get("customization_options") or product.get("options"))
        ),
    }
    product_text = _normalize(
        " ".join(
            _flatten(
                [
                    product.get("title"),
                    product.get("brand"),
                    product.get("shop_name"),
                    product.get("category"),
                    product.get("attribute"),
                    product.get("Attributes"),
                    product.get("BulletPoints"),
                    product.get("customization_options"),
                    product.get("options"),
                    product.get("small_description"),
                    product.get("sub_title"),
                    product.get("full_description"),
                    product.get("Description"),
                ]
            )
        )
    )
    gates = {}
    for name in ("brand", "model", "core_functions", "key_specs"):
        required = constraints.get(name)
        if required in (None, "", [], {}):
            continue
        expected_values = _flatten(required)
        actual = fields[name]
        verifiable = bool(_normalize(actual))
        missing = [
            value for value in expected_values if not _contains(product_text, value)
        ]
        gates[name] = {
            "passed": verifiable and not missing,
            "required": expected_values,
            "actual": actual,
            "missing": missing,
            "verifiable": verifiable,
        }
    complete = constraints.get("complete") is True
    return gates, complete


def _weighted_preferences(product: dict, goal: dict) -> tuple[float, dict]:
    preferences = goal.get("weighted_preferences")
    if not isinstance(preferences, list) or not preferences:
        return 0.0, {"items": [], "weight_total": 0.0}
    product_text = _normalize(
        " ".join(
            _flatten(
                [
                    product.get("title"),
                    product.get("brand"),
                    product.get("category"),
                    product.get("attribute"),
                    product.get("customization_options"),
                    product.get("small_description"),
                ]
            )
        )
    )
    items = []
    earned = 0.0
    total = 0.0
    for preference in preferences:
        if not isinstance(preference, dict) or "value" not in preference:
            continue
        weight = float(preference.get("weight", 1.0))
        if not math.isfinite(weight) or weight <= 0:
            continue
        matched = _contains(product_text, preference["value"])
        earned += weight * int(matched)
        total += weight
        items.append(
            {
                "name": preference.get("name", "preference"),
                "value": preference["value"],
                "weight": weight,
                "matched": matched,
            }
        )
    return (earned / total if total else 0.0), {
        "items": items,
        "weight_total": total,
    }


def evaluate_purchase(
    product: dict,
    goal: dict,
    *,
    price: object,
    selected_options: object,
    rewards: dict[str, float] | None = None,
) -> RewardResult:
    """Score a purchase without treating missing metadata as success."""
    values = {**DEFAULT_REWARDS, **(rewards or {})}
    asin_match = str(product.get("asin")) == str(goal.get("asin"))
    gates = {
        "category": _category_gate(product, goal),
        "budget": _price_gate(price, goal),
        "key_options": _option_gate(selected_options, goal),
    }
    declared, constraints_complete = _declared_constraint_gates(product, goal)
    gates.update(declared)
    weighted_score, weighted_evidence = _weighted_preferences(product, goal)

    if asin_match:
        verifiable = all(gate["verifiable"] for gate in gates.values())
        passed = verifiable and all(gate["passed"] for gate in gates.values())
        reward_type = "gold_purchase" if passed else "wrong_purchase"
        reward = values[reward_type]
        valid = True
    else:
        verifiable = (
            constraints_complete
            and all(gate["verifiable"] for gate in gates.values())
        )
        if not verifiable:
            reward_type = "reward_unverifiable"
            reward = 0.0
            valid = False
        elif all(gate["passed"] for gate in gates.values()):
            reward_type = "valid_alternative_purchase"
            reward = values[reward_type]
            valid = True
        else:
            reward_type = "wrong_purchase"
            reward = values[reward_type]
            valid = True

    return RewardResult(
        reward=float(reward),
        reward_type=reward_type,
        reward_valid=valid,
        termination_reason=reward_type,
        target_asin_match=asin_match,
        hard_gates=gates,
        weighted_score=weighted_score,
        evidence={
            "constraints_complete": constraints_complete,
            "weighted_preferences": weighted_evidence,
        },
    )


def evaluate_abstain(
    *,
    distinct_normalized_queries: int,
    opened_asins: int,
    rewards: dict[str, float] | None = None,
) -> RewardResult:
    values = {**DEFAULT_REWARDS, **(rewards or {})}
    eligible = (
        int(distinct_normalized_queries) >= 2
        or (int(distinct_normalized_queries) >= 1 and int(opened_asins) >= 1)
    )
    reward_type = "graceful_stop" if eligible else "early_abstain"
    return RewardResult(
        reward=float(values[reward_type]),
        reward_type=reward_type,
        reward_valid=True,
        termination_reason=reward_type,
        target_asin_match=False,
        hard_gates={},
        weighted_score=0.0,
        evidence={
            "eligible": eligible,
            "distinct_normalized_queries": int(distinct_normalized_queries),
            "opened_asins": int(opened_asins),
        },
    )


def fixed_termination(reason: str, rewards: dict[str, float] | None = None) -> RewardResult:
    values = {**DEFAULT_REWARDS, **(rewards or {})}
    if reason not in {"repeat_loop", "max_steps"}:
        raise ValueError(f"unsupported fixed termination reason: {reason}")
    return RewardResult(
        reward=float(values[reason]),
        reward_type=reason,
        reward_valid=True,
        termination_reason=reason,
        target_asin_match=False,
        hard_gates={},
        weighted_score=0.0,
        evidence={},
    )
