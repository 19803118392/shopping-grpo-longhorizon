"""Reward v3 runtime validation and public diagnostic tests."""

import unittest

from shopping_grpo.training.grpo.adapter.runtime import (
    ENVIRONMENT_REWARD_V3,
    OPTIMIZATION_REWARD_V4,
    constraint_complete_purchase_v4,
    make_runtime_state,
    optimization_reward_v4,
    record_action_attempt,
    reward_breakdown,
    terminal_reward,
    validate_reward,
)


def gate(status: str = "pass") -> dict:
    return {
        "status": status,
        "passed": status == "pass",
        "verifiable": status != "unverifiable",
        "comparator": "equals",
        "source_field": "public_observation",
    }


def reward_detail(
    *,
    reward_type: str = "gold_purchase",
    reward_valid: bool = True,
    terminal_utility: float = 1.0,
    weighted_score: float = 1.0,
    evidence_coverage: float = 1.0,
    category_status: str = "pass",
    budget_status: str = "pass",
    dimension_scores: dict | None = None,
) -> dict:
    return {
        "reward_version": "shopsimulator-reward-v3",
        "reward_type": reward_type,
        "termination_reason": reward_type,
        "reward_valid": reward_valid,
        "target_asin_match": reward_type == "gold_purchase",
        "hard_gates": {
            "category": gate(category_status),
            "budget": gate(budget_status),
        },
        "weighted_score": weighted_score,
        "evidence_coverage": evidence_coverage,
        "dimension_scores": dimension_scores
        or {
            "brand": 1.0,
            "model": 1.0,
            "core_functions": 1.0,
            "key_options": 1.0,
        },
        "terminal_utility": terminal_utility,
        "purchase_success": reward_type
        in {"gold_purchase", "valid_alternative_purchase"},
        "sampling_invalid": not reward_valid,
    }


def terminal_state(*, detail: dict | None = None, native_reward: float = 1.0) -> dict:
    validated = validate_reward(detail or reward_detail())
    state = make_runtime_state(task_id=1, max_steps=35)
    state.update(
        {
            "done": True,
            "terminal_result": {"done": True, "over": True},
            "final_reward": native_reward,
            "reward_version": validated["reward_version"],
            "reward_type": validated["reward_type"],
            "reward_valid": validated["reward_valid"],
            "reward_unverifiable": not validated["reward_valid"],
            "reward_public": validated,
        }
    )
    return state


class ShoppingRewardTest(unittest.TestCase):
    def test_gold_purchase_preserves_native_reward_and_public_diagnostics(self):
        result = reward_breakdown(terminal_state())

        self.assertEqual(result["full"], 1.0)
        self.assertEqual(result["strict"], 1.0)
        self.assertEqual(result["terminal_utility"], 1.0)
        self.assertEqual(result["purchase_success"], 1.0)
        self.assertEqual(result["r_type"], 1.0)
        self.assertEqual(result["r_price"], 1.0)
        self.assertEqual(result["r_att"], 1.0)
        self.assertEqual(result["r_option"], 1.0)
        self.assertEqual(result["total"], 1.0)
        self.assertFalse(result["sampling_invalid"])

    def test_valid_alternative_is_success_but_not_strict_gold(self):
        detail = reward_detail(
            reward_type="valid_alternative_purchase",
            terminal_utility=0.55,
            weighted_score=1.0,
            dimension_scores={
                "brand": 0.5,
                "model": 0.75,
                "core_functions": 1.0,
                "key_options": 0.5,
            },
        )
        result = reward_breakdown(terminal_state(detail=detail, native_reward=0.55))

        self.assertEqual(result["full"], 0.0)
        self.assertEqual(result["strict"], 0.0)
        self.assertEqual(result["purchase_success"], 1.0)
        self.assertEqual(result["option_score"], 0.5)
        self.assertEqual(result["total"], 0.55)
        self.assertEqual(result["optimization_reward_v4"], 1.0)
        self.assertEqual(result["constraint_complete_purchase_v4"], 1.0)

    def test_v4_removes_only_the_target_asin_bonus(self):
        gold = validate_reward(reward_detail())
        alternative_raw = reward_detail(
            reward_type="valid_alternative_purchase",
            terminal_utility=0.55,
        )
        alternative_raw["target_asin_match"] = False
        alternative = validate_reward(alternative_raw)

        self.assertEqual(optimization_reward_v4(gold), 1.0)
        self.assertEqual(optimization_reward_v4(alternative), 1.0)
        self.assertTrue(constraint_complete_purchase_v4(gold))
        self.assertTrue(constraint_complete_purchase_v4(alternative))

    def test_v4_partial_hard_failure_and_termination_values(self):
        partial = validate_reward(
            reward_detail(
                reward_type="partial_alternative_purchase",
                terminal_utility=0.03,
                weighted_score=0.8,
                evidence_coverage=0.75,
            )
        )
        wrong = validate_reward(
            reward_detail(
                reward_type="wrong_purchase",
                terminal_utility=-0.85,
                category_status="fail",
            )
        )
        stopped = validate_reward(
            reward_detail(
                reward_type="graceful_stop",
                terminal_utility=-0.15,
            )
        )

        self.assertAlmostEqual(optimization_reward_v4(partial), 0.03)
        self.assertEqual(optimization_reward_v4(wrong), -0.85)
        self.assertEqual(optimization_reward_v4(stopped), -0.15)

    def test_training_profile_selects_v3_or_v4_without_changing_either_metric(self):
        detail = reward_detail(
            reward_type="valid_alternative_purchase",
            terminal_utility=0.55,
        )
        state = terminal_state(detail=detail, native_reward=0.55)
        v3 = reward_breakdown(state, mode=ENVIRONMENT_REWARD_V3)
        v4 = reward_breakdown(state, mode=OPTIMIZATION_REWARD_V4)

        self.assertEqual(terminal_reward(state, ENVIRONMENT_REWARD_V3), 0.55)
        self.assertEqual(terminal_reward(state, OPTIMIZATION_REWARD_V4), 1.0)
        self.assertEqual(v3["environment_reward_v3"], v4["environment_reward_v3"])
        self.assertEqual(v3["optimization_reward_v4"], v4["optimization_reward_v4"])
        self.assertEqual(v3["optimization_reward_profile"], ENVIRONMENT_REWARD_V3)
        self.assertEqual(v4["optimization_reward_profile"], OPTIMIZATION_REWARD_V4)

    def test_reward_unverifiable_never_creates_learning_signal(self):
        detail = reward_detail(
            reward_type="reward_unverifiable",
            reward_valid=False,
            terminal_utility=0.0,
        )
        result = reward_breakdown(terminal_state(detail=detail, native_reward=0.0))

        self.assertTrue(result["sampling_invalid"])
        self.assertTrue(result["reward_unverifiable"])
        self.assertFalse(result["infrastructure_invalid"])
        self.assertEqual(result["total"], 0.0)

    def test_unfinished_state_is_infrastructure_invalid(self):
        state = make_runtime_state(task_id=1, max_steps=35)
        state["termination_reason"] = "assistant_finished_without_environment_done"
        state["error"] = state["termination_reason"]

        result = reward_breakdown(state)

        self.assertTrue(result["sampling_invalid"])
        self.assertTrue(result["infrastructure_invalid"])
        self.assertEqual(result["total"], 0.0)

    def test_validate_reward_rejects_contract_mismatches(self):
        cases = []

        wrong_version = reward_detail()
        wrong_version["reward_version"] = "shopsimulator-reward-v2"
        cases.append((wrong_version, "reward_version"))

        unknown_type = reward_detail()
        unknown_type["reward_type"] = unknown_type["termination_reason"] = "mystery"
        cases.append((unknown_type, "unknown Reward v3"))

        wrong_termination = reward_detail()
        wrong_termination["termination_reason"] = "max_steps"
        cases.append((wrong_termination, "termination_reason"))

        non_boolean_validity = reward_detail()
        non_boolean_validity["reward_valid"] = 1
        cases.append((non_boolean_validity, "reward_valid"))

        inconsistent_sampling = reward_detail()
        inconsistent_sampling["sampling_invalid"] = True
        cases.append((inconsistent_sampling, "sampling_invalid"))

        missing_gate = reward_detail()
        missing_gate["hard_gates"].pop("budget")
        cases.append((missing_gate, "missing required hard gate"))

        for raw, pattern in cases:
            with self.subTest(pattern=pattern), self.assertRaisesRegex(
                ValueError, pattern
            ):
                validate_reward(raw)

    def test_validate_reward_rejects_invalid_public_scores_and_gates(self):
        invalid_values = [
            ("terminal_utility", float("nan"), "finite"),
            ("weighted_score", 1.1, r"\[0, 1\]"),
            ("evidence_coverage", -0.1, r"\[0, 1\]"),
        ]
        for field, value, pattern in invalid_values:
            raw = reward_detail()
            raw[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, pattern):
                validate_reward(raw)

        raw = reward_detail()
        raw["dimension_scores"]["brand"] = float("inf")
        with self.assertRaisesRegex(ValueError, "dimension score brand"):
            validate_reward(raw)

        raw = reward_detail()
        raw["hard_gates"]["category"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "inconsistent passed"):
            validate_reward(raw)

    def test_same_action_on_same_page_within_three_attempts_is_repeated(self):
        state = make_runtime_state(task_id=1, max_steps=35)

        record_action_attempt(state, "search_products", {"query": "mug"}, "search page")
        record_action_attempt(state, "open_product", {"asin": "123"}, "search page")
        record_action_attempt(state, "search_products", {"query": "mug"}, "search page")

        self.assertEqual(state["action_attempt_count"], 3)
        self.assertEqual(state["repeat_action_count"], 1)
        self.assertAlmostEqual(reward_breakdown(state)["repeat_action_rate"], 1 / 3)

    def test_different_parameters_or_page_are_not_repeated(self):
        state = make_runtime_state(task_id=1, max_steps=35)

        record_action_attempt(state, "search_products", {"query": "mug"}, "page 1")
        record_action_attempt(state, "search_products", {"query": "cup"}, "page 1")
        record_action_attempt(state, "search_products", {"query": "mug"}, "page 2")

        self.assertEqual(state["repeat_action_count"], 0)

    def test_think_is_not_an_environment_action_attempt(self):
        state = make_runtime_state(task_id=1, max_steps=35)

        record_action_attempt(state, "think", {"note": "plan"}, "page")

        self.assertEqual(state["action_attempt_count"], 0)
        self.assertEqual(state["recent_action_signatures"], [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
