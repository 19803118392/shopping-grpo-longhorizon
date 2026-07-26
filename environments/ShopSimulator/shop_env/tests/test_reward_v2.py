import unittest

from web_agent_site.engine.reward_v2 import (
    evaluate_abstain,
    evaluate_purchase,
    fixed_termination,
)


GOAL = {
    "asin": "111111111111",
    "category": "家电›洗地机",
    "price_upper": 2200,
    "goal_options": ["白色"],
    "hard_constraints": {
        "complete": True,
        "brand": ["石头"],
        "model": ["A20"],
        "core_functions": ["洗地"],
    },
    "weighted_preferences": [
        {"name": "color", "value": "白色", "weight": 1},
        {"name": "appearance", "value": "轻巧", "weight": 0.5},
    ],
}


def product(asin="111111111111", *, brand="石头", title="石头 A20 智能洗地机"):
    return {
        "asin": asin,
        "title": title,
        "brand": brand,
        "category": "家电›洗地机",
        "attribute": ["洗地", "白色"],
    }


class RewardV2Test(unittest.TestCase):
    def test_gold_purchase_is_highest(self):
        result = evaluate_purchase(
            product(), GOAL, price=2199, selected_options={"颜色": "白色"}
        )
        self.assertEqual(result.reward_type, "gold_purchase")
        self.assertEqual(result.reward, 1.0)

    def test_valid_alternative_requires_all_hard_gates(self):
        result = evaluate_purchase(
            product("222222222222"),
            GOAL,
            price=2000,
            selected_options={"颜色": "白色"},
        )
        self.assertEqual(result.reward_type, "valid_alternative_purchase")
        self.assertEqual(result.reward, 0.4)

    def test_budget_or_brand_violation_is_wrong_purchase(self):
        over_budget = evaluate_purchase(
            product("222222222222"),
            GOAL,
            price=3000,
            selected_options={"颜色": "白色"},
        )
        wrong_brand = evaluate_purchase(
            product("222222222222", brand="苏泊尔", title="苏泊尔 A20 洗地机"),
            GOAL,
            price=2000,
            selected_options={"颜色": "白色"},
        )
        self.assertEqual(over_budget.reward_type, "wrong_purchase")
        self.assertEqual(wrong_brand.reward_type, "wrong_purchase")
        self.assertEqual(over_budget.reward, -0.4)

    def test_non_target_missing_constraint_contract_is_unverifiable(self):
        goal = {key: value for key, value in GOAL.items() if key != "hard_constraints"}
        result = evaluate_purchase(
            product("222222222222"),
            goal,
            price=2000,
            selected_options={"颜色": "白色"},
        )
        self.assertEqual(result.reward_type, "reward_unverifiable")
        self.assertFalse(result.reward_valid)
        self.assertEqual(result.reward, 0.0)

    def test_abstain_gate(self):
        early = evaluate_abstain(distinct_normalized_queries=1, opened_asins=0)
        queried = evaluate_abstain(distinct_normalized_queries=2, opened_asins=0)
        inspected = evaluate_abstain(distinct_normalized_queries=1, opened_asins=1)
        self.assertEqual(early.reward, -0.25)
        self.assertEqual(queried.reward, -0.1)
        self.assertEqual(inspected.reward_type, "graceful_stop")

    def test_loop_is_worse_than_wrong_purchase_and_max_steps_is_worst(self):
        wrong = evaluate_purchase(
            product("222222222222"),
            GOAL,
            price=3000,
            selected_options={"颜色": "白色"},
        )
        repeat = fixed_termination("repeat_loop")
        max_steps = fixed_termination("max_steps")
        self.assertGreater(wrong.reward, repeat.reward)
        self.assertGreater(repeat.reward, max_steps.reward)


if __name__ == "__main__":
    unittest.main()
