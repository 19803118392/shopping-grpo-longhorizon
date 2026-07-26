import unittest

from web_agent_site.engine.goal_v2 import (
    deterministic_price_upper,
    explicit_budget_from_instruction,
)


class GoalV2Test(unittest.TestCase):
    def test_explicit_budget_is_used_as_written(self):
        self.assertEqual(
            explicit_budget_from_instruction("预算在1000元以下。"),
            1000.0,
        )
        self.assertEqual(
            explicit_budget_from_instruction("价格不超过 2199 元"),
            2199.0,
        )
        self.assertEqual(
            explicit_budget_from_instruction("预算在1万元左右"),
            11000.0,
        )
        self.assertEqual(
            explicit_budget_from_instruction("预算1万2以内"),
            12000.0,
        )

    def test_missing_budget_has_deterministic_fallback(self):
        first = deterministic_price_upper("123", "买一个枕头", 999)
        second = deterministic_price_upper("123", "买一个枕头", 999)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
