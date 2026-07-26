import unittest
from unittest.mock import patch

from web_agent_site.engine.goal import get_existed_goals
from web_agent_site.engine.goal_v2 import (
    CONSTRAINT_CONTRACT_VERSION,
    compile_task_constraint_contract,
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
            explicit_budget_from_instruction("价格在70元左右"),
            77.0,
        )
        self.assertEqual(
            explicit_budget_from_instruction("价格在130-140元之间"),
            140.0,
        )
        self.assertEqual(
            explicit_budget_from_instruction("价格30元到40元之间"),
            40.0,
        )
        self.assertIsNone(
            explicit_budget_from_instruction("预算4k+"),
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

    def test_task_annotations_compile_to_complete_constraint_contract(self):
        contract = compile_task_constraint_contract(
            {
                "instruction": "买一台支持热洗的洗地机",
                "attributes": ["智能", "热洗", "智能"],
                "instruction_options": ["白色"],
                # This target-only field must not become a constraint.
                "hidden_target_brand": "石头",
            }
        )
        hard = contract["hard_constraints"]
        self.assertTrue(hard["complete"])
        self.assertEqual(hard["contract_version"], CONSTRAINT_CONTRACT_VERSION)
        self.assertEqual(hard["core_functions"], ["智能", "热洗"])
        self.assertEqual(hard["brand"], [])
        self.assertEqual(hard["model"], [])
        self.assertEqual(hard["key_specs"], [])
        self.assertEqual(hard["annotated_option_count"], 1)
        self.assertEqual(contract["weighted_preferences"], [])

    def test_missing_annotation_schema_is_fail_closed(self):
        contract = compile_task_constraint_contract(
            {
                "instruction": "买一台洗地机",
                "attributes": ["洗地"],
            }
        )
        self.assertFalse(contract["hard_constraints"]["complete"])

    def test_environment_v2_goal_uses_current_instruction_annotations(self):
        item = {
            "asin": "123456789012",
            "category": "家电›洗地机",
            "query": "洗地机",
            "title": "测试洗地机",
            # This legacy field intentionally differs from the second task.
            "instruction_attributes": ["第一条要求"],
            "user_persona": {},
            "reason_key": "",
            "instructions": [
                {
                    "instruction": "第一条任务",
                    "instruction_simple": "第一条",
                    "instruction_options": [],
                    "attributes": ["第一条要求"],
                },
                {
                    "instruction": "第二条任务",
                    "instruction_simple": "第二条",
                    "instruction_options": ["白色"],
                    "attributes": ["第二条要求"],
                },
            ],
        }
        with (
            patch.dict(
                "os.environ",
                {"SHOP_ENVIRONMENT_VERSION": "shopsimulator-environment-v2"},
            ),
            patch("web_agent_site.engine.goal.print"),
        ):
            goals = get_existed_goals(
                [item],
                {"123456789012": 1000},
            )
        self.assertEqual(len(goals), 2)
        self.assertEqual(goals[1]["attributes"], ["第二条要求"])
        self.assertEqual(
            goals[1]["hard_constraints"]["core_functions"],
            ["第二条要求"],
        )
        self.assertTrue(goals[1]["hard_constraints"]["complete"])


if __name__ == "__main__":
    unittest.main()
