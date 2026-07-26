from pathlib import Path
import sys
import unittest

SHOP_ENV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHOP_ENV))

from scripts.audit_environment_v2 import iter_tasks


class EnvironmentV2AuditTest(unittest.TestCase):
    def test_task_ids_follow_environment_goal_order(self):
        products = [
            {
                "asin": "1",
                "instructions": [
                    {"instruction": "one", "attributes": ["a"]},
                    {"instruction": "two", "attributes": ["b"]},
                ],
            },
            {
                "asin": "2",
                "instructions": [{"instruction": "skip", "attributes": []}],
            },
            {
                "asin": "3",
                "instructions": [{"instruction": "three", "attributes": ["c"]}],
            },
        ]
        rows = list(iter_tasks(products))
        self.assertEqual(
            [(task_id, product["asin"]) for task_id, product, _ in rows],
            [(0, "1"), (1, "1"), (2, "3")],
        )


if __name__ == "__main__":
    unittest.main()
