import copy
import unittest

from scripts.verify_seed_replay import verify_seed_replay


def row(attempt, *, content="same", trajectory_id="random"):
    return {
        "trajectory_id": trajectory_id,
        "created_at": "different",
        "task_id": 1,
        "attempt_index": attempt,
        "actor_sampling": {"attempt_seed": 100 + attempt},
        "messages": [
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": trajectory_id,
                        "type": "function",
                        "function": {
                            "name": "search_products",
                            "arguments": '{"query":"mug"}',
                        },
                    }
                ],
            }
        ],
        "steps": [{"tool_name": "search_products", "parameters": {"query": "mug"}}],
    }


class SeedReplayTest(unittest.TestCase):
    def test_ignores_run_ids_but_compares_model_trace_and_actions(self):
        first = [row(0, trajectory_id="a"), row(1, trajectory_id="b")]
        second = copy.deepcopy(first)
        second[0]["trajectory_id"] = "c"
        second[0]["created_at"] = "later"
        second[0]["messages"][0]["tool_calls"][0]["id"] = "server-random"
        report = verify_seed_replay(first, second, expected_tasks=1, attempts_per_task=2)
        self.assertTrue(report["exact_model_and_action_replay"])
        self.assertEqual(report["pairing_interpretation"], "common_random_numbers")

        second[1]["messages"][0]["content"] = "changed"
        report = verify_seed_replay(first, second, expected_tasks=1, attempts_per_task=2)
        self.assertFalse(report["exact_model_and_action_replay"])
        self.assertEqual(report["pairing_interpretation"], "task_paired_only")


if __name__ == "__main__":
    unittest.main()
