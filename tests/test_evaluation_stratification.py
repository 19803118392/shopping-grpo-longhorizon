import unittest

from shopping_grpo.evaluation.reporting import render_markdown_report, render_overall_csv
from shopping_grpo.evaluation.statistics import compare_repeated_runs
from shopping_grpo.evaluation.stratification import (
    build_stratified_comparison,
    query_difficulty_features,
)


def trajectory(task_id, attempt_index, success, query, *, searches=1, steps=4):
    reward_type = "gold_purchase" if success else "repeat_loop"
    actions = [{"tool_name": "search_products"} for _ in range(searches)]
    actions.extend({"tool_name": "click"} for _ in range(max(steps - searches, 0)))
    return {
        "task_id": task_id,
        "attempt_index": attempt_index,
        "initial_result": {"instruction": query},
        "status": "done",
        "done": True,
        "steps": actions,
        "terminal_result": {
            "done": True,
            "over": True,
            "reward_detail": {
                "reward_version": "shopsimulator-reward-v3",
                "reward_type": reward_type,
                "reward_valid": True,
                "purchase_success": bool(success),
                "termination_reason": reward_type,
            },
        },
    }


class EvaluationStratificationTest(unittest.TestCase):
    def test_query_features_are_public_and_cover_requested_axes(self):
        features = query_difficulty_features(
            "想买耐用的黑色背包，容量30升，预算500元以内"
        )
        self.assertTrue(features["has_option_selection"])
        self.assertTrue(features["has_price_constraint"])
        self.assertIn("color", features["option_axes"])
        self.assertIn("capacity", features["option_axes"])
        self.assertEqual(features["constraint_count_bucket"], "4+")

    def test_strata_keep_static_and_model_conditional_buckets_separate(self):
        tasks = [
            {"task_id": 1, "length_bucket": "short"},
            {"task_id": 2, "length_bucket": "long"},
        ]
        queries = {
            1: "想买水杯",
            2: "想买黑色背包，容量30升，预算500元以内",
        }
        baseline = [
            trajectory(1, 0, True, queries[1], searches=1, steps=4),
            trajectory(2, 0, False, queries[2], searches=4, steps=24),
        ]
        candidate = [
            trajectory(1, 0, True, queries[1], searches=1, steps=5),
            trajectory(2, 0, True, queries[2], searches=2, steps=12),
        ]
        report = build_stratified_comparison(
            benchmark_tasks=tasks,
            baseline_trajectories=baseline,
            candidate_trajectories=candidate,
            attempts_per_task=1,
            bootstrap_samples=100,
            seed=7,
        )

        static = report["static_task_strata"]
        self.assertIn("4+", static["constraint_count"])
        self.assertIn("yes", static["option_selection"])
        self.assertIn("long", static["reference_length"])
        behavior = report["behavior_strata"]
        self.assertEqual(behavior["baseline"]["search_steps"]["4+"]["attempts"], 1)
        self.assertEqual(
            behavior["candidate"]["trajectory_length"]["medium11-20"]["attempts"],
            1,
        )

    def test_renderer_outputs_requested_headline_table(self):
        baseline = [trajectory(1, 0, False, "想买水杯")]
        candidate = [trajectory(1, 0, True, "想买水杯")]
        report = compare_repeated_runs(
            expected_task_ids=[1],
            baseline_trajectories=baseline,
            candidate_trajectories=candidate,
            attempts_per_task=1,
            bootstrap_samples=10,
        )
        report["labels"] = {"baseline": "SFT", "candidate": "GRPO"}
        markdown = render_markdown_report(report)
        csv_text = render_overall_csv(report)
        self.assertIn("Win/Tie/Loss", markdown)
        self.assertIn("SFT", markdown)
        self.assertIn("GRPO", csv_text)
        self.assertIn("loop_rate", csv_text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
