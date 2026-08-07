import unittest

from shopping_grpo.evaluation.statistics import (
    compare_repeated_runs,
    mcnemar_exact,
    paired_bootstrap_mean_delta,
    summarize_failure_profile,
    summarize_repeated_run,
)


def trajectory(task_id, attempt_index, success):
    reward_type = "gold_purchase" if success else "wrong_purchase"
    return {
        "task_id": task_id,
        "attempt_index": attempt_index,
        "status": "done",
        "done": True,
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


def repeated_rows(outcomes):
    return [
        trajectory(task_id, attempt_index, success)
        for task_id, attempts in outcomes.items()
        for attempt_index, success in enumerate(attempts)
    ]


class RepeatedEvaluationStatisticsTest(unittest.TestCase):
    def test_run_summary_reports_attempt_rate_and_both_pass_k_notions(self):
        rows = repeated_rows(
            {
                1: [True, False],
                2: [False, False],
                3: [True, True],
                4: [False, True],
            }
        )

        report = summarize_repeated_run(
            expected_task_ids=[1, 2, 3, 4],
            trajectories=rows,
            attempts_per_task=2,
        )

        self.assertEqual(report["strict_successes"], 4)
        self.assertEqual(report["strict_success_rate"], 0.5)
        self.assertEqual(report["empirical_pass_at_k"], 0.75)
        self.assertEqual(report["empirical_pass_power_k"], 0.25)
        self.assertEqual(report["attempt_coverage_rate"], 1.0)
        self.assertEqual(report["missing_attempts"], [])
        self.assertEqual(
            [attempt["strict_success_rate"] for attempt in report["by_attempt_index"]],
            [0.5, 0.5],
        )
        interval = report["strict_success_rate_wilson_95"]
        self.assertLess(interval["low"], 0.5)
        self.assertGreater(interval["high"], 0.5)

    def test_missing_attempts_remain_in_the_fixed_denominator(self):
        report = summarize_repeated_run(
            expected_task_ids=[1, 2],
            trajectories=[trajectory(1, 0, True)],
            attempts_per_task=2,
        )

        self.assertEqual(report["completed_attempts"], 1)
        self.assertEqual(report["strict_success_rate"], 0.25)
        self.assertEqual(report["attempt_coverage_rate"], 0.25)
        self.assertEqual(len(report["missing_attempts"]), 3)

    def test_paired_report_uses_task_level_bootstrap_and_attempt_mcnemar(self):
        baseline = repeated_rows(
            {
                1: [True, False],
                2: [False, False],
                3: [True, True],
                4: [False, True],
            }
        )
        candidate = repeated_rows(
            {
                1: [True, True],
                2: [True, False],
                3: [True, True],
                4: [False, False],
            }
        )

        report = compare_repeated_runs(
            expected_task_ids=[1, 2, 3, 4],
            baseline_trajectories=baseline,
            candidate_trajectories=candidate,
            attempts_per_task=2,
            bootstrap_samples=1_000,
            seed=7,
        )

        paired = report["paired_task_delta"]
        self.assertEqual(paired["candidate_minus_baseline"], 0.125)
        self.assertEqual(paired["candidate_minus_baseline_percentage_points"], 12.5)
        self.assertEqual((paired["wins"], paired["ties"], paired["losses"]), (2, 1, 1))
        self.assertEqual(paired["bootstrap"]["seed"], 7)
        mcnemar = report["paired_attempt_test"]
        self.assertEqual(mcnemar["baseline_only_successes"], 1)
        self.assertEqual(mcnemar["candidate_only_successes"], 2)
        self.assertEqual(mcnemar["p_value"], 1.0)
        self.assertEqual(mcnemar["paired_completed_attempts"], 8)
        self.assertEqual(mcnemar["excluded_unpaired_attempts"], 0)

    def test_bootstrap_is_reproducible(self):
        first = paired_bootstrap_mean_delta([0.5, -0.5, 0.0], samples=250, seed=9)
        second = paired_bootstrap_mean_delta([0.5, -0.5, 0.0], samples=250, seed=9)

        self.assertEqual(first, second)

    def test_mcnemar_handles_no_discordant_attempts(self):
        self.assertEqual(mcnemar_exact(0, 0)["p_value"], 1.0)

    def test_duplicate_attempt_is_rejected(self):
        duplicate = trajectory(1, 0, True)
        with self.assertRaisesRegex(ValueError, "duplicate trajectory"):
            summarize_repeated_run(
                expected_task_ids=[1],
                trajectories=[duplicate, dict(duplicate)],
                attempts_per_task=1,
            )

    def test_mcnemar_excludes_attempts_missing_from_either_run(self):
        report = compare_repeated_runs(
            expected_task_ids=[1],
            baseline_trajectories=[trajectory(1, 0, True)],
            candidate_trajectories=[],
            attempts_per_task=1,
            bootstrap_samples=10,
        )

        paired = report["paired_attempt_test"]
        self.assertEqual(paired["paired_completed_attempts"], 0)
        self.assertEqual(paired["excluded_unpaired_attempts"], 1)
        self.assertEqual(paired["discordant_pairs"], 0)

    def test_failure_profile_keeps_reward_and_runtime_failures_separate(self):
        failed = trajectory(1, 0, False)
        failed["blocked_tool_calls"] = [{"reason": "asin_not_visible"}]
        failed["steps"] = [
            {"projection": {"critical_footer_preserved": False}}
        ]
        infrastructure = trajectory(2, 0, False)
        infrastructure["status"] = "error"
        infrastructure["terminal_result"] = {}
        infrastructure["error"] = {
            "type": "TimeoutError",
            "message": "model server timed out",
        }

        profile = summarize_failure_profile([failed, infrastructure])

        self.assertEqual(profile["strict_failures"], 2)
        self.assertEqual(profile["reward_type_counts"], {"missing": 1, "wrong_purchase": 1})
        self.assertEqual(profile["infrastructure_invalid_attempts"], 1)
        self.assertEqual(profile["guard_reason_counts"], {"asin_not_visible": 1})
        self.assertEqual(profile["critical_footer_failures"], 1)
        self.assertEqual(profile["loop_rate"], 0.0)

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
