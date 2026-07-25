"""Unit tests for the project-side reward-group filter."""

import unittest

from shopping_grpo.verl_dynamic_sampling import (
    aggregate_shopping_metrics,
    extract_shopping_group_signals,
    select_reward_varying_groups,
)


class RewardGroupSelectionTest(unittest.TestCase):
    def test_all_zero_group_is_dropped(self):
        indices, stats = select_reward_varying_groups(["a"] * 4, [0, 0, 0, 0])
        self.assertEqual(indices, [])
        self.assertEqual(stats["dropped_uids"], ("a",))
        self.assertEqual(stats["all_zero_semantic_group_count"], 1)
        self.assertEqual(stats["all_full_success_group_count"], 0)

    def test_all_one_group_is_dropped(self):
        indices, stats = select_reward_varying_groups(
            ["a"] * 4,
            [1, 1, 1, 1],
            semantic_rewards=[1.7, 1.7, 1.7, 1.7],
        )
        self.assertEqual(indices, [])
        self.assertEqual(stats["kept_group_count"], 0)
        self.assertEqual(stats["all_full_success_group_count"], 1)

    def test_fractional_reward_variance_is_kept(self):
        rewards = [2 / 7, 4 / 7, 2 / 7, 2 / 7]
        indices, stats = select_reward_varying_groups(["a"] * 4, rewards)
        self.assertEqual(indices, [0, 1, 2, 3])
        self.assertEqual(stats["kept_uids"], ("a",))

    def test_mixed_uids_preserve_trajectory_indices(self):
        uids = ["a", "b", "a", "b", "a", "b", "a", "b"]
        rewards = [0, 2 / 7, 0, 4 / 7, 0, 2 / 7, 0, 2 / 7]
        indices, stats = select_reward_varying_groups(uids, rewards)
        self.assertEqual(indices, [1, 3, 5, 7])
        self.assertEqual(stats["kept_uids"], ("b",))
        self.assertEqual(stats["dropped_uids"], ("a",))

    def test_zero_and_varying_groups_keep_only_varying_group(self):
        uids = ["zero"] * 4 + ["signal"] * 4
        rewards = [0, 0, 0, 0, 2 / 7, 4 / 7, 2 / 7, 2 / 7]
        indices, stats = select_reward_varying_groups(uids, rewards)
        self.assertEqual(indices, [4, 5, 6, 7])
        self.assertEqual(stats["kept_group_count"], 1)
        self.assertEqual(stats["dropped_group_count"], 1)

    def test_tolerance_treats_tiny_roundoff_as_constant(self):
        indices, _ = select_reward_varying_groups(
            ["a"] * 4,
            [0.5, 0.5 + 1.0e-9, 0.5, 0.5],
            tolerance=1.0e-8,
        )
        self.assertEqual(indices, [])

    def test_varying_behavior_penalties_without_semantic_progress_are_dropped(self):
        indices, stats = select_reward_varying_groups(
            ["a"] * 4,
            [-0.05, -0.02, -0.01, 0.0],
            semantic_rewards=[0.0, 0.0, 0.0, 0.0],
            infrastructure_invalid=[False] * 4,
        )

        self.assertEqual(indices, [])
        self.assertEqual(stats["groups"][0]["drop_reason"], "no_semantic_signal")

    def test_varying_group_with_semantic_progress_is_kept(self):
        indices, stats = select_reward_varying_groups(
            ["a"] * 4,
            [0.0, 0.2, 0.0, 0.0],
            semantic_rewards=[0.0, 0.2, 0.0, 0.0],
            infrastructure_invalid=[False] * 4,
        )

        self.assertEqual(indices, [0, 1, 2, 3])
        self.assertIsNone(stats["groups"][0]["drop_reason"])

    def test_infrastructure_invalid_member_drops_the_whole_group(self):
        indices, stats = select_reward_varying_groups(
            ["a"] * 4,
            [0.0, 0.2, 0.0, 0.0],
            semantic_rewards=[0.0, 0.2, 0.0, 0.0],
            infrastructure_invalid=[False, True, False, False],
        )

        self.assertEqual(indices, [])
        self.assertEqual(stats["groups"][0]["drop_reason"], "infrastructure_invalid")
        self.assertEqual(stats["infrastructure_invalid_group_count"], 1)

    def test_shopping_extra_fields_are_reduced_to_filter_signals(self):
        semantic, invalid = extract_shopping_group_signals(
            [
                {
                    "infrastructure_invalid": False,
                    "reward": {"semantic": 0.2, "native": 0.5},
                },
                {
                    "infrastructure_invalid": True,
                    "reward": {"semantic": 0.0, "native": 0.0},
                },
            ]
        )

        self.assertEqual(semantic, [0.2, 0.0])
        self.assertEqual(invalid, [False, True])

    def test_missing_shopping_filter_signal_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "shopping"):
            extract_shopping_group_signals([None])

    def test_shopping_metrics_are_aggregated_for_a0_and_a1(self):
        infos = [
            {
                "steps": 10,
                "done": True,
                "termination_reason": "environment_done",
                "infrastructure_invalid": False,
                "reward": {
                    "full": 1.0,
                    "strict": 1.0,
                    "native": 1.0,
                    "semantic": 1.7,
                    "total": 1.73,
                    "efficiency": 0.03,
                    "penalty_overlong": 0.0,
                    "penalty_unfinished": 0.0,
                    "penalty_repeat": 0.0,
                    "repeat_action_rate": 0.0,
                    "r_type": 1.0,
                    "r_att": 1.0,
                    "r_option": 1.0,
                    "r_price": 1.0,
                },
            },
            {
                "steps": 35,
                "done": False,
                "termination_reason": "max_steps",
                "infrastructure_invalid": False,
                "reward": {
                    "full": 0.0,
                    "strict": 0.0,
                    "native": 0.0,
                    "semantic": 0.0,
                    "total": -0.05,
                    "efficiency": 0.0,
                    "penalty_overlong": 0.05,
                    "penalty_unfinished": 0.0,
                    "penalty_repeat": 0.0,
                    "repeat_action_rate": 0.0,
                    "r_type": 0.0,
                    "r_att": 0.0,
                    "r_option": 0.0,
                    "r_price": 0.0,
                },
            },
        ]

        metrics = aggregate_shopping_metrics(infos)

        self.assertEqual(metrics["reward/full_mean"], 0.5)
        self.assertEqual(metrics["reward/shaped_min"], -0.05)
        self.assertEqual(metrics["reward/shaped_max"], 1.73)
        self.assertEqual(metrics["component/r_type_mean"], 0.5)
        self.assertEqual(metrics["trajectory/average_steps"], 22.5)
        self.assertEqual(metrics["trajectory/done_rate"], 0.5)
        self.assertEqual(metrics["trajectory/max_steps_rate"], 0.5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
