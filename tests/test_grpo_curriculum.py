import unittest

from shopping_grpo.training.grpo.curriculum import (
    build_curriculum_plan,
    task_ids_sha256,
    validate_no_task_overlap,
    validate_length_metadata,
)

ROWS = [
    {"task_id": 1, "probe_steps": 7, "length_bucket": "short"},
    {"task_id": 2, "probe_steps": 15, "length_bucket": "medium"},
    {"task_id": 3, "probe_steps": 25, "length_bucket": "long"},
    {"task_id": 4, "probe_steps": 9, "length_bucket": "short"},
]


class GrpoCurriculumTest(unittest.TestCase):
    def test_default_stages_are_cumulative_and_preserve_source_order(self):
        plan = build_curriculum_plan(ROWS)

        self.assertFalse(plan["hidden_goal_fields_used"])
        self.assertEqual(
            [stage["tasks"] for stage in plan["stages"]],
            [2, 3, 4],
        )
        self.assertEqual(plan["stages"][1]["task_ids"], [1, 2, 4])
        self.assertEqual(plan["stages"][2]["task_ids"], [1, 2, 3, 4])

    def test_bucket_must_match_frozen_probe_step_boundaries(self):
        wrong = [{"task_id": 1, "probe_steps": 21, "length_bucket": "medium"}]

        with self.assertRaisesRegex(ValueError, "conflicts with probe_steps"):
            validate_length_metadata(wrong)

    def test_duplicate_task_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate metadata task_id"):
            validate_length_metadata([ROWS[0], ROWS[0]])

    def test_custom_stages_must_remain_cumulative(self):
        with self.assertRaisesRegex(ValueError, "must be cumulative"):
            build_curriculum_plan(
                ROWS,
                stages=(("first", ("short", "medium")), ("second", ("long",))),
            )

    def test_held_out_overlap_is_rejected_and_ids_are_hashed_in_order(self):
        audit = validate_no_task_overlap([1, 2, 3], [4, 5])
        self.assertEqual(audit["overlap_tasks"], 0)
        self.assertNotEqual(task_ids_sha256([1, 2]), task_ids_sha256([2, 1]))
        with self.assertRaisesRegex(ValueError, "overlaps"):
            validate_no_task_overlap([1, 2, 3], [3, 4])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
