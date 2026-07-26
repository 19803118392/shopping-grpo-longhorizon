import unittest

from web_agent_site.engine.termination_v2 import ProgressTracker


class TerminationV2Test(unittest.TestCase):
    def test_three_identical_actions_trigger_repeat_loop(self):
        tracker = ProgressTracker(no_new_asin_limit=99)
        first = tracker.record("search", "洁牙器", ["1"])
        second = tracker.record("search", "洁牙器", ["2"])
        third = tracker.record("search", "洁牙器", ["3"])
        self.assertIsNone(first["termination_reason"])
        self.assertIsNone(second["termination_reason"])
        self.assertEqual(third["termination_reason"], "repeat_loop")

    def test_four_steps_without_new_asin_trigger_loop(self):
        tracker = ProgressTracker(exact_repeat_limit=99)
        tracker.record("search", "query", ["1", "2"])
        for index in range(3):
            result = tracker.record("search", f"same-results-{index}", ["1", "2"])
            self.assertIsNone(result["termination_reason"])
        result = tracker.record("search", "same-results-3", ["1", "2"])
        self.assertEqual(result["termination_reason"], "repeat_loop")

    def test_normal_product_inspection_does_not_count_as_no_progress(self):
        tracker = ProgressTracker(exact_repeat_limit=99)
        tracker.record("search", "洗地机", ["123456789012", "999999999999"])
        opened = tracker.record("click", "123456789012", ["123456789012"])
        self.assertEqual(opened["newly_opened_asin_count"], 1)
        for action in ("Description", "Features", "Attributes", "Back to Search"):
            result = tracker.record("click", action, ["123456789012"])
            self.assertIsNone(result["termination_reason"])
            self.assertEqual(result["no_new_asin_steps"], 0)

    def test_reopening_old_candidate_counts_as_no_progress(self):
        tracker = ProgressTracker(exact_repeat_limit=99, no_new_asin_limit=2)
        tracker.record("search", "洗地机", ["123456789012"])
        tracker.record("click", "123456789012", ["123456789012"])
        tracker.record("click", "Back to Search", [])
        first_reopen = tracker.record("click", "123456789012", ["123456789012"])
        self.assertEqual(first_reopen["no_new_asin_steps"], 1)
        tracker.record("click", "Back to Search", [])
        second_reopen = tracker.record("click", "123456789012", ["123456789012"])
        self.assertEqual(second_reopen["termination_reason"], "repeat_loop")

    def test_new_asin_resets_no_progress_counter(self):
        tracker = ProgressTracker(exact_repeat_limit=99)
        tracker.record("search", "one", ["1"])
        tracker.record("search", "same-one", ["1"])
        tracker.record("search", "same-two", ["1"])
        result = tracker.record("search", "two", ["1", "2"])
        self.assertEqual(result["new_asin_count"], 1)
        self.assertEqual(result["no_new_asin_steps"], 0)

    def test_max_steps_is_bounded(self):
        tracker = ProgressTracker(
            max_steps=2,
            exact_repeat_limit=99,
            no_new_asin_limit=99,
        )
        tracker.record("search", "one", ["1"])
        result = tracker.record("search", "two", ["2"])
        self.assertEqual(result["termination_reason"], "max_steps")


if __name__ == "__main__":
    unittest.main()
