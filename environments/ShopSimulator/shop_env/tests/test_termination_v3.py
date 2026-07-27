import unittest

from web_agent_site.engine.termination_v3 import EvidenceProgressTracker


class TerminationV3Test(unittest.TestCase):
    def test_result_set_requires_three_new_asins_and_new_fingerprint(self):
        tracker = EvidenceProgressTracker(exact_repeat_limit=99, no_progress_limit=99)
        first = tracker.record("search", "one", ["1", "2"])
        second = tracker.record("search", "two", ["1", "2", "3", "4", "5"])
        repeated = tracker.record("search", "three", ["1", "2", "3", "4", "5"])
        self.assertEqual(first["effective_result_sets"], 0)
        self.assertEqual(second["effective_result_sets"], 1)
        self.assertEqual(repeated["effective_result_sets"], 1)

    def test_evidence_budgets_prevent_unlimited_product_progress(self):
        tracker = EvidenceProgressTracker(
            exact_repeat_limit=99,
            no_progress_limit=99,
            product_open_progress_budget=2,
        )
        first = tracker.record("click", "111111111111", ["111111111111"])
        second = tracker.record("click", "222222222222", ["222222222222"])
        third = tracker.record("click", "333333333333", ["333333333333"])
        self.assertTrue(first["evidence_added"])
        self.assertTrue(second["evidence_added"])
        self.assertFalse(
            any(item.startswith("product:") for item in third["evidence_added"])
        )
        self.assertEqual(third["no_progress_steps"], 1)

    def test_subpage_and_option_are_unique_evidence(self):
        tracker = EvidenceProgressTracker(exact_repeat_limit=99, no_progress_limit=99)
        tracker.record("click", "111111111111", ["111111111111"])
        first = tracker.record(
            "click",
            "Features",
            ["111111111111"],
            page_type="information_subpage",
        )
        repeated = tracker.record(
            "click",
            "Features",
            ["111111111111"],
            page_type="information_subpage",
        )
        option = tracker.record(
            "click",
            "白色",
            ["111111111111"],
            page_type="product_detail",
            selected_options={"颜色分类": "白色"},
        )
        self.assertTrue(any(item.startswith("subpage:") for item in first["evidence_added"]))
        self.assertFalse(
            any(item.startswith("subpage:") for item in repeated["evidence_added"])
        )
        self.assertTrue(any(item.startswith("option:") for item in option["evidence_added"]))

    def test_four_actions_without_new_evidence_terminate(self):
        tracker = EvidenceProgressTracker(
            exact_repeat_limit=99,
            no_progress_limit=4,
        )
        tracker.record("search", "one", ["1", "2", "3"])
        for index in range(3):
            result = tracker.record("click", f"invalid-{index}", [])
            self.assertIsNone(result["termination_reason"])
        result = tracker.record("click", "invalid-final", [])
        self.assertEqual(result["termination_reason"], "repeat_loop")

    def test_eleven_digit_catalog_id_adds_product_evidence(self):
        tracker = EvidenceProgressTracker(exact_repeat_limit=99, no_progress_limit=99)
        result = tracker.record("click", "35842622441", ["35842622441"])
        self.assertIn("product:35842622441", result["evidence_added"])
        self.assertEqual(result["opened_candidate_count"], 1)


if __name__ == "__main__":
    unittest.main()
