import unittest

from shopping_grpo.environment.evidence import (
    EVIDENCE_MEMORY_HEADER,
    EvidenceMemory,
    augment_observation_with_evidence,
)
from shopping_grpo.environment.observation import render_structured_observation

ASIN = "100000000001"


def search_state():
    return {
        "observation_version": "shopping-observation-v2",
        "page_type": "search_results",
        "query": "天然乳胶枕",
        "normalized_query": "天然乳胶枕",
        "page": 1,
        "total_pages": 1,
        "total_results": 1,
        "rank_start": 1,
        "rank_end": 1,
        "products": [
            {
                "rank": 1,
                "asin": ASIN,
                "price": "¥199",
                "brand": "Example",
                "category": "乳胶枕",
                "key_attributes": ["天然乳胶"],
                "title": "Example 天然乳胶枕",
            }
        ],
        "search_available": True,
        "actions": [ASIN, "back to search"],
    }


def product_state(page_type="product_detail"):
    state = {
        "observation_version": "shopping-observation-v2",
        "page_type": page_type,
        "product": {
            "asin": ASIN,
            "title": "Example 天然乳胶枕升级款",
            "brand": "Example",
            "category": "乳胶枕",
            "price": "¥199",
            "key_attributes": ["可拆洗"],
        },
        "selected_price": "¥219",
        "selected_options": {"高度": "10cm"},
        "available_options": {"高度": ["8cm", "10cm"]},
        "search_available": False,
        "actions": ["Buy Now"] if page_type == "product_detail" else ["< Prev"],
    }
    if page_type == "information_subpage":
        state.update({"subpage": "features", "content": "天然乳胶，可拆洗。"})
    return state


class EvidenceMemoryTest(unittest.TestCase):
    def test_merges_search_detail_options_and_subpage_evidence(self):
        memory = EvidenceMemory()
        memory.observe(search_state(), event_id=1, tool_name="search_products")
        memory.observe(product_state(), event_id=2, tool_name="open_product")
        memory.observe(
            product_state("information_subpage"),
            event_id=3,
            tool_name="view_features",
        )

        snapshot = memory.snapshot()
        candidate = snapshot["candidates"][0]
        self.assertEqual(snapshot["queries"], ["天然乳胶枕"])
        self.assertEqual(candidate["selected_price"], "¥219")
        self.assertEqual(candidate["selected_options"], {"高度": "10cm"})
        self.assertEqual(
            candidate["key_attributes"], ["天然乳胶", "可拆洗"]
        )
        self.assertEqual(candidate["subpages"]["features"], "天然乳胶，可拆洗。")
        self.assertIn('available_options={"高度": ["8cm", "10cm"]}', memory.render())

    def test_snapshot_does_not_mutate_after_later_observations(self):
        memory = EvidenceMemory()
        memory.observe(search_state())
        snapshot = memory.snapshot()

        memory.observe(product_state())

        self.assertEqual(snapshot["candidates"][0]["key_attributes"], ["天然乳胶"])

    def test_canonical_renderer_blocks_hidden_fields_before_memory_update(self):
        memory = EvidenceMemory()
        leaked = search_state()
        leaked["reward_detail"] = {"target_asin": ASIN}

        with self.assertRaisesRegex(ValueError, "forbidden fields"):
            memory.observe(leaked)
        self.assertEqual(memory.event_count, 0)
        self.assertEqual(memory.candidates, {})

    def test_render_is_bounded_and_does_not_create_actionable_search_rows(self):
        memory = EvidenceMemory(max_chars=256)
        state = search_state()
        state["products"][0]["title"] = "长标题" * 200
        memory.observe(state)

        rendered = memory.render()

        self.assertLessEqual(len(rendered), 256)
        self.assertIn("EVIDENCE_MEMORY_TRUNCATED", rendered)
        self.assertNotRegex(rendered, rf"(?m)^\d+\|{ASIN}\|")

    def test_augmented_message_keeps_current_observation_footer_last(self):
        memory = EvidenceMemory()
        state = search_state()
        memory.observe(state)
        observation = render_structured_observation(state)

        augmented = augment_observation_with_evidence(observation, memory)

        self.assertTrue(augmented.startswith(EVIDENCE_MEMORY_HEADER))
        self.assertTrue(augmented.endswith('可点击的按钮: ["100000000001", "back to search"]'))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
