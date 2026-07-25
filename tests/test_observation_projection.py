"""Observation projection preserves actionable state and bounds visible tokens."""

import json
import unittest

from shopping_grpo.action_validation import (
    action_reject_reason,
    clickable_buttons,
    product_ids,
)
from shopping_grpo.observation_projection import (
    ObservationProjectionError,
    TRUNCATION_MARKER,
    project_observation,
)
from shopping_grpo.sft_data import project_sft_messages


def search_page(product_count=12):
    segments = [
        "Instruction:",
        "find a useful product",
        "Back to Search",
        "Page 1 (Total results: 12)",
        "Next >",
    ]
    buttons = ["back to search", "next >"]
    for index in range(product_count):
        asin = f"{100000000000 + index}"
        segments.extend([asin, f"product title {index} " + "x" * 40, f"{index + 1}.0"])
        buttons.append(asin)
    return (
        " [SEP] ".join(segments)
        + "\n\n搜索功能是否可用: False"
        + "\n\n可点击的按钮: "
        + json.dumps(buttons, ensure_ascii=False)
    )


class ObservationProjectionTest(unittest.TestCase):
    def test_search_projection_keeps_navigation_and_only_visible_product_targets(self):
        raw = search_page()
        visible, meta = project_observation(
            "search_products",
            raw,
            parameters={"query": "useful product"},
            count_tokens=len,
            token_budget=500,
            search_top_k=8,
        )

        self.assertLessEqual(len(visible), 500)
        self.assertTrue(meta.truncated)
        self.assertTrue(meta.critical_footer_preserved)
        self.assertIn(TRUNCATION_MARKER, visible)
        self.assertEqual(
            {button.casefold() for button in clickable_buttons(visible) if not button.isdigit()},
            {"back to search", "next >"},
        )
        self.assertEqual(
            set(product_ids(visible)),
            {button for button in clickable_buttons(visible) if button.isdigit()},
        )

    def test_guard_accepts_visible_asin_and_rejects_projected_away_asin(self):
        raw = search_page()
        visible, _ = project_observation(
            "search_products",
            raw,
            parameters={"query": "useful product"},
            count_tokens=len,
            token_budget=500,
            search_top_k=8,
        )
        visible_asin = product_ids(visible)[0]
        omitted_asin = product_ids(raw)[-1]

        self.assertIsNone(
            action_reject_reason("open_product", {"asin": visible_asin}, visible)
        )
        self.assertEqual(
            action_reject_reason("open_product", {"asin": omitted_asin}, visible),
            "click_not_in_previous_observation",
        )

    def test_short_product_page_is_identity_projection(self):
        raw = (
            "Product [SEP] price: 20 [SEP] Buy Now"
            "\n\n搜索功能是否可用: False"
            '\n\n可点击的按钮: ["back to search", "buy now"]'
        )
        visible, meta = project_observation(
            "open_product",
            raw,
            count_tokens=len,
            token_budget=448,
        )

        self.assertEqual(visible, raw)
        self.assertFalse(meta.truncated)

    def test_generic_projection_keeps_complete_footer(self):
        raw = (
            "Description " + "detail " * 200 + "TAIL_SPECIFICATION"
            + "\n\n搜索功能是否可用: False"
            + '\n\n可点击的按钮: ["back to search", "< prev"]'
        )
        visible, meta = project_observation(
            "view_description",
            raw,
            count_tokens=len,
            generic_token_budget=300,
        )

        self.assertLessEqual(len(visible), 300)
        self.assertEqual(
            clickable_buttons(visible),
            ["back to search", "< prev"],
        )
        self.assertIn("TAIL_SPECIFICATION", visible)
        self.assertTrue(meta.critical_footer_preserved)

    def test_sft_projection_uses_the_same_visible_tool_message(self):
        raw = search_page()
        messages = [
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "search_products",
                            "arguments": '{"query":"useful product"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "search_products",
                "content": raw,
            },
        ]

        projected, rows = project_sft_messages(
            messages,
            lambda name, observation, parameters: (
                lambda result: (result[0], result[1].to_dict())
            )(
                project_observation(
                    name,
                    observation,
                    parameters=parameters,
                    count_tokens=len,
                    token_budget=500,
                )
            ),
        )

        self.assertNotEqual(projected[2]["content"], raw)
        self.assertEqual(messages[2]["content"], raw)
        self.assertTrue(rows[0]["truncated"])

    def test_long_page_without_action_footer_fails_closed(self):
        with self.assertRaisesRegex(ObservationProjectionError, "action footer"):
            project_observation(
                "unknown",
                "x" * 1000,
                count_tokens=len,
                generic_token_budget=128,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
