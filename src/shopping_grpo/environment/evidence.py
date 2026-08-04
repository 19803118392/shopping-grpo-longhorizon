"""Bounded task-local memory built only from public observation v2 fields."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from copy import deepcopy

from shopping_grpo.environment.observation import render_structured_observation

EVIDENCE_MEMORY_VERSION = "shopping-evidence-memory-v1"
EVIDENCE_MEMORY_HEADER = "[SHOPPING_EVIDENCE_MEMORY_V1]"


def _text(value, limit=240):
    normalized = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 1)].rstrip() + "…"


def _string_list(value, *, item_limit=120):
    if not isinstance(value, list):
        return []
    return [_text(item, item_limit) for item in value if _text(item, item_limit)]


class EvidenceMemory:
    """Merge public candidate evidence across pages without retaining hidden goals."""

    def __init__(self, *, max_candidates=5, max_chars=2_000):
        self.max_candidates = int(max_candidates)
        self.max_chars = int(max_chars)
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if self.max_chars < 256:
            raise ValueError("max_chars must be at least 256")
        self.queries = []
        self.candidates = {}
        self.current_asin = None
        self.event_count = 0

    def observe(self, state: Mapping, *, event_id=None, tool_name=None) -> dict:
        """Validate and merge one canonical public observation state."""
        # The canonical renderer is the security boundary: version mismatches and
        # hidden reward/goal fields fail before anything can enter memory.
        render_structured_observation(state)
        self.event_count += 1
        event_id = self.event_count if event_id is None else event_id
        page_type = str(state.get("page_type") or "")
        if page_type == "search_results":
            query = _text(state.get("query"), 160)
            if query and query not in self.queries:
                self.queries.append(query)
            for product in state.get("products") or []:
                self._merge_product(product, event_id=event_id, source="search_results")
            self.current_asin = None
        elif page_type in {"product_detail", "information_subpage"}:
            product = state["product"]
            asin = _text(product.get("asin"), 32)
            candidate = self._merge_product(
                product,
                event_id=event_id,
                source=page_type,
            )
            candidate["selected_price"] = _text(
                state.get("selected_price", product.get("price")), 80
            )
            selected_options = state.get("selected_options") or {}
            available_options = state.get("available_options") or {}
            if not isinstance(selected_options, Mapping):
                selected_options = {}
            if not isinstance(available_options, Mapping):
                available_options = {}
            candidate["selected_options"] = {
                _text(key, 80): _text(value, 120)
                for key, value in selected_options.items()
            }
            candidate["available_options"] = {
                _text(key, 80): _string_list(values)
                for key, values in available_options.items()
            }
            if page_type == "information_subpage":
                subpage = _text(state.get("subpage"), 40) or "information"
                candidate["subpages"][subpage] = _text(state.get("content"), 360)
            self.current_asin = asin
        elif page_type in {"search_home", "terminal"}:
            self.current_asin = None
        return {
            "event_id": event_id,
            "tool_name": str(tool_name or ""),
            "page_type": page_type,
            "candidate_count": len(self.candidates),
            "current_asin": self.current_asin,
        }

    def _merge_product(self, product, *, event_id, source):
        asin = _text(product.get("asin"), 32)
        candidate = self.candidates.setdefault(
            asin,
            {
                "asin": asin,
                "title": "",
                "brand": "",
                "category": "",
                "price": "",
                "key_attributes": [],
                "selected_price": "",
                "selected_options": {},
                "available_options": {},
                "subpages": {},
                "sources": [],
                "last_event_id": event_id,
            },
        )
        for field in ("title", "brand", "category", "price"):
            value = _text(product.get(field), 240)
            if value:
                candidate[field] = value
        attributes = _string_list(product.get("key_attributes"))
        candidate["key_attributes"] = list(
            dict.fromkeys([*candidate["key_attributes"], *attributes])
        )
        if source not in candidate["sources"]:
            candidate["sources"].append(source)
        candidate["last_event_id"] = event_id
        return candidate

    def snapshot(self) -> dict:
        """Return a deterministic, JSON-serializable public snapshot."""
        candidates = sorted(
            self.candidates.values(),
            key=lambda item: (-int(item["last_event_id"]), item["asin"]),
        )[: self.max_candidates]
        return {
            "schema_version": EVIDENCE_MEMORY_VERSION,
            "event_count": self.event_count,
            "queries": list(self.queries[-5:]),
            "current_asin": self.current_asin,
            "candidate_count": len(self.candidates),
            "candidates": [deepcopy(candidate) for candidate in candidates],
        }

    def render(self) -> str:
        """Render a bounded model-visible ledger; the newest candidates come first."""
        snapshot = self.snapshot()
        lines = [
            EVIDENCE_MEMORY_HEADER,
            "queries: " + json.dumps(snapshot["queries"], ensure_ascii=False),
            (
                f"candidate_count: {snapshot['candidate_count']}; "
                f"current_asin: {snapshot['current_asin'] or ''}"
            ),
        ]
        for candidate in snapshot["candidates"]:
            lines.append(
                "candidate "
                + " | ".join(
                    (
                        f"asin={candidate['asin']}",
                        f"title={candidate['title']}",
                        f"brand={candidate['brand']}",
                        f"category={candidate['category']}",
                        f"price={candidate['selected_price'] or candidate['price']}",
                        "attributes=" + ", ".join(candidate["key_attributes"]),
                        "selected_options="
                        + json.dumps(
                            candidate["selected_options"],
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "available_options="
                        + json.dumps(
                            candidate["available_options"],
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    )
                )
            )
            for subpage, content in sorted(candidate["subpages"].items()):
                lines.append(
                    f"evidence asin={candidate['asin']} subpage={subpage}: {content}"
                )
        rendered = "\n".join(lines)
        if len(rendered) <= self.max_chars:
            return rendered
        marker = "\n[EVIDENCE_MEMORY_TRUNCATED]"
        return rendered[: self.max_chars - len(marker)].rstrip() + marker


def augment_observation_with_evidence(observation: str, memory: EvidenceMemory) -> str:
    """Place memory before the current page so the actionable footer remains last."""
    return memory.render() + "\n\n[CURRENT_OBSERVATION]\n" + str(observation)
