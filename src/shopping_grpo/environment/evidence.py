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

    def __init__(self, *, max_candidates=5, max_chars=384):
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
        self._last_emitted_render = None
        self.last_emission_chars = 0

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
                "best_search_rank": None,
                "detail_visits": 0,
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
        if source == "search_results":
            try:
                rank = int(product.get("rank"))
            except (TypeError, ValueError):
                rank = 0
            if rank > 0:
                previous_rank = candidate["best_search_rank"]
                candidate["best_search_rank"] = (
                    rank if previous_rank is None else min(int(previous_rank), rank)
                )
        else:
            candidate["detail_visits"] += 1
        candidate["last_event_id"] = event_id
        return candidate

    def _candidate_priority(self, candidate):
        evidence_score = (
            len(candidate["key_attributes"])
            + len(candidate["selected_options"])
            + len(candidate["subpages"])
        )
        search_rank = candidate["best_search_rank"]
        return (
            candidate["asin"] != self.current_asin,
            not bool(candidate["detail_visits"]),
            -evidence_score,
            -int(candidate["last_event_id"]),
            int(search_rank) if search_rank is not None else 10**9,
            candidate["asin"],
        )

    def snapshot(self) -> dict:
        """Return a deterministic, JSON-serializable public snapshot."""
        candidates = sorted(
            self.candidates.values(),
            key=self._candidate_priority,
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
        """Render a bounded, action-inert ledger; newest evidence comes first."""
        snapshot = self.snapshot()
        lines = [
            EVIDENCE_MEMORY_HEADER,
            "历史证据只用于比较，不可点击；仅使用 CURRENT_OBSERVATION footer。",
            f"candidate_count: {snapshot['candidate_count']}",
        ]
        header_line_count = len(lines)
        for candidate in snapshot["candidates"]:
            evidence = "; ".join(
                f"{name}={_text(content, 48)}"
                for name, content in sorted(candidate["subpages"].items())
            )
            selected_options = _text(
                json.dumps(
                    candidate["selected_options"],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                72,
            )
            candidate_line = "candidate " + " | ".join(
                (
                    f"title={_text(candidate['title'], 64)}",
                    f"brand={_text(candidate['brand'], 24)}",
                    f"category={_text(candidate['category'], 24)}",
                    f"price={_text(candidate['selected_price'] or candidate['price'], 24)}",
                    "attributes="
                    + ", ".join(_text(value, 32) for value in candidate["key_attributes"][:2]),
                    "selected_options=" + selected_options,
                    f"evidence={_text(evidence, 72)}",
                )
            )
            candidate_render = "\n".join([*lines, candidate_line])
            if len(candidate_render) > self.max_chars:
                break
            lines.append(candidate_line)
        marker = "[EVIDENCE_MEMORY_TRUNCATED]"
        rendered = "\n".join(lines)
        if len(lines) - header_line_count < len(snapshot["candidates"]):
            with_marker = rendered + "\n" + marker
            if len(with_marker) <= self.max_chars:
                rendered = with_marker
        if len(rendered) > self.max_chars:
            # Headers are deliberately short, but retain a deterministic fallback.
            rendered = rendered[: self.max_chars - len(marker) - 1].rstrip() + "\n" + marker
        return rendered

    def render_update(self) -> str:
        """Emit a snapshot only when its actor-visible contents changed."""
        rendered = self.render()
        if rendered == self._last_emitted_render:
            self.last_emission_chars = 0
            return ""
        self._last_emitted_render = rendered
        self.last_emission_chars = len(rendered)
        return rendered


def augment_observation_with_evidence(observation: str, memory: EvidenceMemory) -> str:
    """Place memory before the current page so the actionable footer remains last."""
    update = memory.render_update()
    if not update:
        return str(observation)
    return update + "\n\n[CURRENT_OBSERVATION]\n" + str(observation)
