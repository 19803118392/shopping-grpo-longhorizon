"""Canonical renderer for ShopSimulator Environment v2 structured state."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from shopping_grpo.environment.product_id import is_product_id

OBSERVATION_VERSION = "shopping-observation-v2"
HEADER = "[SHOPPING_OBSERVATION_V2]"


class StructuredObservationError(ValueError):
    """The environment supplied a malformed or unsafe public state."""


def _text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _list(value):
    if not isinstance(value, list):
        return []
    return [_text(item) for item in value if _text(item)]


def _footer(state):
    actions = _list(state.get("actions"))
    return [
        f"搜索功能是否可用: {bool(state.get('search_available'))}",
        "可点击的按钮: " + json.dumps(actions, ensure_ascii=False),
    ]


def render_structured_observation(state: Mapping) -> str:
    """Render one state without accepting hidden goal or reward payloads."""
    if not isinstance(state, Mapping):
        raise StructuredObservationError("observation_state must be an object")
    if state.get("observation_version") != OBSERVATION_VERSION:
        raise StructuredObservationError("unsupported observation_state version")
    forbidden = {"goal", "reward", "reward_detail", "target_asin", "answer"}
    leaked = forbidden.intersection(state)
    if leaked:
        raise StructuredObservationError(
            "observation_state contains forbidden fields: " + ", ".join(sorted(leaked))
        )

    page_type = str(state.get("page_type") or "unknown")
    lines = [HEADER, f"page_type: {page_type}"]
    if page_type == "search_home":
        lines.append("使用 search_products 提交简短、具有区分度的商品查询。")
    elif page_type == "search_results":
        lines.extend(_render_search_results(state))
    elif page_type in {"product_detail", "information_subpage"}:
        lines.extend(_render_product(state))
        if page_type == "information_subpage":
            lines.append(f"subpage: {_text(state.get('subpage'))}")
            lines.append("content: " + _text(state.get("content")))
    elif page_type != "terminal":
        raise StructuredObservationError(f"unsupported page_type: {page_type!r}")
    return "\n".join(lines) + "\n\n" + "\n".join(_footer(state))


def _render_search_results(state):
    products = state.get("products")
    if not isinstance(products, list):
        raise StructuredObservationError("search results must contain a products list")
    actions = set(_list(state.get("actions")))
    product_asins = []
    lines = [
        f"query: {_text(state.get('query'))}",
        f"normalized_query: {_text(state.get('normalized_query'))}",
        (
            f"Page {int(state.get('page', 1))} of {int(state.get('total_pages', 1))} "
            f"(Total results: {int(state.get('total_results', 0))}; "
            f"ranks {int(state.get('rank_start', 0))}-{int(state.get('rank_end', 0))})"
        ),
        "格式: rank|asin|price|brand|category|key_attributes|title",
    ]
    for product in products:
        if not isinstance(product, Mapping):
            raise StructuredObservationError("each product must be an object")
        asin = _text(product.get("asin"))
        if not is_product_id(asin):
            raise StructuredObservationError(f"invalid search-result ASIN: {asin!r}")
        product_asins.append(asin)
        attributes = ",".join(_list(product.get("key_attributes")))
        lines.append(
            "|".join(
                (
                    str(int(product.get("rank", 0))),
                    asin,
                    _text(product.get("price")),
                    _text(product.get("brand")),
                    _text(product.get("category")),
                    attributes,
                    _text(product.get("title")),
                )
            )
        )
    actionable_asins = {action for action in actions if is_product_id(action)}
    if set(product_asins) != actionable_asins:
        raise StructuredObservationError(
            "model-visible search ASINs differ from environment-actionable ASINs"
        )
    if len(product_asins) > 20:
        raise StructuredObservationError("search page exceeds the frozen page size of 20")
    lines.append(f"products_shown: {len(product_asins)}")
    return lines


def _render_product(state):
    product = state.get("product")
    if not isinstance(product, Mapping):
        raise StructuredObservationError("product page must contain a product object")
    asin = _text(product.get("asin"))
    if not is_product_id(asin):
        raise StructuredObservationError(f"invalid product ASIN: {asin!r}")
    lines = [
        f"asin: {asin}",
        f"title: {_text(product.get('title'))}",
        f"brand: {_text(product.get('brand'))}",
        f"category: {_text(product.get('category'))}",
        f"price: {_text(state.get('selected_price', product.get('price')))}",
        "key_attributes: " + ", ".join(_list(product.get("key_attributes"))),
        "selected_options: "
        + json.dumps(state.get("selected_options") or {}, ensure_ascii=False, sort_keys=True),
        "available_options: "
        + json.dumps(state.get("available_options") or {}, ensure_ascii=False, sort_keys=True),
    ]
    return lines
