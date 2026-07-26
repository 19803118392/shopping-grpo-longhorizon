"""Pure, reproducible goal helpers for Environment v2."""

from __future__ import annotations

import hashlib
import math
import random
import re


def explicit_budget_from_instruction(instruction):
    """Extract a clearly stated upper budget; return None when ambiguous."""
    text = str(instruction or "").replace(",", "")
    shorthand = re.search(
        r"预算(?:控制)?在?\s*(\d+)\s*万\s*(\d+)\s*(?:千)?\s*(以内|以下|内|左右)?",
        text,
    )
    if shorthand:
        value = float(shorthand.group(1)) * 10000 + float(shorthand.group(2)) * 1000
        if shorthand.group(3) == "左右":
            value *= 1.1
        return value
    patterns = (
        r"预算(?:控制)?在?\s*(\d+(?:\.\d+)?)\s*(万|千)?\s*元?(以内|以下|内|左右)?",
        r"(?:不超过|不高于|最高)\s*(\d+(?:\.\d+)?)\s*(万|千)?\s*元",
        r"(\d+(?:\.\d+)?)\s*(万|千)?\s*元(以内|以下)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            if unit == "万":
                value *= 10000
            elif unit == "千":
                value *= 1000
            qualifier = match.group(3) if match.lastindex and match.lastindex >= 3 else None
            # “左右” is not a strict upper bound. Environment v2 freezes a
            # small deterministic tolerance rather than letting target items
            # just above the round-number budget become impossible.
            if qualifier == "左右":
                value *= 1.1
            if value > 0:
                return value
    return None


def _price_range_above(price):
    if price <= 100:
        step = 3
    elif price <= 1000:
        step = 10
    elif price <= 5000:
        step = 50
    elif price <= 10000:
        step = 100
    else:
        step = 4
    base = math.ceil(price / 10) * 10
    return [base + index * 10 for index in range(step)]


def deterministic_price_upper(asin, instruction, price):
    explicit = explicit_budget_from_instruction(instruction)
    if explicit is not None:
        return explicit
    price_range = _price_range_above(float(price))
    if len(price_range) < 2:
        return 10000000
    digest = hashlib.sha256(
        f"{asin}\0{instruction}".encode("utf-8")
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    _, upper = sorted(rng.sample(price_range, 2))
    return upper
