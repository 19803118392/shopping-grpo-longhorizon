"""Offline trajectory evaluation for this repository's ShopSimulator runs.

This package is intentionally not imported by the training runtime.  It only
operates on already persisted task facts, rubrics, and rollout JSON records.
"""

from shopping_grpo.evaluation.contracts import (
    CONTRACT_VERSION,
    ERROR_TAXONOMY,
    JUDGE_DIMENSIONS,
    JUDGE_SCHEMA_VERSION,
    RUBRIC_SCHEMA_VERSION,
)

__all__ = [
    "CONTRACT_VERSION",
    "ERROR_TAXONOMY",
    "JUDGE_DIMENSIONS",
    "JUDGE_SCHEMA_VERSION",
    "RUBRIC_SCHEMA_VERSION",
]
