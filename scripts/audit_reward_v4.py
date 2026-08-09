#!/usr/bin/env python3
"""Offline audit for the adapter-only target-ASIN-bonus-free terminal objective."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from shopping_grpo.evaluation.summary import (
    is_constraint_complete_purchase_v4,
    is_strict_success,
    optimization_reward_v4_from_trajectory,
)
from shopping_grpo.training.grpo.adapter.runtime import optimization_reward_v4, validate_reward


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _active_dimension_count(detail: dict) -> int | None:
    dimensions = (
        ((detail.get("evidence") or {}).get("preference_scoring") or {}).get("dimensions")
        or {}
    )
    if not isinstance(dimensions, dict) or not dimensions:
        return None
    return sum(bool(value.get("active")) for value in dimensions.values() if isinstance(value, dict))


def _active_bucket(value: int | None) -> str:
    if value is None:
        return "unknown"
    if value >= 3:
        return "3+"
    return str(value)


def audit(paths: list[Path]) -> dict:
    rows = []
    provenance = []
    for path in paths:
        resolved = path.resolve()
        file_rows = _read_jsonl(resolved)
        rows.extend((row, str(resolved)) for row in file_rows)
        provenance.append(
            {"path": str(resolved), "sha256": _sha256(resolved), "rows": len(file_rows)}
        )

    reward_types = Counter()
    active_counts = defaultdict(Counter)
    invariance_failures = []
    manual_review = []
    v4_total = 0.0
    strict = constraint_complete = valid = 0
    for row, source_path in rows:
        detail = ((row.get("terminal_result") or {}).get("reward_detail") or {})
        reward_type = str(detail.get("reward_type") or "missing")
        reward_types[reward_type] += 1
        strict += is_strict_success(row)
        complete = is_constraint_complete_purchase_v4(row)
        constraint_complete += complete
        score = optimization_reward_v4_from_trajectory(row)
        v4_total += score
        valid += detail.get("reward_valid") is True
        bucket = _active_bucket(_active_dimension_count(detail))
        active_counts[bucket]["attempts"] += 1
        active_counts[bucket]["constraint_complete"] += complete
        active_counts[bucket]["strict_gold"] += is_strict_success(row)

        if complete and reward_type == "valid_alternative_purchase":
            manual_review.append(
                {
                    "task_id": int(row["task_id"]),
                    "attempt_index": int(row.get("attempt_index", 0)),
                    "source_path": source_path,
                    "active_dimension_bucket": bucket,
                    "weighted_score": detail.get("weighted_score"),
                    "evidence_coverage": detail.get("evidence_coverage"),
                }
            )
        if reward_type in {"gold_purchase", "valid_alternative_purchase"}:
            try:
                public = validate_reward(detail)
            except (TypeError, ValueError):
                continue
            flipped = deepcopy(public)
            flipped["reward_type"] = (
                "valid_alternative_purchase"
                if reward_type == "gold_purchase"
                else "gold_purchase"
            )
            flipped["termination_reason"] = flipped["reward_type"]
            flipped["target_asin_match"] = not bool(flipped.get("target_asin_match"))
            if optimization_reward_v4(public) != optimization_reward_v4(flipped):
                invariance_failures.append(
                    {
                        "task_id": int(row["task_id"]),
                        "attempt_index": int(row.get("attempt_index", 0)),
                        "source_path": source_path,
                    }
                )

    attempts = len(rows)
    return {
        "schema_version": "shopping-reward-v4-offline-audit-v2",
        "provenance": provenance,
        "attempts": attempts,
        "reward_type_counts": dict(sorted(reward_types.items())),
        "reward_valid_attempts": valid,
        "strict_gold_successes_v3": strict,
        "constraint_complete_successes_v4": constraint_complete,
        "constraint_complete_rate_v4": constraint_complete / attempts if attempts else 0.0,
        "mean_optimization_reward_v4": v4_total / attempts if attempts else 0.0,
        "target_asin_invariance_failures": invariance_failures,
        "by_active_soft_dimension_count": {
            bucket: dict(counts) for bucket, counts in sorted(active_counts.items())
        },
        "manual_review_non_gold_full_matches": manual_review,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(args.input)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite audit output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
