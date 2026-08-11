#!/usr/bin/env python3
"""Verify two seed-replay outputs after removing non-model identifiers."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--attempts-per-task", type=int, default=2)
    return parser.parse_args()


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _message(message: object) -> object:
    if not isinstance(message, Mapping):
        return message
    normalized = {
        key: value
        for key, value in message.items()
        if key not in {"id", "created_at", "timestamp", "tool_call_id"}
    }
    calls = normalized.get("tool_calls")
    if isinstance(calls, list):
        normalized["tool_calls"] = [
            {key: value for key, value in call.items() if key not in {"id", "tool_call_id"}}
            if isinstance(call, Mapping)
            else call
            for call in calls
        ]
    return normalized


def _model_trace(row: Mapping) -> dict:
    return {
        "attempt_seed": (row.get("actor_sampling") or {}).get("attempt_seed"),
        "messages": [_message(message) for message in row.get("messages") or []],
        "actions": [
            {
                "tool_name": step.get("tool_name"),
                "parameters": step.get("parameters"),
            }
            for step in row.get("steps") or []
        ],
    }


def verify_seed_replay(
    first: list[Mapping],
    second: list[Mapping],
    *,
    expected_tasks: int = 5,
    attempts_per_task: int = 2,
) -> dict:
    def indexed(rows):
        result = {}
        for row in rows:
            key = (int(row["task_id"]), int(row.get("attempt_index", 0)))
            if key in result:
                raise ValueError(f"duplicate seed replay pair {key}")
            result[key] = _model_trace(row)
        return result

    left = indexed(first)
    right = indexed(second)
    expected_pairs = int(expected_tasks) * int(attempts_per_task)
    if len(left) != expected_pairs or len(right) != expected_pairs or set(left) != set(right):
        raise ValueError(
            "seed replay coverage mismatch: "
            f"expected={expected_pairs} first={len(left)} second={len(right)}"
        )
    mismatches = [
        {"task_id": key[0], "attempt_index": key[1]}
        for key in sorted(left)
        if left[key] != right[key]
    ]
    return {
        "schema_version": "shopping-seed-replay-v1",
        "expected_pairs": expected_pairs,
        "matched_pairs": expected_pairs - len(mismatches),
        "mismatches": mismatches,
        "exact_model_and_action_replay": not mismatches,
        "pairing_interpretation": (
            "common_random_numbers" if not mismatches else "task_paired_only"
        ),
    }


def main():
    args = parse_args()
    report = verify_seed_replay(
        _rows(args.first),
        _rows(args.second),
        expected_tasks=args.tasks,
        attempts_per_task=args.attempts_per_task,
    )
    report["provenance"] = {
        "first_sha256": hashlib.sha256(args.first.read_bytes()).hexdigest(),
        "second_sha256": hashlib.sha256(args.second.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
