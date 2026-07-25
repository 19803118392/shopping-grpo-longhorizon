#!/usr/bin/env python3
"""Deterministically merge SFT JSONL files by task_id and exclude benchmark tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def task_id(row: dict, source: Path) -> int:
    value = row.get("task_id")
    if value is None:
        raise ValueError(f"{source} contains a row without task_id")
    return int(value)


def merge_rows(inputs: list[Path], benchmark: Path) -> tuple[list[dict], dict]:
    benchmark_ids = {task_id(row, benchmark) for row in read_jsonl(benchmark)}
    merged: dict[int, dict] = {}
    source_counts: dict[str, int] = {}
    duplicate_count = 0
    excluded_benchmark_count = 0

    for source in inputs:
        rows = read_jsonl(source)
        source_counts[str(source)] = len(rows)
        for row in rows:
            row_task_id = task_id(row, source)
            if row_task_id in benchmark_ids:
                excluded_benchmark_count += 1
                continue
            if row_task_id in merged:
                duplicate_count += 1
                continue
            merged[row_task_id] = row

    stats = {
        "source_counts": source_counts,
        "input_rows": sum(source_counts.values()),
        "duplicate_rows_removed": duplicate_count,
        "benchmark_rows_excluded": excluded_benchmark_count,
        "merged_rows": len(merged),
        "benchmark_task_count": len(benchmark_ids),
    }
    return list(merged.values()), stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, stats = merge_rows(args.input, args.benchmark)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
