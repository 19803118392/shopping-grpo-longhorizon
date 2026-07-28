#!/usr/bin/env python3
"""Freeze a blind Reward v3 final-test manifest without running any model."""

from __future__ import annotations

import argparse
import gzip
import json
import random
import subprocess
from pathlib import Path

from shopping_grpo.grpo_tasks import sha256_file, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "environment-v2.1/reward-v3/fresh-v1"
DEFAULT_TASKS = ROOT / "data/shop_tasks.jsonl"
DEFAULT_OUTPUT = ROOT / "data/benchmarks/shop_benchmark_reward_v3_final_200.jsonl"
DEFAULT_EXCLUSIONS = (
    ROOT / "data/benchmarks/shop_benchmark_reward_v3_final_200.exclusions.jsonl"
)
DEFAULT_METADATA = (
    ROOT / "data/benchmarks/shop_benchmark_reward_v3_final_200.metadata.json"
)


def repository_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def task_id_from_row(row: dict) -> int | None:
    task_id = row.get("task_id")
    if task_id is None and isinstance(row.get("extra_info"), dict):
        task_id = row["extra_info"].get("task_id")
    return int(task_id) if task_id is not None else None


def read_task_ids(path: Path) -> tuple[set[int], int]:
    opener = gzip.open if path.suffix == ".gz" else open
    task_ids = set()
    row_count = 0
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            task_id = task_id_from_row(row)
            if task_id is not None:
                task_ids.add(task_id)
    return task_ids, row_count


def default_exclusion_sources(skip: set[Path]) -> list[Path]:
    paths = set()
    for pattern in (
        "data/benchmarks/*.jsonl",
        "data/splits/*.jsonl",
        "outputs/**/*.jsonl",
        "outputs/**/*.jsonl.gz",
    ):
        paths.update(path.resolve() for path in ROOT.glob(pattern) if path.is_file())
    return sorted(paths - {path.resolve() for path in skip})


def build_exclusion_snapshot(paths: list[Path]) -> tuple[set[int], list[dict]]:
    excluded = set()
    sources = []
    for path in paths:
        task_ids, row_count = read_task_ids(path)
        if not task_ids:
            continue
        excluded.update(task_ids)
        resolved = path.resolve()
        sources.append(
            {
                "path": str(resolved.relative_to(ROOT)),
                "sha256": sha256_file(resolved),
                "row_count": row_count,
                "task_count": len(task_ids),
            }
        )
    return excluded, sources


def select_final_test(
    all_task_ids: set[int],
    excluded_task_ids: set[int],
    *,
    size: int,
    seed: int,
) -> list[dict]:
    size = int(size)
    if size < 1:
        raise ValueError("final-test size must be positive")
    eligible = sorted(all_task_ids - excluded_task_ids)
    if len(eligible) < size:
        raise ValueError("final-test size exceeds the unseen task population")
    random.Random(int(seed)).shuffle(eligible)
    return [{"task_id": task_id} for task_id in eligible[:size]]


def refuse_overwrite(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite frozen final-test asset(s): " + ", ".join(existing)
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def main():
    args = parse_args()
    outputs = [args.output.resolve(), args.exclusions.resolve(), args.metadata.resolve()]
    refuse_overwrite(outputs)

    all_task_ids, source_rows = read_task_ids(args.tasks.resolve())
    exclusion_sources = default_exclusion_sources(set(outputs))
    excluded, source_report = build_exclusion_snapshot(exclusion_sources)
    rows = select_final_test(
        all_task_ids,
        excluded,
        size=args.size,
        seed=args.seed,
    )
    final_ids = {int(row["task_id"]) for row in rows}
    overlap = final_ids & excluded
    if overlap:
        raise RuntimeError(f"final test overlaps exclusion snapshot: {sorted(overlap)}")

    write_jsonl(args.exclusions, ({"task_id": task_id} for task_id in sorted(excluded)))
    write_jsonl(args.output, rows)
    metadata = {
        "asset": "shop_benchmark_reward_v3_final_200",
        "contract": CONTRACT,
        "environment_version": "shopsimulator-environment-v2.1",
        "reward_version": "shopsimulator-reward-v3",
        "shopping_grpo_commit": repository_head(),
        "task_count": len(rows),
        "seed": int(args.seed),
        "selection": "deterministic_random_without_replacement",
        "selection_uses_model_rollout": False,
        "evaluated": False,
        "source_tasks": str(args.tasks.resolve().relative_to(ROOT)),
        "source_tasks_sha256": sha256_file(args.tasks),
        "source_task_count": len(all_task_ids),
        "source_row_count": source_rows,
        "exclusion_source_file_count": len(source_report),
        "excluded_task_count": len(excluded),
        "eligible_task_count": len(all_task_ids - excluded),
        "exclusions": str(args.exclusions.resolve().relative_to(ROOT)),
        "exclusions_sha256": sha256_file(args.exclusions),
        "exclusion_sources": source_report,
        "checks": {
            "final_test_exclusion_overlap": len(overlap),
            "unique_final_test_task_count": len(final_ids),
        },
        "planned_evaluation_protocol": {
            "attempts_per_task": 1,
            "max_steps": 35,
            "temperature": 0.0,
            "top_p": 1.0,
            "reward_contract": "shopsimulator-reward-v3",
            "checkpoint_selection_uses_final_test": False,
            "paired_sft_grpo_evaluation": True,
        },
        "output_sha256": sha256_file(args.output),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
