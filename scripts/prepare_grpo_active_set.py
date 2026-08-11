#!/usr/bin/env python3
"""Screen training tasks and materialize a reward-varying GRPO active set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from shopping_grpo.evaluation.stratification import query_difficulty_features

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATION_TASKS = (
    ROOT / "data/grpo/validation.jsonl",
    ROOT / "data/evaluation/tasks.jsonl",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(seed: int, task_id: int) -> str:
    return hashlib.sha256(f"{seed}:grpo-active-set:{task_id}".encode()).hexdigest()


def _query(row: dict) -> str:
    for message in row["prompt"]:
        if message["role"] == "user":
            return str(message["content"])
    raise ValueError(f"task_id={row['extra_info']['task_id']} has no user Query")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_screen(
    train_parquet: Path,
    train_metadata: Path,
    output: Path,
    *,
    screen_size: int,
    seed: int,
) -> dict:
    if screen_size < 1:
        raise ValueError("screen_size must be at least 1")
    rows = pq.read_table(train_parquet).to_pylist()
    metadata = {}
    for row in _read_jsonl(train_metadata):
        task_id = int(row["task_id"])
        if task_id in metadata:
            raise ValueError(f"duplicate task_id={task_id} in training metadata")
        metadata[task_id] = row
    candidates = []
    train_ids = set()
    for row in rows:
        task_id = int(row["extra_info"]["task_id"])
        if task_id in train_ids:
            raise ValueError(f"duplicate task_id={task_id} in training parquet")
        train_ids.add(task_id)
        if task_id not in metadata:
            raise ValueError(f"task_id={task_id} is missing from training metadata")
        features = query_difficulty_features(_query(row))
        candidates.append((row, metadata[task_id], features))

    buckets = ("short", "medium", "long")
    per_bucket, remainder = divmod(screen_size, len(buckets))
    selected = []
    for bucket_index, bucket in enumerate(buckets):
        target = per_bucket + int(bucket_index < remainder)
        pool = [item for item in candidates if item[1].get("length_bucket") == bucket]
        pool.sort(
            key=lambda item: (
                item[2]["constraint_count_bucket"] != "4+",
                not item[2]["has_option_selection"],
                -int(item[2]["constraint_count"]),
                _stable_hash(seed, int(item[0]["extra_info"]["task_id"])),
            )
        )
        if len(pool) < target:
            raise ValueError(f"length bucket {bucket!r} has only {len(pool)} rows")
        selected.extend(pool[:target])

    benchmark = []
    feature_rows = []
    for row, meta, features in selected:
        task_id = int(row["extra_info"]["task_id"])
        benchmark.append({**meta, "task_id": task_id})
        feature_rows.append({**features, **meta, "task_id": task_id})
    _write_jsonl(output, benchmark)
    return {
        "schema_version": "shopping-grpo-active-screen-v1",
        "seed": seed,
        "screen_size": len(benchmark),
        "task_ids": [row["task_id"] for row in benchmark],
        "length_buckets": dict(Counter(row["length_bucket"] for row in benchmark)),
        "constraint_buckets": dict(Counter(row["constraint_count_bucket"] for row in feature_rows)),
        "option_tasks": sum(row["has_option_selection"] for row in feature_rows),
        "train_parquet_sha256": _sha256_file(train_parquet),
        "train_metadata_sha256": _sha256_file(train_metadata),
        "screen_sha256": _sha256_file(output),
    }


def _optimization_reward(row: dict) -> float:
    terminal = row.get("terminal_result") or {}
    reward = terminal.get("reward_detail") or {}
    if "reward" not in reward:
        raise ValueError(f"task_id={row.get('task_id')} is missing Reward v3 scalar")
    value = reward["reward"]
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"task_id={row.get('task_id')} has a non-finite reward")
    if "final_reward" in row and not math.isclose(
        value, float(row["final_reward"]), rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(f"task_id={row.get('task_id')} has inconsistent reward scalars")
    return value


def _task_ids(path: Path) -> set[int]:
    return {int(row["task_id"]) for row in _read_jsonl(path)}


def materialize_active_set(
    train_parquet: Path,
    screening: Path,
    output: Path,
    *,
    attempts_per_task: int,
    tolerance: float,
    evaluation_task_files: tuple[Path, ...] = (),
) -> dict:
    if attempts_per_task < 1:
        raise ValueError("attempts_per_task must be at least 1")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")

    source = pq.read_table(train_parquet)
    source_rows = source.to_pylist()
    source_by_id = {}
    for row in source_rows:
        task_id = int(row["extra_info"]["task_id"])
        if task_id in source_by_id:
            raise ValueError(f"duplicate task_id={task_id} in training parquet")
        source_by_id[task_id] = row

    trajectories = _read_jsonl(screening)
    groups: dict[int, list[dict]] = defaultdict(list)
    seen_attempts = set()
    for row in trajectories:
        task_id = int(row["task_id"])
        if task_id not in source_by_id:
            raise ValueError(f"screening task_id={task_id} is absent from training parquet")
        if "attempt_index" not in row:
            raise ValueError(f"task_id={task_id} is missing attempt_index")
        attempt_index = int(row["attempt_index"])
        if not 0 <= attempt_index < attempts_per_task:
            raise ValueError(
                f"task_id={task_id} attempt_index={attempt_index} is outside "
                f"[0, {attempts_per_task})"
            )
        key = (task_id, attempt_index)
        if key in seen_attempts:
            raise ValueError(
                f"duplicate screening trajectory for task_id={task_id}, "
                f"attempt_index={attempt_index}"
            )
        seen_attempts.add(key)
        groups[task_id].append(row)

    selected_ids = []
    rejected = Counter()
    group_reports = []
    for task_id, rows in sorted(groups.items()):
        attempts = {int(row.get("attempt_index", 0)) for row in rows}
        if attempts != set(range(attempts_per_task)):
            rejected["incomplete_attempts"] += 1
            continue
        if any(row.get("status") != "done" for row in rows):
            rejected["non_terminal"] += 1
            continue
        details = [(row.get("terminal_result") or {}).get("reward_detail") or {} for row in rows]
        if any(detail.get("reward_valid") is not True for detail in details):
            rejected["reward_invalid"] += 1
            continue
        if any(detail.get("reward_version") != "shopsimulator-reward-v3" for detail in details):
            rejected["wrong_reward_contract"] += 1
            continue
        rewards = [_optimization_reward(row) for row in rows]
        varying = max(rewards) - min(rewards) > tolerance
        if varying:
            selected_ids.append(task_id)
        else:
            rejected["constant_reward"] += 1
        group_reports.append(
            {
                "task_id": task_id,
                "rewards": rewards,
                "varying": varying,
                "strict_success_rate": sum(
                    detail.get("reward_type") == "gold_purchase" for detail in details
                )
                / attempts_per_task,
            }
        )

    selected_id_set = set(selected_ids)
    selected = [row for task_id, row in source_by_id.items() if task_id in selected_id_set]
    if not selected:
        raise ValueError("screening produced no valid reward-varying task")

    evaluation_checks = []
    for path in evaluation_task_files:
        evaluation_ids = _task_ids(path)
        overlap = sorted(selected_id_set & evaluation_ids)
        if overlap:
            raise ValueError(
                f"active-set/evaluation overlap in {path.name}: {overlap[:10]}"
            )
        evaluation_checks.append(
            {
                "sha256": _sha256_file(path),
                "tasks": len(evaluation_ids),
                "overlap": 0,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    pq.write_table(pa.Table.from_pylist(selected, schema=source.schema), output)
    return {
        "schema_version": "shopping-grpo-active-set-v1",
        "attempts_per_task": attempts_per_task,
        "screened_groups": len(groups),
        "selected_groups": len(selected_ids),
        "screen_effective_ratio": len(selected_ids) / len(groups),
        "selected_task_ids": selected_ids,
        "rejected": dict(rejected),
        "groups": group_reports,
        "train_parquet_sha256": _sha256_file(train_parquet),
        "screening_sha256": _sha256_file(screening),
        "active_set_sha256": _sha256_file(output),
        "evaluation_checks": evaluation_checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", type=Path, default=Path("data/grpo/train.parquet"))
    parser.add_argument("--train-metadata", type=Path, default=Path("data/grpo/train.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--screen-size", type=int, default=48)
    parser.add_argument("--screening", type=Path)
    parser.add_argument("--attempts-per-task", type=int, default=4)
    parser.add_argument("--reward-tolerance", type=float, default=1.0e-8)
    parser.add_argument(
        "--evaluation-task-file",
        action="append",
        type=Path,
        dest="evaluation_task_files",
        help=(
            "task file that must have zero overlap with the active set; may be repeated. "
            "Defaults to Validation-50 and Final-200."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.screening is None:
        result = build_screen(
            args.train_parquet,
            args.train_metadata,
            args.output,
            screen_size=args.screen_size,
            seed=args.seed,
        )
    else:
        result = materialize_active_set(
            args.train_parquet,
            args.screening,
            args.output,
            attempts_per_task=args.attempts_per_task,
            tolerance=args.reward_tolerance,
            evaluation_task_files=tuple(args.evaluation_task_files or DEFAULT_EVALUATION_TASKS),
        )
    if args.manifest.exists():
        raise FileExistsError(f"refusing to overwrite {args.manifest}")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
