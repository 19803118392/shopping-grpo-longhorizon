#!/usr/bin/env python3
"""Build and audit the isolated Reward v3 / fresh-v1 GRPO task generation."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import Counter
from pathlib import Path

from shopping_grpo.grpo_tasks import (
    ELIGIBLE_PROBE_STATUSES,
    LENGTH_BUCKETS,
    build_grpo_candidate_manifest,
    length_bucket,
    read_jsonl,
    select_stratified_grpo_tasks,
    sha256_file,
    task_ids,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "environment-v2.1/reward-v3/fresh-v1"
ENVIRONMENT_VERSION = "shopsimulator-environment-v2.1"
REWARD_VERSION = "shopsimulator-reward-v3"
FRESH_ROOT = ROOT / "outputs/environment_v2_1_teacher_flash_fresh_v1_aggregate_604_20260727"
DEFAULT_CANDIDATES = ROOT / "data/splits/grpo_reward_v3_fresh_v1_probe_pool.jsonl"
DEFAULT_PROBE = ROOT / "outputs/grpo_reward_v3_fresh_v1_probe/raw.jsonl"
DEFAULT_TRAIN = ROOT / "data/splits/grpo_reward_v3_fresh_v1_train.jsonl"
DEFAULT_VAL = ROOT / "data/splits/grpo_reward_v3_fresh_v1_val.jsonl"
DEFAULT_EXCLUSIONS = (
    FRESH_ROOT / "raw.jsonl",
    ROOT / "data/benchmarks/shop_benchmark_v1.jsonl",
    ROOT / "data/benchmarks/shop_benchmark_v2_50.jsonl",
    ROOT / "outputs/flash_accepted_500_parallel/raw.jsonl.gz",
    ROOT / "data/splits/grpo_probe_pool_v1.jsonl",
)


def repository_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def metadata_path(path: Path) -> Path:
    return path.with_suffix(".metadata.json")


def refuse_overwrite(paths: list[Path], force: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not force:
        raise SystemExit(
            "refusing to overwrite frozen Reward v3 asset(s): "
            + ", ".join(existing)
            + "; pass --force only after auditing the replacement"
        )


def write_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def exclusion_report(paths=DEFAULT_EXCLUSIONS) -> tuple[set[int], list[dict]]:
    excluded: set[int] = set()
    report = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"required exclusion source is missing: {path}")
        ids = task_ids(read_jsonl(path))
        excluded.update(ids)
        report.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
                "task_count": len(ids),
            }
        )
    return excluded, report


def available_probe_buckets(candidate_rows: list[dict], probes: list[dict]) -> dict[str, int]:
    candidates = task_ids(candidate_rows)
    grouped = Counter()
    seen = set()
    for trajectory in probes:
        task_id = int(trajectory.get("task_id", -1))
        if task_id not in candidates or task_id in seen:
            continue
        seen.add(task_id)
        if str(trajectory.get("status", "unknown")) not in ELIGIBLE_PROBE_STATUSES:
            continue
        grouped[length_bucket(len(trajectory.get("steps") or []))] += 1
    return {bucket: grouped[bucket] for bucket in LENGTH_BUCKETS}


def proportional_targets(available: dict[str, int], total: int) -> dict[str, int]:
    total = int(total)
    population = sum(int(available.get(bucket, 0)) for bucket in LENGTH_BUCKETS)
    if total < 1:
        raise ValueError("target size must be positive")
    if population < total:
        raise ValueError(
            f"eligible probe population {population} is smaller than target {total}"
        )
    exact = {
        bucket: total * int(available.get(bucket, 0)) / population
        for bucket in LENGTH_BUCKETS
    }
    targets = {bucket: int(exact[bucket]) for bucket in LENGTH_BUCKETS}
    remaining = total - sum(targets.values())
    order = sorted(
        LENGTH_BUCKETS,
        key=lambda bucket: (exact[bucket] - targets[bucket], available.get(bucket, 0)),
        reverse=True,
    )
    for bucket in order:
        if remaining <= 0:
            break
        if targets[bucket] < int(available.get(bucket, 0)):
            targets[bucket] += 1
            remaining -= 1
    if remaining:
        for bucket in LENGTH_BUCKETS:
            capacity = int(available.get(bucket, 0)) - targets[bucket]
            take = min(capacity, remaining)
            targets[bucket] += take
            remaining -= take
            if remaining == 0:
                break
    if remaining:
        raise ValueError("unable to allocate proportional bucket targets")
    return targets


def validate_probe_contract(candidate_rows: list[dict], probes: list[dict]) -> dict:
    candidates = task_ids(candidate_rows)
    seen = set()
    statuses = Counter()
    reward_types = Counter()
    for trajectory in probes:
        task_id = int(trajectory.get("task_id", -1))
        if task_id not in candidates:
            raise ValueError(f"probe contains task outside candidate pool: {task_id}")
        if task_id in seen:
            raise ValueError(f"probe contains duplicate task_id: {task_id}")
        seen.add(task_id)
        statuses[str(trajectory.get("status", "unknown"))] += 1
        initial = trajectory.get("initial_result") or {}
        if initial.get("environment_version") != ENVIRONMENT_VERSION:
            raise ValueError(
                f"task {task_id} did not run in {ENVIRONMENT_VERSION}"
            )
        if trajectory.get("done") is True:
            terminal = trajectory.get("terminal_result") or {}
            detail = terminal.get("reward_detail") or {}
            if detail.get("reward_version") != REWARD_VERSION:
                raise ValueError(f"task {task_id} terminal reward is not Reward v3")
            if terminal.get("termination_reason") != detail.get("termination_reason"):
                raise ValueError(f"task {task_id} has inconsistent termination reason")
            reward_types[str(detail.get("reward_type", "unknown"))] += 1
    return {
        "probe_task_count": len(seen),
        "status_counts": dict(sorted(statuses.items())),
        "reward_type_counts": dict(sorted(reward_types.items())),
    }


def common_metadata() -> dict:
    return {
        "contract": CONTRACT,
        "environment_version": ENVIRONMENT_VERSION,
        "reward_version": REWARD_VERSION,
        "shopping_grpo_commit": repository_head(),
        "fresh_train": {
            "path": str((FRESH_ROOT / "train/sft.jsonl").relative_to(ROOT)),
            "sha256": sha256_file(FRESH_ROOT / "train/sft.jsonl"),
            "task_count": len(task_ids(read_jsonl(FRESH_ROOT / "train/sft.jsonl"))),
        },
        "fresh_validation": {
            "path": str((FRESH_ROOT / "val/sft.jsonl").relative_to(ROOT)),
            "sha256": sha256_file(FRESH_ROOT / "val/sft.jsonl"),
            "task_count": len(task_ids(read_jsonl(FRESH_ROOT / "val/sft.jsonl"))),
        },
    }


def command_candidate(args) -> None:
    output = args.output.resolve()
    metadata = metadata_path(output)
    refuse_overwrite([output, metadata], args.force)
    all_tasks = read_jsonl(args.tasks)
    excluded, exclusions = exclusion_report()
    rows = build_grpo_candidate_manifest(
        task_ids(all_tasks),
        excluded,
        (),
        args.size,
        args.seed,
    )
    write_jsonl(output, rows)
    payload = {
        **common_metadata(),
        "asset": "grpo_reward_v3_fresh_v1_probe_pool",
        "task_count": len(rows),
        "seed": args.seed,
        "source_tasks": str(args.tasks.resolve().relative_to(ROOT)),
        "source_tasks_sha256": sha256_file(args.tasks),
        "selection": "deterministic_random_without_replacement",
        "exclusions": exclusions,
        "output_sha256": sha256_file(output),
    }
    write_metadata(metadata, payload)
    print(json.dumps(payload, ensure_ascii=False))


def command_select(args) -> None:
    output = args.output.resolve()
    metadata = metadata_path(output)
    refuse_overwrite([output, metadata], args.force)
    candidates = read_jsonl(args.candidates)
    probes = read_jsonl(args.probes)
    probe_report = validate_probe_contract(candidates, probes)
    available = available_probe_buckets(candidates, probes)
    targets = proportional_targets(available, args.train_size)
    rows, selection_report = select_stratified_grpo_tasks(
        candidates,
        probes,
        targets,
        args.seed,
    )
    write_jsonl(output, rows)
    payload = {
        **common_metadata(),
        "asset": "grpo_reward_v3_fresh_v1_train",
        "candidate_pool": str(args.candidates.resolve().relative_to(ROOT)),
        "candidate_pool_sha256": sha256_file(args.candidates),
        "probe_rollouts": str(args.probes.resolve().relative_to(ROOT)),
        "probe_rollouts_sha256": sha256_file(args.probes),
        "seed": args.seed,
        "selection": "proportional_to_fresh_policy_probe_length_distribution",
        "probe_contract": probe_report,
        **selection_report,
        "output_sha256": sha256_file(output),
    }
    write_metadata(metadata, payload)
    print(json.dumps(payload, ensure_ascii=False))


def command_validation(args) -> None:
    output = args.output.resolve()
    metadata = metadata_path(output)
    refuse_overwrite([output, metadata], args.force)
    candidates = read_jsonl(args.candidates)
    train_rows = read_jsonl(args.train)
    probes = read_jsonl(args.probes)
    validate_probe_contract(candidates, probes)
    train_ids = task_ids(train_rows)
    remaining = [row for row in candidates if int(row["task_id"]) not in train_ids]
    available = available_probe_buckets(remaining, probes)
    targets = proportional_targets(available, args.validation_size)
    rows, selection_report = select_stratified_grpo_tasks(
        remaining,
        probes,
        targets,
        args.seed,
    )
    random.Random(f"{args.seed}:validation-order").shuffle(rows)
    write_jsonl(output, rows)
    payload = {
        **common_metadata(),
        "asset": "grpo_reward_v3_fresh_v1_validation",
        "candidate_pool": str(args.candidates.resolve().relative_to(ROOT)),
        "candidate_pool_sha256": sha256_file(args.candidates),
        "train_split": str(args.train.resolve().relative_to(ROOT)),
        "train_split_sha256": sha256_file(args.train),
        "probe_rollouts": str(args.probes.resolve().relative_to(ROOT)),
        "probe_rollouts_sha256": sha256_file(args.probes),
        "seed": args.seed,
        "selection": "held_out_proportional_to_fresh_policy_probe_length_distribution",
        **selection_report,
        "output_sha256": sha256_file(output),
    }
    write_metadata(metadata, payload)
    print(json.dumps(payload, ensure_ascii=False))


def command_audit(args) -> None:
    candidates = task_ids(read_jsonl(args.candidates))
    train = task_ids(read_jsonl(args.train))
    validation = task_ids(read_jsonl(args.validation))
    excluded, exclusions = exclusion_report()
    probes = read_jsonl(args.probes)
    probe_report = validate_probe_contract(read_jsonl(args.candidates), probes)
    checks = {
        "train_validation_overlap": len(train & validation),
        "candidate_exclusion_overlap": len(candidates & excluded),
        "train_exclusion_overlap": len(train & excluded),
        "validation_exclusion_overlap": len(validation & excluded),
        "train_not_in_candidates": len(train - candidates),
        "validation_not_in_candidates": len(validation - candidates),
    }
    if any(checks.values()):
        raise SystemExit("Reward v3 GRPO isolation audit failed: " + json.dumps(checks))
    report = {
        **common_metadata(),
        "candidate_count": len(candidates),
        "train_count": len(train),
        "validation_count": len(validation),
        "checks": checks,
        "probe_contract": probe_report,
        "exclusions": exclusions,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    candidate = subparsers.add_parser("candidate")
    candidate.add_argument("--tasks", type=Path, default=ROOT / "data/shop_tasks.jsonl")
    candidate.add_argument("--output", type=Path, default=DEFAULT_CANDIDATES)
    candidate.add_argument("--size", type=int, default=2000)
    candidate.add_argument("--seed", type=int, default=20260728)
    candidate.add_argument("--force", action="store_true")
    candidate.set_defaults(func=command_candidate)

    select = subparsers.add_parser("select")
    select.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    select.add_argument("--probes", type=Path, default=DEFAULT_PROBE)
    select.add_argument("--output", type=Path, default=DEFAULT_TRAIN)
    select.add_argument("--train-size", type=int, default=1000)
    select.add_argument("--seed", type=int, default=20260728)
    select.add_argument("--force", action="store_true")
    select.set_defaults(func=command_select)

    validation = subparsers.add_parser("validation")
    validation.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    validation.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    validation.add_argument("--probes", type=Path, default=DEFAULT_PROBE)
    validation.add_argument("--output", type=Path, default=DEFAULT_VAL)
    validation.add_argument("--validation-size", type=int, default=50)
    validation.add_argument("--seed", type=int, default=20260729)
    validation.add_argument("--force", action="store_true")
    validation.set_defaults(func=command_validation)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    audit.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    audit.add_argument("--validation", type=Path, default=DEFAULT_VAL)
    audit.add_argument("--probes", type=Path, default=DEFAULT_PROBE)
    audit.set_defaults(func=command_audit)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
