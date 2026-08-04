#!/usr/bin/env python3
"""Create a reproducible paired statistics report for two rollout JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from shopping_grpo.evaluation.rollout import load_tasks
from shopping_grpo.evaluation.statistics import compare_repeated_runs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare baseline and candidate Shopping Agent rollouts"
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts-per-task", type=int, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.attempts_per_task < 1:
        raise SystemExit("--attempts-per-task must be positive")
    if args.bootstrap_samples < 1:
        raise SystemExit("--bootstrap-samples must be positive")

    tasks = load_tasks(args.benchmark)
    report = compare_repeated_runs(
        expected_task_ids=[task["task_id"] for task in tasks],
        baseline_trajectories=read_jsonl(args.baseline),
        candidate_trajectories=read_jsonl(args.candidate),
        attempts_per_task=args.attempts_per_task,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    report["labels"] = {
        "baseline": args.baseline_label,
        "candidate": args.candidate_label,
    }
    report["provenance"] = {
        "benchmark_sha256": sha256_file(args.benchmark),
        "baseline_sha256": sha256_file(args.baseline),
        "candidate_sha256": sha256_file(args.candidate),
        "attempts_per_task": args.attempts_per_task,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
