#!/usr/bin/env python3
"""Create a reproducible paired statistics report for two rollout JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from shopping_grpo.evaluation.reporting import render_markdown_report, render_overall_csv
from shopping_grpo.evaluation.rollout import load_tasks
from shopping_grpo.evaluation.statistics import compare_repeated_runs
from shopping_grpo.evaluation.stratification import build_stratified_comparison


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare baseline and candidate Shopping Agent rollouts"
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts-per-task", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
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
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")

    tasks = load_tasks(args.benchmark)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    selected_task_ids = [task["task_id"] for task in tasks]
    selected_task_id_set = set(selected_task_ids)
    baseline_rows = [
        row for row in read_jsonl(args.baseline) if int(row["task_id"]) in selected_task_id_set
    ]
    candidate_rows = [
        row for row in read_jsonl(args.candidate) if int(row["task_id"]) in selected_task_id_set
    ]
    report = compare_repeated_runs(
        expected_task_ids=selected_task_ids,
        baseline_trajectories=baseline_rows,
        candidate_trajectories=candidate_rows,
        attempts_per_task=args.attempts_per_task,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    report["labels"] = {
        "baseline": args.baseline_label,
        "candidate": args.candidate_label,
    }
    report["stratified_statistics"] = build_stratified_comparison(
        benchmark_tasks=tasks,
        baseline_trajectories=baseline_rows,
        candidate_trajectories=candidate_rows,
        attempts_per_task=args.attempts_per_task,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    report["provenance"] = {
        "benchmark_sha256": sha256_file(args.benchmark),
        "baseline_sha256": sha256_file(args.baseline),
        "candidate_sha256": sha256_file(args.candidate),
        "attempts_per_task": args.attempts_per_task,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "task_limit": args.limit,
        "selected_task_ids_sha256": hashlib.sha256(
            json.dumps(selected_task_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown_report(report), encoding="utf-8")
    if args.csv_output is not None:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        args.csv_output.write_text(render_overall_csv(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
