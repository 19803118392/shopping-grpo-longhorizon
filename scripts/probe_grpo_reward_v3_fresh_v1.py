#!/usr/bin/env python3
"""Resumable concurrent probe for Reward v3 GRPO task stratification."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from shopping_grpo.grpo_tasks import read_jsonl, sha256_file, task_ids
from shopping_grpo.teacher_rollout import (
    CollectionInfrastructureError,
    OpenAIChatClient,
    _is_infrastructure_failure,
    append_jsonl,
    collect_for_task,
    completed_task_attempts,
    load_tasks,
    rollout_interrupted,
)


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_VERSION = "shopsimulator-environment-v2.1"
REWARD_VERSION = "shopsimulator-reward-v3"
CONTEXT_HARD_LIMIT_STATUS = "context_hard_limit_exceeded"
DEFAULT_TASKS = ROOT / "data/splits/grpo_reward_v3_fresh_v1_probe_pool.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/grpo_reward_v3_fresh_v1_probe/raw.jsonl"


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def repository_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def validate_trajectory_contract(trajectory: dict) -> None:
    task_id = int(trajectory.get("task_id", -1))
    initial = trajectory.get("initial_result") or {}
    if initial.get("environment_version") != ENVIRONMENT_VERSION:
        raise ValueError(
            f"task {task_id}: expected {ENVIRONMENT_VERSION}, "
            f"got {initial.get('environment_version')!r}"
        )
    if trajectory.get("done") is not True:
        return
    terminal = trajectory.get("terminal_result") or {}
    detail = terminal.get("reward_detail") or {}
    if detail.get("reward_version") != REWARD_VERSION:
        raise ValueError(
            f"task {task_id}: terminal result is not {REWARD_VERSION}"
        )
    if terminal.get("termination_reason") != detail.get("termination_reason"):
        raise ValueError(f"task {task_id}: inconsistent termination reason")
    if detail.get("reward_type") != detail.get("termination_reason"):
        raise ValueError(f"task {task_id}: reward_type does not match termination")
    if not isinstance(detail.get("reward_valid"), bool):
        raise ValueError(f"task {task_id}: reward_valid is not boolean")
    if not isinstance(detail.get("purchase_success"), bool):
        raise ValueError(f"task {task_id}: purchase_success is not boolean")


def normalize_expected_limit_status(trajectory: dict) -> dict:
    """Match the non-compacting veRL AgentLoop's context-limit termination."""
    error = trajectory.get("error") or {}
    if (
        trajectory.get("status") == "error"
        and error.get("type") == "ContextBudgetError"
    ):
        trajectory["status"] = CONTEXT_HARD_LIMIT_STATUS
        trajectory["termination_reason"] = CONTEXT_HARD_LIMIT_STATUS
        trajectory["infrastructure_invalid"] = True
    return trajectory


def existing_probe_ids(output: Path, expected: set[int]) -> set[int]:
    if not output.is_file():
        return set()
    rows = read_jsonl(output)
    ids = []
    for row in rows:
        validate_trajectory_contract(row)
        task_id = int(row["task_id"])
        if task_id not in expected:
            raise ValueError(f"existing probe has task outside candidate pool: {task_id}")
        ids.append(task_id)
    if len(ids) != len(set(ids)):
        raise ValueError("existing probe contains duplicate task_id")
    return set(ids)


def write_probe_metadata(args, expected_ids: set[int]) -> dict:
    rows = read_jsonl(args.output) if args.output.is_file() else []
    for row in rows:
        validate_trajectory_contract(row)
    status_counts = Counter(str(row.get("status", "unknown")) for row in rows)
    reward_types = Counter()
    for row in rows:
        if row.get("done") is True:
            detail = ((row.get("terminal_result") or {}).get("reward_detail") or {})
            reward_types[str(detail.get("reward_type", "unknown"))] += 1
    payload = {
        "contract": "environment-v2.1/reward-v3/fresh-v1",
        "environment_version": ENVIRONMENT_VERSION,
        "reward_version": REWARD_VERSION,
        "shopping_grpo_commit": repository_head(),
        "candidate_pool": display_path(args.tasks),
        "candidate_pool_sha256": sha256_file(args.tasks),
        "candidate_task_count": len(expected_ids),
        "completed_task_count": len(rows),
        "complete": len(rows) == len(expected_ids),
        "raw_sha256": sha256_file(args.output) if args.output.is_file() else None,
        "status_counts": dict(sorted(status_counts.items())),
        "infrastructure_invalid_count": sum(
            bool(row.get("infrastructure_invalid")) for row in rows
        ),
        "reward_type_counts": dict(sorted(reward_types.items())),
        "protocol": {
            "model": args.model,
            "model_path": str(args.model_path),
            "model_manifest_sha256": sha256_file(args.model_path / "merge_manifest.json"),
            "temperature": 0.0,
            "top_p": 1.0,
            "max_steps": 35,
            "max_tokens": 512,
            "context_window": 24576,
            "context_safety_margin": 512,
            "context_compaction": False,
            "context_hard_limit_status": CONTEXT_HARD_LIMIT_STATUS,
            "observation_token_budget": 1536,
            "observation_detail_token_budget": 4096,
            "observation_generic_token_budget": 768,
            "observation_search_top_k": 20,
            "workers": args.workers,
        },
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def collect(args) -> int:
    tasks = load_tasks(args.tasks)
    expected_ids = task_ids(tasks)
    existing = existing_probe_ids(args.output, expected_ids)
    remaining = [task for task in tasks if int(task["task_id"]) not in existing]
    if not remaining:
        print(json.dumps(write_probe_metadata(args, expected_ids), ensure_ascii=False))
        return 0

    def collect_one(task):
        # OpenAIChatClient keeps per-trajectory context diagnostics, so each
        # concurrent rollout gets an independent client instance.
        client = OpenAIChatClient(
            model=args.model,
            base_url=args.llm_base_url,
            api_key=args.api_key,
            temperature=0.0,
            top_p=1.0,
            timeout=args.timeout,
            max_tokens=512,
            context_window=24576,
            context_safety_margin=512,
            context_compaction_enable=False,
            observation_token_budget=1536,
            observation_detail_token_budget=4096,
            observation_generic_token_budget=768,
            observation_search_top_k=20,
        )
        return collect_for_task(
            task,
            client=client,
            base_url=args.base_url,
            max_steps=35,
            attempt_index=0,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = {}
    task_iter = iter(remaining)
    failure = None
    completed_count = len(existing)

    def submit(executor):
        while len(pending) < args.workers:
            try:
                task = next(task_iter)
            except StopIteration:
                return
            future = executor.submit(collect_one, task)
            pending[future] = int(task["task_id"])

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        submit(executor)
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                task_id = pending.pop(future)
                trajectory = normalize_expected_limit_status(future.result())
                append_jsonl(args.output, [trajectory])
                completed_count += 1
                try:
                    validate_trajectory_contract(trajectory)
                except ValueError as exc:
                    failure = f"Reward v3 contract failure after task {task_id}: {exc}"
                if trajectory.get("status") == "error":
                    error = trajectory.get("error") or {}
                    failure = (
                        f"probe error after task {task_id}: "
                        f"{error.get('type', 'unknown')}: {error.get('message', '')}"
                    )
                elif _is_infrastructure_failure(trajectory):
                    failure = f"infrastructure failure after task {task_id}"
            print(
                f"probe_progress={completed_count}/{len(expected_ids)} "
                f"pending={len(pending)}",
                flush=True,
            )
            if failure:
                continue
            submit(executor)
    payload = write_probe_metadata(args, expected_ids)
    print(json.dumps(payload, ensure_ascii=False))
    if failure:
        raise CollectionInfrastructureError(failure)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_OUTPUT.parent / "probe.metadata.json",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/root/autodl-tmp/checkpoints/qwen35-2b-sft-v1-fresh-merged"),
    )
    parser.add_argument("--model", default="qwen35-2b-sft-v1-fresh")
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1 or args.workers > 8:
        raise SystemExit("--workers must be in [1, 8] for the frozen 8-slot environment")
    if completed_task_attempts(args.output) - {
        (task_id, 0) for task_id in task_ids(read_jsonl(args.tasks))
    }:
        raise SystemExit("probe output contains unexpected task attempts")
    signal.signal(signal.SIGTERM, rollout_interrupted)
    signal.signal(signal.SIGINT, rollout_interrupted)
    try:
        return collect(args)
    except CollectionInfrastructureError as exc:
        print(f"probe stopped: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
