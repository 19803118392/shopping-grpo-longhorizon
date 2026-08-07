#!/usr/bin/env python3
"""在固定 ShopSimulator benchmark 上评测 OpenAI-compatible 本地或远端模型。"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from shopping_grpo.evaluation.rollout import OpenAIChatClient, collect_tasks, load_tasks
from shopping_grpo.evaluation.statistics import summarize_repeated_run
from shopping_grpo.evaluation.summary import summarize_trajectories

ROOT = Path(__file__).resolve().parents[1]
FINAL_BENCHMARK = (ROOT / "data/evaluation/tasks.jsonl").resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="评测 Base、SFT 或 GRPO Shopping Agent")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="原始评测轨迹 JSONL")
    parser.add_argument("--summary", type=Path, required=True, help="汇总指标 JSON")
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--model", required=True)
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument("--api-key", required=True, help="本地 vLLM 可传 EMPTY")
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument(
        "--attempts-per-task",
        type=int,
        default=1,
        help="每个任务的独立 rollout 次数；重复运行会按 task_id/attempt_index 续跑。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只运行任务文件中稳定排序的前 N 个任务；用于非 Final-200 的开发集冒烟。",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--protocol",
        choices=("custom", "seed-replay", "memory-pilot", "dev50x3", "dev50x5"),
        default="custom",
    )
    parser.add_argument(
        "--final-200",
        action="store_true",
        help="authorize one frozen deterministic pass over data/evaluation/tasks.jsonl",
    )
    parser.add_argument("--frozen-artifact-manifest", type=Path)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="单次模型生成上限；防止未调用工具时耗尽完整上下文。",
    )
    parser.add_argument("--context-window", type=int, default=24576)
    parser.add_argument("--context-safety-margin", type=int, default=512)
    parser.add_argument(
        "--context-compaction",
        action="store_true",
        help="上下文接近上限时压缩较早的交互；默认关闭。",
    )
    parser.add_argument("--observation-token-budget", type=int, default=1536)
    parser.add_argument("--observation-detail-token-budget", type=int, default=2048)
    parser.add_argument("--observation-generic-token-budget", type=int, default=768)
    parser.add_argument("--observation-search-top-k", type=int, default=20)
    parser.add_argument(
        "--evidence-memory",
        action="store_true",
        help="启用仅由公开 observation v2 构建的有界候选证据账本。",
    )
    parser.add_argument("--evidence-memory-max-candidates", type=int, default=5)
    parser.add_argument("--evidence-memory-max-chars", type=int, default=384)
    return parser.parse_args()


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    if not path.is_dir():
        raise SystemExit(f"frozen artifact directory does not exist: {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise SystemExit(f"frozen artifact directory is empty: {path}")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_frozen_candidate(
    manifest_path: Path,
    *,
    served_model_name: str,
    benchmark: Path,
    root: Path = ROOT,
    git_commit: str | None = None,
) -> dict:
    """Re-hash the exact frozen candidate immediately before Final-200."""
    try:
        frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid frozen artifact manifest: {exc}") from exc
    required = {
        "git_commit",
        "config_path",
        "config_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "model_path",
        "model_sha256",
        "served_model_name",
        "evaluation_report_path",
        "evaluation_report_sha256",
        "final_benchmark_sha256",
    }
    if frozen.get("schema_version") != "shopping-final-candidate-v1":
        raise SystemExit("frozen artifact manifest has an unsupported schema")
    if not required.issubset(frozen) or any(not str(frozen[key]) for key in required):
        raise SystemExit("frozen artifact manifest is incomplete")
    if str(frozen["served_model_name"]) != str(served_model_name):
        raise SystemExit("evaluator --model differs from frozen served_model_name")
    evaluation_report = Path(frozen["evaluation_report_path"]).resolve()
    if (
        not evaluation_report.is_file()
        or _sha256_file(evaluation_report) != frozen["evaluation_report_sha256"]
    ):
        raise SystemExit("frozen repeated evaluation report hash mismatch")
    try:
        report = json.loads(evaluation_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid frozen repeated evaluation report: {exc}") from exc
    if report.get("schema_version") != "shopping-paired-statistics-v1":
        raise SystemExit("frozen repeated evaluation report has an unsupported schema")
    actual_commit = git_commit
    if actual_commit is None:
        actual_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    if actual_commit != frozen["git_commit"]:
        raise SystemExit("current Git commit differs from frozen Final candidate")
    config = Path(frozen["config_path"]).resolve()
    if not config.is_file() or _sha256_file(config) != frozen["config_sha256"]:
        raise SystemExit("frozen Final config hash mismatch")
    checkpoint = Path(frozen["checkpoint_path"]).resolve()
    if _sha256_tree(checkpoint) != frozen["checkpoint_sha256"]:
        raise SystemExit("frozen Final checkpoint hash mismatch")
    model = Path(frozen["model_path"]).resolve()
    if _sha256_tree(model) != frozen["model_sha256"]:
        raise SystemExit("frozen Final model hash mismatch")
    if _sha256_file(benchmark.resolve()) != frozen["final_benchmark_sha256"]:
        raise SystemExit("frozen Final benchmark hash mismatch")
    return frozen


def main():
    args = parse_args()
    if args.max_steps < 1:
        raise SystemExit("--max-steps 必须为正数")
    if args.attempts_per_task < 1:
        raise SystemExit("--attempts-per-task 必须为正数")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit 必须为正数")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens 必须为正数")
    if args.context_window <= args.max_tokens + args.context_safety_margin:
        raise SystemExit("--context-window 必须大于 --max-tokens 与安全余量之和")
    tasks = load_tasks(args.benchmark)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit("benchmark 没有可运行任务")
    is_final_benchmark = args.benchmark.resolve() == FINAL_BENCHMARK
    if is_final_benchmark and not args.final_200:
        raise SystemExit(
            "data/evaluation/tasks.jsonl is Final-200; pass --final-200 only after promotion"
        )
    if args.final_200:
        if not is_final_benchmark:
            raise SystemExit("--final-200 requires data/evaluation/tasks.jsonl")
        if args.frozen_artifact_manifest is None:
            raise SystemExit("--final-200 requires --frozen-artifact-manifest")
        validate_frozen_candidate(
            args.frozen_artifact_manifest,
            served_model_name=args.model,
            benchmark=args.benchmark,
        )
        final_protocol = (
            len(tasks),
            args.attempts_per_task,
            args.temperature,
            args.top_p,
            args.max_steps,
            args.limit,
        )
        if final_protocol != (200, 1, 0.0, 1.0, 35, None):
            raise SystemExit(
                "Final-200 requires 200 tasks, one attempt, temperature=0, top_p=1, max_steps=35"
            )
        if (args.output.exists() and args.output.stat().st_size) or args.summary.exists():
            raise SystemExit("Final-200 output and summary must be new")
    protocol_expectations = {
        "seed-replay": (5, 2),
        "memory-pilot": (20, 2),
        "dev50x3": (50, 3),
        "dev50x5": (50, 5),
    }
    if args.protocol != "custom":
        expected_tasks, expected_attempts = protocol_expectations[args.protocol]
        actual = (
            len(tasks),
            args.attempts_per_task,
            args.temperature,
            args.top_p,
            args.max_steps,
            args.seed,
        )
        expected = (expected_tasks, expected_attempts, 0.7, 0.9, 35, 2026)
        if actual != expected:
            raise SystemExit(
                f"protocol {args.protocol} requires tasks/attempts/temperature/top_p/"
                f"max_steps/seed={expected}, got {actual}"
            )
    client = OpenAIChatClient(
        model=args.model,
        base_url=args.llm_base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        context_window=args.context_window,
        context_safety_margin=args.context_safety_margin,
        context_compaction_enable=args.context_compaction,
        observation_token_budget=args.observation_token_budget,
        observation_detail_token_budget=args.observation_detail_token_budget,
        observation_generic_token_budget=args.observation_generic_token_budget,
        observation_search_top_k=args.observation_search_top_k,
        evidence_memory_enable=args.evidence_memory,
        evidence_memory_max_candidates=args.evidence_memory_max_candidates,
        evidence_memory_max_chars=args.evidence_memory_max_chars,
    )
    collect_tasks(
        tasks,
        client=client,
        output_path=args.output,
        base_url=args.base_url,
        max_steps=args.max_steps,
        attempts_per_task=args.attempts_per_task,
    )
    trajectories = _read_jsonl(args.output)
    expected_task_ids = [task["task_id"] for task in tasks]
    reference_trajectories = [row for row in trajectories if int(row.get("attempt_index", 0)) == 0]
    summary = summarize_trajectories(expected_task_ids, reference_trajectories)
    summary["repeated_sampling"] = summarize_repeated_run(
        expected_task_ids=expected_task_ids,
        trajectories=trajectories,
        attempts_per_task=args.attempts_per_task,
    )
    summary["protocol"] = {
        "benchmark": str(args.benchmark),
        "model": args.model,
        "reward_contract": "shopsimulator-reward-v3",
        "max_steps": args.max_steps,
        "attempts_per_task": args.attempts_per_task,
        "task_limit": args.limit,
        "selected_task_ids_sha256": hashlib.sha256(
            json.dumps(expected_task_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "deterministic_reference_attempt": 0,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "protocol_name": args.protocol,
        "final_200": bool(args.final_200),
        "frozen_artifact_manifest": (
            str(args.frozen_artifact_manifest.resolve()) if args.frozen_artifact_manifest else None
        ),
        "frozen_artifact_manifest_sha256": (
            _sha256_file(args.frozen_artifact_manifest) if args.frozen_artifact_manifest else None
        ),
        "context_window": args.context_window,
        "context_safety_margin": args.context_safety_margin,
        "context_compaction": args.context_compaction,
        "observation_token_budget": args.observation_token_budget,
        "observation_detail_token_budget": args.observation_detail_token_budget,
        "observation_generic_token_budget": args.observation_generic_token_budget,
        "observation_search_top_k": args.observation_search_top_k,
        "evidence_memory": args.evidence_memory,
        "evidence_memory_max_candidates": args.evidence_memory_max_candidates,
        "evidence_memory_max_chars": args.evidence_memory_max_chars,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
