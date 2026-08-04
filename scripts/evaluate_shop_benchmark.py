#!/usr/bin/env python3
"""在固定 ShopSimulator benchmark 上评测 OpenAI-compatible 本地或远端模型。"""

import argparse
import json
from pathlib import Path

from shopping_grpo.evaluation.rollout import OpenAIChatClient, collect_tasks, load_tasks
from shopping_grpo.evaluation.statistics import summarize_repeated_run
from shopping_grpo.evaluation.summary import summarize_trajectories


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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
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
    parser.add_argument("--observation-detail-token-budget", type=int, default=4096)
    parser.add_argument("--observation-generic-token-budget", type=int, default=768)
    parser.add_argument("--observation-search-top-k", type=int, default=20)
    parser.add_argument(
        "--evidence-memory",
        action="store_true",
        help="启用仅由公开 observation v2 构建的有界候选证据账本。",
    )
    parser.add_argument("--evidence-memory-max-candidates", type=int, default=5)
    parser.add_argument("--evidence-memory-max-chars", type=int, default=2_000)
    return parser.parse_args()


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    args = parse_args()
    if args.max_steps < 1:
        raise SystemExit("--max-steps 必须为正数")
    if args.attempts_per_task < 1:
        raise SystemExit("--attempts-per-task 必须为正数")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens 必须为正数")
    if args.context_window <= args.max_tokens + args.context_safety_margin:
        raise SystemExit("--context-window 必须大于 --max-tokens 与安全余量之和")
    tasks = load_tasks(args.benchmark)
    client = OpenAIChatClient(
        model=args.model,
        base_url=args.llm_base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
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
    reference_trajectories = [
        row for row in trajectories if int(row.get("attempt_index", 0)) == 0
    ]
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
        "deterministic_reference_attempt": 0,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
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
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
