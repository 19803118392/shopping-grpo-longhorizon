#!/usr/bin/env python3
"""把冻结的 GRPO task_id 转为 veRL 所需 parquet prompt 数据。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shopping_grpo.grpo_tasks import read_jsonl, sha256_file
from shopping_grpo.shop_http_env import ShopAgentEnv
from shopping_grpo.teacher_rollout import SYSTEM_PROMPT


def build_verl_record(task_id: int, user_instruction: str, split: str, index: int) -> dict:
    """只写模型本应看到的 system、用户需求和 task_id；不写 goal/reward。"""
    return {
        "data_source": "shopsimulator",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(user_instruction)},
        ],
        "ability": "shopping",
        "reward_model": {"style": "rule", "ground_truth": None},
        "extra_info": {
            "split": split,
            "index": int(index),
            "task_id": int(task_id),
        },
    }


def fetch_user_instruction(
    task_id: int,
    base_url: str,
    timeout: int,
    required_environment_version: str | None = None,
) -> str:
    """通过 reset 获取真实用户 query，并在 finally 中归还 probe 租约。"""
    with ShopAgentEnv(base_url=base_url, timeout=timeout) as env:
        initial = env.reset(task_id)
        if (
            required_environment_version is not None
            and initial.get("environment_version") != required_environment_version
        ):
            raise RuntimeError(
                "ShopSimulator environment version mismatch: "
                f"expected {required_environment_version!r}, "
                f"got {initial.get('environment_version')!r}"
            )
        return str(initial.get("instruction", initial.get("observation", "")))


def parse_args():
    parser = argparse.ArgumentParser(description="生成 veRL ShopSimulator GRPO parquet 数据集")
    parser.add_argument("--tasks", type=Path, required=True, help="grpo_train_v1.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="输出 parquet，例如 data/verl/grpo_train_v1.parquet")
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--split", default="train")
    parser.add_argument("--required-environment-version")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--reward-contract")
    parser.add_argument("--source-probe", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("缺少 pyarrow；请在 veRL 环境执行：uv pip install pyarrow") from exc
    rows = []
    for index, task in enumerate(read_jsonl(args.tasks)):
        task_id = int(task["task_id"])
        instruction = fetch_user_instruction(
            task_id,
            args.base_url,
            args.timeout,
            required_environment_version=args.required_environment_version,
        )
        if not instruction:
            raise SystemExit(f"task_id={task_id} reset 没有返回用户需求，停止生成数据集")
        rows.append(build_verl_record(task_id, instruction, args.split, index))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output)
    if args.metadata:
        payload = {
            "reward_contract": args.reward_contract,
            "required_environment_version": args.required_environment_version,
            "split": args.split,
            "row_count": len(rows),
            "source_tasks": str(args.tasks),
            "source_tasks_sha256": sha256_file(args.tasks),
            "source_probe": str(args.source_probe) if args.source_probe else None,
            "source_probe_sha256": (
                sha256_file(args.source_probe) if args.source_probe else None
            ),
            "output": str(args.output),
            "output_sha256": sha256_file(args.output),
            "schema_columns": list(pa.Table.from_pylist(rows).column_names),
        }
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"veRL parquet 已写入 {args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()
