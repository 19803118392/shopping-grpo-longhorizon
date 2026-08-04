#!/usr/bin/env python3
"""Plan or materialize cumulative short/medium/long GRPO training datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from shopping_grpo.training.grpo.curriculum import build_curriculum_plan


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_plan(plan: dict) -> dict:
    visible = dict(plan)
    visible["stages"] = [
        {key: value for key, value in stage.items() if key != "task_ids"}
        for stage in plan["stages"]
    ]
    return visible


def main():
    args = parse_args()
    plan = build_curriculum_plan(_read_jsonl(args.metadata))
    plan["source_parquet_sha256"] = _sha256(args.source_parquet)
    plan["metadata_sha256"] = _sha256(args.metadata)
    if args.dry_run:
        print(json.dumps(_display_plan(plan), ensure_ascii=False, indent=2))
        return

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("materialization requires pyarrow") from exc
    from shopping_grpo.training.grpo.curriculum import select_parquet_rows

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("--output-dir must be new or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = pq.read_table(args.source_parquet)
    for stage in plan["stages"]:
        output = args.output_dir / f"{stage['name']}.parquet"
        selected = select_parquet_rows(source, stage["task_ids"])
        pq.write_table(selected, output, compression="zstd")
        stage["parquet"] = output.name
        stage["parquet_sha256"] = _sha256(output)
    manifest = args.output_dir / "manifest.json"
    manifest.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_display_plan(plan), ensure_ascii=False))


if __name__ == "__main__":
    main()
