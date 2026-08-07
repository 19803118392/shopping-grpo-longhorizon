#!/usr/bin/env python3
"""Freeze commit/config/checkpoint/model hashes before an authorized Final-200."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--served-model-name",
        required=True,
        help="exact vLLM served-model-name used by the evaluator",
    )
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        required=True,
        help="complete repeated paired validation report used before freezing",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    if not path.is_dir():
        raise SystemExit(f"artifact directory does not exist: {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise SystemExit(f"artifact directory is empty: {path}")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.output.exists():
        raise SystemExit("freeze output must be new")
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status.strip():
        raise SystemExit("refusing to freeze Final candidate from a dirty worktree")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config = args.config.resolve()
    if not config.is_file():
        raise SystemExit(f"config does not exist: {config}")
    evaluation_report = args.evaluation_report.resolve()
    try:
        report = json.loads(evaluation_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid repeated evaluation report: {exc}") from exc
    if report.get("schema_version") != "shopping-paired-statistics-v1":
        raise SystemExit("Final freeze requires a paired repeated evaluation report")
    for side in ("baseline", "candidate"):
        if float((report.get(side) or {}).get("attempt_coverage_rate", 0.0)) != 1.0:
            raise SystemExit("Final freeze requires 100% validation attempt coverage")
        profile = (report.get("failure_profiles") or {}).get(side) or {}
        if int(profile.get("infrastructure_invalid_attempts", -1)) != 0:
            raise SystemExit("Final freeze forbids infrastructure-invalid validation attempts")
        if int(profile.get("critical_footer_failures", -1)) != 0:
            raise SystemExit("Final freeze forbids critical footer failures")
    manifest = {
        "schema_version": "shopping-final-candidate-v1",
        "label": args.label,
        "git_commit": commit,
        "config_path": str(config),
        "config_sha256": _sha256_file(config),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256_tree(args.checkpoint.resolve()),
        "model_path": str(args.model.resolve()),
        "model_sha256": _sha256_tree(args.model.resolve()),
        "served_model_name": args.served_model_name,
        "evaluation_report_path": str(evaluation_report),
        "evaluation_report_sha256": _sha256_file(evaluation_report),
        "final_benchmark": "data/evaluation/tasks.jsonl",
        "final_benchmark_sha256": _sha256_file(ROOT / "data/evaluation/tasks.jsonl"),
        "final_protocol": {
            "tasks": 200,
            "attempts_per_task": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_steps": 35,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
