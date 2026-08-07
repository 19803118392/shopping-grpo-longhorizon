#!/usr/bin/env python3
"""Periodically snapshot the health and progress of a persisted training run."""

from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


STEP_RE = re.compile(r"training/global_step:(\d+)(?!\d)")
UPDATED_STEP_RE = re.compile(
    r"training/global_step:(\d+)(?!\d)[^\n]*training/optimizer_updated:1"
)
SKIPPED_TOTAL_RE = re.compile(r"skipped_updates_total[\":= ]+(\d+)")
CONSECUTIVE_SKIP_RE = re.compile(r"consecutive_skips[\":= ]+(\d+)")
ERROR_PATTERNS = {
    "traceback": re.compile(r"Traceback"),
    "cuda_oom": re.compile(r"CUDA out of memory", re.IGNORECASE),
    "dynamic_sampling_abort": re.compile(r"SHOPPING_GRPO_DYNAMIC_SAMPLING_ABORT"),
    "reward_invalid": re.compile(r"reward_valid[\"'=:\s]+false", re.IGNORECASE),
    "footer_failure": re.compile(r"footer failure", re.IGNORECASE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-glob", required=True, help="glob matching one or more stage logs")
    parser.add_argument("--output", type=Path, required=True, help="atomic current-state JSON")
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--screen-name", action="append", default=[])
    parser.add_argument("--interval-seconds", type=int, default=1200)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.target_step <= 0:
        parser.error("--target-step must be positive")
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    return args


def _run(command: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return result.returncode, (result.stdout + result.stderr).strip()


def _screen_state(names: list[str]) -> dict[str, bool]:
    _, listing = _run(["screen", "-ls"])
    return {name: f".{name}" in listing for name in names}


def _gpu_state() -> list[dict[str, Any]]:
    returncode, output = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if returncode != 0:
        return [{"error": output}]
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        rows.append(
            {
                "index": int(fields[0]),
                "memory_used_mib": int(fields[1]),
                "memory_total_mib": int(fields[2]),
                "utilization_percent": int(fields[3]),
            }
        )
    return rows


def collect_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    log_paths = sorted(Path(path) for path in glob.glob(args.log_glob))
    texts: list[str] = []
    logs: list[dict[str, Any]] = []
    for path in log_paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        texts.append(text)
        stat = path.stat()
        logs.append(
            {
                "path": str(path.resolve()),
                "bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
            }
        )
    combined = "\n".join(texts)
    steps = [int(value) for value in STEP_RE.findall(combined)]
    updated_steps = [int(value) for value in UPDATED_STEP_RE.findall(combined)]
    skipped_totals = [int(value) for value in SKIPPED_TOTAL_RE.findall(combined)]
    consecutive_skips = [int(value) for value in CONSECUTIVE_SKIP_RE.findall(combined)]
    errors = {name: len(pattern.findall(combined)) for name, pattern in ERROR_PATTERNS.items()}
    screens = _screen_state(args.screen_name)
    target_checkpoint = args.checkpoint_dir / f"global_step_{args.target_step}"
    checkpoint_files = {
        "directory": target_checkpoint.is_dir(),
        "model": any((target_checkpoint / "actor").glob("model*_rank_*.pt"))
        if (target_checkpoint / "actor").is_dir()
        else False,
        "optimizer": any((target_checkpoint / "actor").glob("optim*_rank_*.pt"))
        if (target_checkpoint / "actor").is_dir()
        else False,
        "scheduler": any((target_checkpoint / "actor").glob("extra_state*_rank_*.pt"))
        if (target_checkpoint / "actor").is_dir()
        else False,
        "dataloader": (target_checkpoint / "data.pt").is_file(),
        "adaptive": (target_checkpoint / "shopping_state.pt").is_file(),
    }
    anomaly_count = sum(errors.values())
    complete = (
        max(steps, default=0) >= args.target_step
        and "Final validation metrics" in combined
        and all(checkpoint_files.values())
    )
    if anomaly_count:
        status = "error"
    elif complete:
        status = "complete"
    elif any(screens.values()):
        status = "running"
    else:
        status = "waiting_for_next_stage"
    return {
        "schema_version": "shopping-training-monitor-v1",
        "checked_at": datetime.now(UTC).isoformat(),
        "status": status,
        "target_step": args.target_step,
        "max_global_step": max(steps, default=0),
        "optimizer_updated_steps": sorted(set(updated_steps)),
        "dynamic_sampling": {
            "skip_events": combined.count("SHOPPING_GRPO_DYNAMIC_SAMPLING_SKIPPED"),
            "skipped_updates_total": max(skipped_totals, default=0),
            "consecutive_skips": consecutive_skips[-1] if consecutive_skips else 0,
        },
        "validation_batches_finished": combined.count("validation generation end"),
        "final_validation_present": "Final validation metrics" in combined,
        "errors": errors,
        "screens": screens,
        "gpu": _gpu_state(),
        "target_checkpoint": checkpoint_files,
        "logs": logs,
    }


def write_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    while True:
        snapshot = collect_snapshot(args)
        write_snapshot(args.output, snapshot)
        print(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")), flush=True)
        if args.once or snapshot["status"] in {"complete", "error"}:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
