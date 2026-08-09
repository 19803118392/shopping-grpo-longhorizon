#!/usr/bin/env python3
"""Materialize deterministic nested SFT subsets for the single-seed experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from shopping_grpo.evaluation.stratification import query_difficulty_features


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/sft/train.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/single-seed-42/data"
DEFAULT_EVALUATION = (
    ROOT / "data/grpo/validation.jsonl",
    ROOT / "data/evaluation/tasks.jsonl",
)
TARGET_SIZES = (95, 190, 379)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _query(row: dict) -> str:
    for message in row.get("messages") or []:
        if message.get("role") == "user":
            value = re.sub(r"^Instruction\s*:\s*", "", str(message.get("content") or ""))
            value = " ".join(value.split())
            if value:
                return value
    raise ValueError(f"task_id={row.get('task_id')} has no public user query")


def _length_bucket(row: dict) -> str:
    tool_steps = sum(message.get("role") == "tool" for message in row.get("messages") or [])
    if tool_steps <= 10:
        return "short<=10"
    if tool_steps <= 20:
        return "medium11-20"
    return "long21+"


def _row_features(row: dict) -> dict:
    features = query_difficulty_features(_query(row))
    return {
        "length_bucket": _length_bucket(row),
        "constraint_count_bucket": features["constraint_count_bucket"],
        "has_option_selection": bool(features["has_option_selection"]),
    }


def _stable_hash(seed: int, row: dict) -> str:
    material = f"{seed}:{int(row['task_id'])}:{row.get('trajectory_id', '')}".encode()
    return hashlib.sha256(material).hexdigest()


def stratified_nested_order(rows: list[dict], *, seed: int) -> tuple[list[dict], dict[int, dict]]:
    """Return one deterministic order whose prefixes preserve joint strata."""
    if not rows:
        raise ValueError("SFT source is empty")
    task_ids = [int(row["task_id"]) for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("single-seed SFT ablation requires one row per task_id")

    features = {int(row["task_id"]): _row_features(row) for row in rows}
    groups: dict[tuple[str, str, bool], list[dict]] = defaultdict(list)
    for row in rows:
        item = features[int(row["task_id"])]
        key = (
            item["length_bucket"],
            item["constraint_count_bucket"],
            item["has_option_selection"],
        )
        groups[key].append(row)
    for group in groups.values():
        group.sort(key=lambda row: _stable_hash(seed, row))

    selected = Counter()
    offsets = Counter()
    order = []
    while len(order) < len(rows):
        available = [key for key, group in groups.items() if offsets[key] < len(group)]
        key = min(
            available,
            key=lambda value: (
                (selected[value] + 1) / len(groups[value]),
                value,
            ),
        )
        order.append(groups[key][offsets[key]])
        offsets[key] += 1
        selected[key] += 1
    return order, features


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite experiment data: {path}")
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def materialize(
    source: Path,
    output_dir: Path,
    *,
    evaluation_paths: tuple[Path, ...],
    seed: int,
) -> dict:
    source = source.resolve()
    output_dir = output_dir.resolve()
    rows = _read_jsonl(source)
    if len(rows) != TARGET_SIZES[-1]:
        raise ValueError(f"expected {TARGET_SIZES[-1]} SFT rows, got {len(rows)}")
    evaluation_ids = {
        int(row["task_id"])
        for path in evaluation_paths
        for row in _read_jsonl(path.resolve())
    }
    train_ids = {int(row["task_id"]) for row in rows}
    overlap = sorted(train_ids & evaluation_ids)
    if overlap:
        raise ValueError(f"SFT/evaluation task overlap: {overlap}")

    order, features = stratified_nested_order(rows, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "shopping-single-seed-sft-data-v1",
        "seed": int(seed),
        "source": str(source),
        "source_sha256": _sha256(source),
        "evaluation": [
            {"path": str(path.resolve()), "sha256": _sha256(path.resolve())}
            for path in evaluation_paths
        ],
        "evaluation_overlap": 0,
        "subsets": {},
    }
    previous_ids: set[int] = set()
    for size in TARGET_SIZES:
        subset = order[:size]
        path = output_dir / f"sft_n{size}.jsonl"
        _write_jsonl(path, subset)
        ids = {int(row["task_id"]) for row in subset}
        if previous_ids - ids:
            raise AssertionError("nested subset invariant failed")
        previous_ids = ids
        strata = Counter(
            "|".join(
                (
                    features[task_id]["length_bucket"],
                    features[task_id]["constraint_count_bucket"],
                    "option" if features[task_id]["has_option_selection"] else "no_option",
                )
            )
            for task_id in ids
        )
        manifest["subsets"][str(size)] = {
            "path": str(path),
            "rows": len(subset),
            "sha256": _sha256(path),
            "task_ids": sorted(ids),
            "strata": dict(sorted(strata.items())),
        }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite experiment manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evaluation", type=Path, nargs="+", default=list(DEFAULT_EVALUATION))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = materialize(
        args.source,
        args.output_dir,
        evaluation_paths=tuple(args.evaluation),
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
