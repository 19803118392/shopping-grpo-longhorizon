"""Content- and task-ID-based protection for the frozen blind final test."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from shopping_grpo.evaluation.artifacts import ArtifactError

BLIND_GUARD_SCHEMA = "shopping-blind-asset-guard-v1"
_ROOT = Path(__file__).resolve().parents[3]
_GUARD_MANIFEST = (
    _ROOT
    / "data/benchmarks/shop_benchmark_reward_v3_final_200.guard.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read blind asset manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"blind asset manifest must be an object: {path}")
    return value


def _row_task_id(row: Mapping) -> int | None:
    value = row.get("task_id")
    if value is None:
        extra = row.get("extra_info")
        if isinstance(extra, Mapping):
            value = extra.get("task_id")
    if value is None:
        normalized = row.get("normalized_trajectory")
        if isinstance(normalized, Mapping):
            value = normalized.get("task_id")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"invalid task_id in {row!r}") from exc


def _jsonl_task_ids(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    opener = gzip.open if path.name.endswith(".gz") else open
    task_ids = set()
    try:
        with opener(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ArtifactError(
                        f"{path}:{line_number}: JSONL row must be an object"
                    )
                task_id = _row_task_id(value)
                if task_id is not None:
                    task_ids.add(task_id)
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{path}: invalid JSONL during blind guard") from exc
    return task_ids


def validate_canonical_blind_asset() -> tuple[dict, set[int]]:
    """Validate the versioned guard, frozen metadata, content hash, and IDs."""

    guard = _load_object(_GUARD_MANIFEST)
    if guard.get("schema_version") != BLIND_GUARD_SCHEMA:
        raise ArtifactError("unsupported blind guard schema")
    if guard.get("manifest_version") != 1:
        raise ArtifactError("unsupported blind guard manifest version")
    if guard.get("split_role") != "blind_final_test":
        raise ArtifactError("blind guard split_role must be blind_final_test")

    task_path = _ROOT / str(guard.get("task_file") or "")
    metadata_path = _ROOT / str(guard.get("metadata_file") or "")
    for path in (task_path, metadata_path):
        if not path.is_file():
            raise ArtifactError(f"canonical blind asset is missing: {path}")
    if _sha256_file(task_path) != guard.get("task_sha256"):
        raise ArtifactError("canonical blind task SHA256 mismatch")
    if _sha256_file(metadata_path) != guard.get("metadata_sha256"):
        raise ArtifactError("canonical blind metadata SHA256 mismatch")

    metadata = _load_object(metadata_path)
    required = guard.get("required_metadata")
    if not isinstance(required, Mapping):
        raise ArtifactError("blind guard required_metadata must be an object")
    mismatches = {
        key: {"required": expected, "actual": metadata.get(key)}
        for key, expected in required.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ArtifactError(
            "canonical blind metadata contract mismatch: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    if metadata.get("output_sha256") != guard.get("task_sha256"):
        raise ArtifactError("blind metadata output_sha256 mismatch")

    task_ids = _jsonl_task_ids(task_path)
    if len(task_ids) != int(guard.get("task_count", -1)):
        raise ArtifactError("canonical blind task count mismatch")
    return guard, task_ids


def guard_blind_final(
    paths: Iterable[Path],
    *,
    allowed: bool,
) -> None:
    """Reject any artifact containing final-test task IDs, independent of name."""

    guard, final_task_ids = validate_canonical_blind_asset()
    if allowed:
        return
    blocked = {}
    canonical_sha = str(guard["task_sha256"])
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        same_content = _sha256_file(path) == canonical_sha
        overlap = sorted(_jsonl_task_ids(path) & final_task_ids)
        if same_content or overlap:
            blocked[str(path)] = {
                "same_content": same_content,
                "overlap_count": len(overlap),
                "sample_task_ids": overlap[:10],
            }
    if blocked:
        raise ArtifactError(
            "refusing to consume frozen blind-final tasks without "
            "--allow-blind-final: "
            + json.dumps(blocked, ensure_ascii=False, sort_keys=True)
        )
