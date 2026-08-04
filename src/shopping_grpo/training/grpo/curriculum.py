"""Deterministic curriculum stages derived from public rollout-length metadata."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

CURRICULUM_SCHEMA_VERSION = "shopping-length-curriculum-v1"
BUCKET_ORDER = ("short", "medium", "long")
DEFAULT_STAGES = (
    ("stage_1_short", ("short",)),
    ("stage_2_short_medium", ("short", "medium")),
    ("stage_3_all", ("short", "medium", "long")),
)


def validate_length_metadata(rows: Iterable[Mapping]) -> list[dict]:
    """Validate task-local difficulty labels without reading hidden environment goals."""
    validated = []
    seen = set()
    for index, row in enumerate(rows):
        try:
            task_id = int(row["task_id"])
            probe_steps = int(row["probe_steps"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"metadata row {index} has invalid task_id/probe_steps") from exc
        bucket = str(row.get("length_bucket") or "")
        if task_id in seen:
            raise ValueError(f"duplicate metadata task_id {task_id}")
        if probe_steps < 1:
            raise ValueError(f"task {task_id} has non-positive probe_steps")
        if bucket not in BUCKET_ORDER:
            raise ValueError(f"task {task_id} has unknown length_bucket {bucket!r}")
        expected_bucket = (
            "short" if probe_steps <= 10 else "medium" if probe_steps <= 20 else "long"
        )
        if bucket != expected_bucket:
            raise ValueError(
                f"task {task_id} bucket {bucket!r} conflicts with probe_steps={probe_steps}"
            )
        seen.add(task_id)
        validated.append(
            {
                "task_id": task_id,
                "probe_steps": probe_steps,
                "length_bucket": bucket,
            }
        )
    if not validated:
        raise ValueError("length metadata must not be empty")
    return validated


def build_curriculum_plan(
    rows: Iterable[Mapping],
    *,
    stages: Sequence[tuple[str, Sequence[str]]] = DEFAULT_STAGES,
) -> dict:
    """Return cumulative task-id stages while preserving source metadata order."""
    metadata = validate_length_metadata(rows)
    stage_reports = []
    previous: set[int] = set()
    names: set[str] = set()
    for name, raw_buckets in stages:
        name = str(name)
        buckets = tuple(str(bucket) for bucket in raw_buckets)
        if not name or name in names:
            raise ValueError(f"duplicate or empty curriculum stage name {name!r}")
        if not buckets or any(bucket not in BUCKET_ORDER for bucket in buckets):
            raise ValueError(f"stage {name!r} contains an invalid bucket")
        selected = [
            row for row in metadata if row["length_bucket"] in set(buckets)
        ]
        selected_ids = {row["task_id"] for row in selected}
        if not previous.issubset(selected_ids):
            raise ValueError("curriculum stages must be cumulative")
        counts = Counter(row["length_bucket"] for row in selected)
        stage_reports.append(
            {
                "name": name,
                "buckets": list(buckets),
                "tasks": len(selected),
                "bucket_counts": {
                    bucket: counts.get(bucket, 0) for bucket in BUCKET_ORDER
                },
                "task_ids": [row["task_id"] for row in selected],
            }
        )
        previous = selected_ids
        names.add(name)
    return {
        "schema_version": CURRICULUM_SCHEMA_VERSION,
        "source_tasks": len(metadata),
        "source_bucket_counts": dict(
            sorted(Counter(row["length_bucket"] for row in metadata).items())
        ),
        "difficulty_signal": "teacher_probe_executed_steps",
        "hidden_goal_fields_used": False,
        "stages": stage_reports,
    }


def select_parquet_rows(table, task_ids: Sequence[int]):
    """Select PyArrow rows in source order and fail if task coverage differs."""
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
    except ImportError as exc:  # pragma: no cover - exercised in data/GPU environments
        raise RuntimeError("curriculum materialization requires pyarrow") from exc

    wanted = {int(task_id) for task_id in task_ids}
    extra_info = table.column("extra_info")
    source_ids = pc.struct_field(extra_info, "task_id")
    selected = table.filter(pc.is_in(source_ids, value_set=pa.array(sorted(wanted))))
    selected_ids = {
        int(value)
        for value in pc.struct_field(selected.column("extra_info"), "task_id").to_pylist()
    }
    if selected_ids != wanted or selected.num_rows != len(wanted):
        missing = sorted(wanted - selected_ids)
        raise ValueError(
            "source parquet does not contain exactly one row per curriculum task; "
            f"missing={missing[:10]} selected_rows={selected.num_rows}"
        )
    return selected
