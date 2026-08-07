"""Exploratory, Query-only strata for repeated benchmark comparisons.

Static task strata are derived from the user-visible Query, never from the gold
ASIN or reward details. Search-count and trajectory-length strata are reported
separately for each model because they are model behaviours, not task labels.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from shopping_grpo.evaluation.statistics import (
    compare_repeated_runs,
    summarize_failure_profile,
)
from shopping_grpo.evaluation.summary import is_strict_success

STRATIFIED_STATISTICS_SCHEMA_VERSION = "shopping-stratified-statistics-v1"

_PRICE_RE = re.compile(
    r"(?:预算|价格|价位|不超过|低于|以内|最多|至多|under|below|up\s*to)\D{0,10}\d",
    re.IGNORECASE,
)
_BRAND_MODEL_RE = re.compile(
    r"(?:品牌|牌子|型号|款号|brand|model)\s*(?:是|为|:|：)?\s*"
    r"[A-Za-z0-9\u3400-\u9fff][A-Za-z0-9\u3400-\u9fff._+\-/]{1,31}",
    re.IGNORECASE,
)
_OPTION_PATTERNS = {
    "color": re.compile(
        r"(?:颜色|花色|色号|配色|color|黑色|白色|红色|蓝色|绿色|黄色|"
        r"灰色|粉色|紫色|棕色|米色|金色|银色)",
        re.IGNORECASE,
    ),
    "size": re.compile(
        r"(?:尺寸|大小|尺码|长度|宽度|高度|直径|size|"
        r"\d+(?:\.\d+)?\s*(?:毫米|厘米|mm|cm|米))",
        re.IGNORECASE,
    ),
    "capacity": re.compile(r"(?:容量|内存|存储|毫升|升|GB|TB|capacity|memory)", re.IGNORECASE),
    "material": re.compile(r"(?:材质|面料|皮质|布料|material|fabric)", re.IGNORECASE),
    "version": re.compile(r"(?:版本|款式|套装|接口|规格|version|style)", re.IGNORECASE),
}
_CONSTRAINT_MARKER_RE = re.compile(
    r"(?:想要|希望|最好|必须|要求|需要|要有|带有|具有|支持|适合|用于|"
    r"不能|不要|不含|至少|最多|至多|不超过|低于|以内)",
    re.IGNORECASE,
)


def _clean_query(value: object) -> str:
    query = re.sub(r"\s+", " ", str(value or "")).strip()
    query = re.sub(r"^Instruction\s*:\s*", "", query, flags=re.IGNORECASE)
    return query


def query_from_trajectory(row: Mapping) -> str:
    initial = row.get("initial_result") or {}
    if isinstance(initial, Mapping):
        query = _clean_query(initial.get("instruction"))
        if query:
            return query
    raise ValueError(
        f"trajectory task_id={row.get('task_id')} is missing public initial_result.instruction"
    )


def query_difficulty_features(query: str) -> dict:
    """Return conservative, reproducible features from the public Query only."""
    query = _clean_query(query)
    if not query:
        raise ValueError("difficulty stratification requires a non-empty Query")
    option_axes = sorted(name for name, pattern in _OPTION_PATTERNS.items() if pattern.search(query))
    price_constraint = bool(_PRICE_RE.search(query))
    brand_or_model = bool(_BRAND_MODEL_RE.search(query))

    clauses = [
        part.strip()
        for part in re.split(r"[，,。；;！？!?\n]+", query)
        if part.strip()
    ]
    marked_clauses = sum(bool(_CONSTRAINT_MARKER_RE.search(clause)) for clause in clauses)
    # Category/product intent contributes one base constraint. Explicit option
    # axes, price, and brand/model are counted at least once even when expressed
    # without a marker such as "红色" or "预算 200".
    constraint_count = max(
        1 + len(option_axes) + int(price_constraint) + int(brand_or_model),
        1 + marked_clauses,
    )
    return {
        "query_only": True,
        "constraint_count": constraint_count,
        "constraint_count_bucket": (
            "1" if constraint_count == 1 else "2-3" if constraint_count <= 3 else "4+"
        ),
        "has_option_selection": bool(option_axes),
        "option_axes": option_axes,
        "has_price_constraint": price_constraint,
        "brand_or_model": brand_or_model,
    }


def _task_queries(
    expected_task_ids: Sequence[int],
    baseline_rows: Sequence[Mapping],
    candidate_rows: Sequence[Mapping],
) -> dict[int, str]:
    expected = {int(task_id) for task_id in expected_task_ids}
    queries: dict[int, str] = {}
    for row in [*baseline_rows, *candidate_rows]:
        task_id = int(row["task_id"])
        if task_id not in expected:
            continue
        query = query_from_trajectory(row)
        previous = queries.setdefault(task_id, query)
        if previous != query:
            raise ValueError(f"public Query differs across trajectories for task_id={task_id}")
    missing = sorted(expected - set(queries))
    if missing:
        raise ValueError(f"no public Query is available for task_ids={missing}")
    return queries


def _compact_pair(report: Mapping) -> dict:
    return {
        "baseline": report["baseline"],
        "candidate": report["candidate"],
        "paired_task_delta": report["paired_task_delta"],
        "paired_attempt_test": report["paired_attempt_test"],
    }


def _static_axis_report(
    *,
    groups: Mapping[str, Sequence[int]],
    baseline_rows: Sequence[Mapping],
    candidate_rows: Sequence[Mapping],
    attempts_per_task: int,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    reports = {}
    for bucket, task_ids in sorted(groups.items()):
        task_set = set(task_ids)
        baseline = [row for row in baseline_rows if int(row["task_id"]) in task_set]
        candidate = [row for row in candidate_rows if int(row["task_id"]) in task_set]
        reports[bucket] = _compact_pair(
            compare_repeated_runs(
                expected_task_ids=task_ids,
                baseline_trajectories=baseline,
                candidate_trajectories=candidate,
                attempts_per_task=attempts_per_task,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
        )
    return reports


def _step_bucket(value: int, *, kind: str) -> str:
    if kind == "search":
        return "0" if value == 0 else "1" if value == 1 else "2-3" if value <= 3 else "4+"
    if kind == "trajectory":
        return "short<=10" if value <= 10 else "medium11-20" if value <= 20 else "long21+"
    raise ValueError(f"unknown step bucket kind: {kind}")


def _behavior_report(rows: Sequence[Mapping], *, kind: str) -> dict:
    grouped: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        steps = row.get("steps") or []
        value = (
            sum(str(step.get("tool_name") or "") == "search_products" for step in steps)
            if kind == "search"
            else len(steps)
        )
        grouped[_step_bucket(value, kind=kind)].append(row)
    reports = {}
    for bucket, bucket_rows in sorted(grouped.items()):
        profile = summarize_failure_profile(bucket_rows)
        successes = sum(bool(is_strict_success(row)) for row in bucket_rows)
        reports[bucket] = {
            "attempts": len(bucket_rows),
            "strict_successes": successes,
            "strict_success_rate": successes / len(bucket_rows),
            "average_steps": profile["average_steps"],
            "search_steps_per_attempt": profile["search_steps_per_attempt"],
            "loop_rate": profile["loop_rate"],
        }
    return reports


def build_stratified_comparison(
    *,
    benchmark_tasks: Sequence[Mapping],
    baseline_trajectories: Iterable[Mapping],
    candidate_trajectories: Iterable[Mapping],
    attempts_per_task: int,
    bootstrap_samples: int = 10_000,
    seed: int = 2026,
) -> dict:
    """Build static paired strata plus model-conditional behaviour profiles."""
    tasks = list(benchmark_tasks)
    expected_task_ids = [int(task["task_id"]) for task in tasks]
    baseline_rows = list(baseline_trajectories)
    candidate_rows = list(candidate_trajectories)
    queries = _task_queries(expected_task_ids, baseline_rows, candidate_rows)
    features = {
        task_id: query_difficulty_features(query) for task_id, query in queries.items()
    }

    axes: dict[str, dict[str, list[int]]] = {
        "constraint_count": defaultdict(list),
        "option_selection": defaultdict(list),
        "price_constraint": defaultdict(list),
    }
    for task_id in expected_task_ids:
        task_features = features[task_id]
        axes["constraint_count"][task_features["constraint_count_bucket"]].append(task_id)
        axes["option_selection"]["yes" if task_features["has_option_selection"] else "no"].append(
            task_id
        )
        axes["price_constraint"]["yes" if task_features["has_price_constraint"] else "no"].append(
            task_id
        )

    reference_groups: dict[str, list[int]] = defaultdict(list)
    for task in tasks:
        bucket = str(task.get("length_bucket") or "").strip()
        if bucket:
            reference_groups[bucket].append(int(task["task_id"]))
    if reference_groups:
        axes["reference_length"] = reference_groups

    static_reports = {
        axis: _static_axis_report(
            groups=groups,
            baseline_rows=baseline_rows,
            candidate_rows=candidate_rows,
            attempts_per_task=attempts_per_task,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        for axis, groups in axes.items()
    }
    return {
        "schema_version": STRATIFIED_STATISTICS_SCHEMA_VERSION,
        "definitions": {
            "static_task_strata": "public Query only; reference_length uses frozen benchmark metadata",
            "constraint_count": "one product intent plus explicit Query constraints; buckets 1, 2-3, 4+",
            "behavior_strata": "model-conditional diagnostics; not used for causal paired claims",
            "search_steps": "number of search_products calls: 0, 1, 2-3, 4+",
            "trajectory_length": "executed steps: short<=10, medium11-20, long21+",
            "multiple_comparisons": "exploratory; no multiplicity correction",
        },
        "task_features": [
            {"task_id": task_id, **features[task_id]} for task_id in expected_task_ids
        ],
        "static_task_strata": static_reports,
        "behavior_strata": {
            "baseline": {
                "search_steps": _behavior_report(baseline_rows, kind="search"),
                "trajectory_length": _behavior_report(baseline_rows, kind="trajectory"),
            },
            "candidate": {
                "search_steps": _behavior_report(candidate_rows, kind="search"),
                "trajectory_length": _behavior_report(candidate_rows, kind="trajectory"),
            },
        },
    }
