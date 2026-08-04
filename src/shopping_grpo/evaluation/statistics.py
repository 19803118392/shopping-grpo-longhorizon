"""Repeated-run statistics for fixed Shopping Agent benchmarks.

All benchmark tasks remain in the denominator.  Missing attempts are reported and
counted as failures for headline rates, preventing an interrupted run from looking
better than a complete one.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence

from shopping_grpo.evaluation.summary import is_strict_success

REPEATED_RUN_SCHEMA_VERSION = "shopping-repeated-run-statistics-v1"
PAIRED_STATISTICS_SCHEMA_VERSION = "shopping-paired-statistics-v1"


def _expected_ids(expected_task_ids: Iterable[int]) -> list[int]:
    expected = [int(task_id) for task_id in expected_task_ids]
    if not expected:
        raise ValueError("expected_task_ids must not be empty")
    if len(expected) != len(set(expected)):
        raise ValueError("expected_task_ids contains duplicates")
    return expected


def _attempt_matrix(
    expected: Sequence[int],
    trajectories: Iterable[Mapping],
    attempts_per_task: int,
) -> tuple[dict[int, list[bool]], list[dict[str, int]], set[tuple[int, int]]]:
    attempts_per_task = int(attempts_per_task)
    if attempts_per_task < 1:
        raise ValueError("attempts_per_task must be at least 1")
    expected_set = set(expected)
    observed: dict[tuple[int, int], bool] = {}
    for trajectory in trajectories:
        task_id = int(trajectory["task_id"])
        if task_id not in expected_set:
            raise ValueError(f"unexpected task_id {task_id}")
        attempt_index = int(trajectory.get("attempt_index", 0))
        if not 0 <= attempt_index < attempts_per_task:
            raise ValueError(
                f"attempt_index {attempt_index} is outside [0, {attempts_per_task})"
            )
        key = (task_id, attempt_index)
        if key in observed:
            raise ValueError(
                f"duplicate trajectory for task_id={task_id}, attempt_index={attempt_index}"
            )
        observed[key] = is_strict_success(trajectory)

    matrix: dict[int, list[bool]] = {}
    missing: list[dict[str, int]] = []
    for task_id in expected:
        matrix[task_id] = []
        for attempt_index in range(attempts_per_task):
            key = (task_id, attempt_index)
            matrix[task_id].append(observed.get(key, False))
            if key not in observed:
                missing.append(
                    {"task_id": task_id, "attempt_index": attempt_index}
                )
    return matrix, missing, set(observed)


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> dict:
    """Wilson score interval using the 95% normal quantile by default."""
    successes = int(successes)
    trials = int(trials)
    confidence = float(confidence)
    if trials < 0 or not 0 <= successes <= trials:
        raise ValueError("successes and trials are inconsistent")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if trials == 0:
        return {"confidence": confidence, "low": 0.0, "high": 0.0}
    # The project exposes 95% intervals in its public report.  Refuse to silently
    # label an approximate quantile as another confidence level.
    if not math.isclose(confidence, 0.95, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("wilson_interval currently supports confidence=0.95")
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return {
        "confidence": confidence,
        "low": max(0.0, center - radius),
        "high": min(1.0, center + radius),
    }


def summarize_repeated_run(
    *,
    expected_task_ids: Iterable[int],
    trajectories: Iterable[Mapping],
    attempts_per_task: int,
) -> dict:
    """Summarize repeated attempts with fixed task and attempt denominators."""
    expected = _expected_ids(expected_task_ids)
    matrix, missing, _ = _attempt_matrix(
        expected, trajectories, attempts_per_task
    )
    attempts_per_task = int(attempts_per_task)
    total_attempts = len(expected) * attempts_per_task
    success_counts = {
        task_id: sum(outcomes) for task_id, outcomes in matrix.items()
    }
    strict_successes = sum(success_counts.values())
    any_successes = sum(count > 0 for count in success_counts.values())
    all_successes = sum(
        count == attempts_per_task for count in success_counts.values()
    )
    return {
        "schema_version": REPEATED_RUN_SCHEMA_VERSION,
        "expected_tasks": len(expected),
        "attempts_per_task": attempts_per_task,
        "expected_attempts": total_attempts,
        "completed_attempts": total_attempts - len(missing),
        "missing_attempts": missing,
        "attempt_coverage_rate": (total_attempts - len(missing)) / total_attempts,
        "strict_successes": strict_successes,
        "strict_success_rate": strict_successes / total_attempts,
        "strict_success_rate_wilson_95": wilson_interval(
            strict_successes, total_attempts
        ),
        "empirical_pass_at_k": any_successes / len(expected),
        "empirical_pass_power_k": all_successes / len(expected),
        "tasks_with_any_success": any_successes,
        "tasks_with_all_successes": all_successes,
        "per_task": [
            {
                "task_id": task_id,
                "strict_successes": success_counts[task_id],
                "attempts": attempts_per_task,
                "success_rate": success_counts[task_id] / attempts_per_task,
            }
            for task_id in expected
        ],
    }


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return (
        float(sorted_values[lower]) * (1.0 - weight)
        + float(sorted_values[upper]) * weight
    )


def paired_bootstrap_mean_delta(
    deltas: Sequence[float],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> dict:
    """Bootstrap task-level paired deltas with a deterministic RNG seed."""
    values = [float(delta) for delta in deltas]
    samples = int(samples)
    if not values:
        raise ValueError("paired deltas must not be empty")
    if samples < 1:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    rng = random.Random(int(seed))
    size = len(values)
    distribution = sorted(
        sum(values[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    )
    alpha = (1.0 - confidence) / 2.0
    return {
        "method": "paired_task_bootstrap_percentile",
        "samples": samples,
        "seed": int(seed),
        "confidence": confidence,
        "mean_delta": sum(values) / size,
        "low": _percentile(distribution, alpha),
        "high": _percentile(distribution, 1.0 - alpha),
        "probability_delta_above_zero": sum(
            value > 0.0 for value in distribution
        )
        / samples,
    }


def mcnemar_exact(baseline_only: int, candidate_only: int) -> dict:
    """Two-sided exact McNemar test over discordant paired attempts."""
    baseline_only = int(baseline_only)
    candidate_only = int(candidate_only)
    if baseline_only < 0 or candidate_only < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = baseline_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(baseline_only, candidate_only)
        lower_tail = sum(
            math.comb(discordant, index) for index in range(tail + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * lower_tail)
    return {
        "method": "mcnemar_exact_two_sided",
        "baseline_only_successes": baseline_only,
        "candidate_only_successes": candidate_only,
        "discordant_pairs": discordant,
        "p_value": p_value,
    }


def compare_repeated_runs(
    *,
    expected_task_ids: Iterable[int],
    baseline_trajectories: Iterable[Mapping],
    candidate_trajectories: Iterable[Mapping],
    attempts_per_task: int,
    bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> dict:
    """Compare two repeated runs on identical tasks and attempt indices."""
    expected = _expected_ids(expected_task_ids)
    baseline_rows = list(baseline_trajectories)
    candidate_rows = list(candidate_trajectories)
    baseline = summarize_repeated_run(
        expected_task_ids=expected,
        trajectories=baseline_rows,
        attempts_per_task=attempts_per_task,
    )
    candidate = summarize_repeated_run(
        expected_task_ids=expected,
        trajectories=candidate_rows,
        attempts_per_task=attempts_per_task,
    )
    baseline_matrix, _, baseline_observed = _attempt_matrix(
        expected, baseline_rows, attempts_per_task
    )
    candidate_matrix, _, candidate_observed = _attempt_matrix(
        expected, candidate_rows, attempts_per_task
    )
    task_deltas = []
    wins = ties = losses = 0
    baseline_only = candidate_only = 0
    for task_id in expected:
        left = baseline_matrix[task_id]
        right = candidate_matrix[task_id]
        delta = (sum(right) - sum(left)) / int(attempts_per_task)
        task_deltas.append(delta)
        wins += delta > 0.0
        ties += delta == 0.0
        losses += delta < 0.0
        for attempt_index, (left_success, right_success) in enumerate(
            zip(left, right)
        ):
            key = (task_id, attempt_index)
            if key not in baseline_observed or key not in candidate_observed:
                continue
            baseline_only += left_success and not right_success
            candidate_only += right_success and not left_success

    bootstrap = paired_bootstrap_mean_delta(
        task_deltas,
        samples=bootstrap_samples,
        confidence=confidence,
        seed=seed,
    )
    return {
        "schema_version": PAIRED_STATISTICS_SCHEMA_VERSION,
        "baseline": baseline,
        "candidate": candidate,
        "paired_task_delta": {
            "unit": "strict_success_rate",
            "candidate_minus_baseline": bootstrap["mean_delta"],
            "candidate_minus_baseline_percentage_points": (
                100.0 * bootstrap["mean_delta"]
            ),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "bootstrap": bootstrap,
        },
        "paired_attempt_test": {
            **mcnemar_exact(baseline_only, candidate_only),
            "paired_completed_attempts": len(
                baseline_observed & candidate_observed
            ),
            "excluded_unpaired_attempts": (
                len(expected) * int(attempts_per_task)
                - len(baseline_observed & candidate_observed)
            ),
        },
    }
