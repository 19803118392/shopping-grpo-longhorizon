"""Human-readable tables for repeated paired evaluation JSON."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from io import StringIO


def _percent(value: object, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def _interval(summary: Mapping) -> str:
    interval = summary["strict_success_rate_wilson_95"]
    return f"[{_percent(interval['low'])}, {_percent(interval['high'])}]"


def _loop_rate(profile: Mapping) -> float:
    if "loop_rate" in profile:
        return float(profile["loop_rate"])
    attempts = int(profile.get("attempts") or 0)
    loops = int((profile.get("reward_type_counts") or {}).get("repeat_loop") or 0)
    return loops / attempts if attempts else 0.0


def overall_table_rows(report: Mapping) -> list[dict]:
    labels = report.get("labels") or {}
    delta = report["paired_task_delta"]
    rows = []
    for side in ("baseline", "candidate"):
        summary = report[side]
        profile = report["failure_profiles"][side]
        paired = "—"
        if side == "candidate":
            paired = f"{delta['wins']}/{delta['ties']}/{delta['losses']}"
        rows.append(
            {
                "method": str(labels.get(side) or side),
                "strict_success": _percent(summary["strict_success_rate"]),
                "wilson_95_ci": _interval(summary),
                "win_tie_loss_vs_baseline": paired,
                "pass_at_k": _percent(summary["pass@k"]),
                "pass_power_k": _percent(summary["pass^k"]),
                "loop_rate": _percent(_loop_rate(profile)),
                "average_steps": f"{float(profile['average_steps']):.2f}",
            }
        )
    return rows


def render_markdown_report(report: Mapping) -> str:
    rows = overall_table_rows(report)
    attempts = int(report["baseline"]["attempts_per_task"])
    delta = report["paired_task_delta"]
    bootstrap = delta["bootstrap"]
    mcnemar = report["paired_attempt_test"]
    lines = [
        "# Repeated paired evaluation",
        "",
        f"Each task has `{attempts}` paired attempts. Strict success uses the fixed attempt denominator.",
        "",
        "| Method | Strict Success | Wilson 95% CI | Win/Tie/Loss vs baseline | pass@k | pass^k | Loop rate | Avg. steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {strict_success} | {wilson_95_ci} | "
            "{win_tie_loss_vs_baseline} | {pass_at_k} | {pass_power_k} | "
            "{loop_rate} | {average_steps} |".format(**row)
        )
    lines.extend(
        [
            "",
            (
                f"Paired delta: `{delta['candidate_minus_baseline_percentage_points']:+.1f} pp`; "
                f"task-bootstrap 95% CI "
                f"`[{100 * float(bootstrap['low']):+.1f}, "
                f"{100 * float(bootstrap['high']):+.1f}] pp`; "
                f"exact McNemar `p={float(mcnemar['p_value']):.4f}`."
            ),
        ]
    )

    stratified = report.get("stratified_statistics") or {}
    static = stratified.get("static_task_strata") or {}
    if static:
        lines.extend(
            [
                "",
                "## Exploratory task strata",
                "",
                "Query-derived strata are exploratory and use no gold product fields. No multiplicity correction is applied.",
            ]
        )
        for axis, buckets in static.items():
            lines.extend(
                [
                    "",
                    f"### {axis}",
                    "",
                    "| Bucket | Tasks | Baseline | Candidate | Delta | Bootstrap 95% CI | W/T/L |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for bucket, bucket_report in buckets.items():
                bucket_delta = bucket_report["paired_task_delta"]
                bucket_bootstrap = bucket_delta["bootstrap"]
                lines.append(
                    f"| {bucket} | {bucket_report['baseline']['expected_tasks']} | "
                    f"{_percent(bucket_report['baseline']['strict_success_rate'])} | "
                    f"{_percent(bucket_report['candidate']['strict_success_rate'])} | "
                    f"{bucket_delta['candidate_minus_baseline_percentage_points']:+.1f} pp | "
                    f"[{100 * float(bucket_bootstrap['low']):+.1f}, "
                    f"{100 * float(bucket_bootstrap['high']):+.1f}] pp | "
                    f"{bucket_delta['wins']}/{bucket_delta['ties']}/{bucket_delta['losses']} |"
                )

    behavior = stratified.get("behavior_strata") or {}
    if behavior:
        lines.extend(
            [
                "",
                "## Behaviour diagnostics",
                "",
                "Search-step and trajectory-length buckets are model-conditional diagnostics, not paired causal strata.",
                "",
                "| Model | Axis | Bucket | Attempts | Strict Success | Loop rate | Avg. steps |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        labels = report.get("labels") or {}
        for side, axes in behavior.items():
            for axis, buckets in axes.items():
                for bucket, values in buckets.items():
                    lines.append(
                        f"| {labels.get(side, side)} | {axis} | {bucket} | "
                        f"{values['attempts']} | {_percent(values['strict_success_rate'])} | "
                        f"{_percent(values['loop_rate'])} | {float(values['average_steps']):.2f} |"
                    )
    return "\n".join(lines) + "\n"


def render_overall_csv(report: Mapping) -> str:
    fields = [
        "method",
        "strict_success",
        "wilson_95_ci",
        "win_tie_loss_vs_baseline",
        "pass_at_k",
        "pass_power_k",
        "loop_rate",
        "average_steps",
    ]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(overall_table_rows(report))
    return buffer.getvalue()
