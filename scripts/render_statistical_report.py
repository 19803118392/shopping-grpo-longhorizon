#!/usr/bin/env python3
"""Render a paired comparison JSON as review-ready Markdown and CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shopping_grpo.evaluation.reporting import render_markdown_report, render_overall_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render repeated paired evaluation tables")
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.comparison.read_text(encoding="utf-8"))
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown_report(report), encoding="utf-8")
    args.csv.write_text(render_overall_csv(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "comparison": str(args.comparison),
                "markdown": str(args.markdown),
                "csv": str(args.csv),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
