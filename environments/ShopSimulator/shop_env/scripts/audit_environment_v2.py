#!/usr/bin/env python3
"""Low-cost code reachability audit for frozen Environment v2 tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SHOP_ENV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHOP_ENV))

from scripts.build_environment_v2_index import iter_json_array  # noqa: E402
from web_agent_site.engine.goal_v2 import deterministic_price_upper  # noqa: E402
from web_agent_site.engine.reward_v2 import evaluate_purchase  # noqa: E402
from web_agent_site.engine.search_v2 import MultiFieldBM25Searcher  # noqa: E402


def _options(product):
    values = {}
    prices = {}
    for name, entries in (product.get("customization_options") or {}).items():
        values[str(name)] = []
        for entry in entries or []:
            value = str(entry.get("value", "")).strip()
            if not value:
                continue
            values[str(name)].append(value)
            prices[value] = entry.get("price")
    return values, prices


def iter_tasks(products):
    task_id = 0
    for product in products:
        instructions = product.get("instructions") or []
        first_attributes = (
            instructions[0].get("attributes") if instructions else []
        )
        if not first_attributes:
            continue
        for instruction in instructions:
            yield task_id, product, instruction
            task_id += 1


def audit_task(task_id, product, instruction, searcher):
    asin = str(product.get("asin"))
    title = str(product.get("title") or "")
    options, option_prices = _options(product)
    required_options = list(instruction.get("instruction_options") or [])
    missing_options = [
        value
        for value in required_options
        if not any(value in values for values in options.values())
    ]
    selected_options = {}
    for required in required_options:
        for name, values in options.items():
            if required in values:
                selected_options[name.lower()] = required
                break
    pricing = product.get("pricing") or []
    selected_price = next(
        (
            option_prices[value]
            for value in required_options
            if option_prices.get(value) is not None
        ),
        pricing[0] if pricing else None,
    )
    goal = {
        "asin": asin,
        "category": product.get("category"),
        "price_upper": deterministic_price_upper(
            asin,
            instruction.get("instruction"),
            selected_price or 0,
        ),
        "goal_options": required_options,
    }
    reward = evaluate_purchase(
        product,
        goal,
        price=selected_price,
        selected_options=selected_options,
    )
    oracle_hits = searcher.search(title, k=150)
    oracle_rank = next(
        (hit.rank for hit in oracle_hits if hit.asin == asin),
        None,
    )
    checks = {
        "product_exists": bool(asin),
        "indexed": searcher.contains_asin(asin),
        "oracle_recall": oracle_rank is not None,
        "detail_openable": bool(title),
        "required_options_exist": not missing_options,
        "price_valid": isinstance(selected_price, (int, float))
        and selected_price >= 0,
        "buy_path_reward_is_gold": reward.reward_type == "gold_purchase",
    }
    return {
        "task_id": task_id,
        "target_asin": asin,
        "checks": checks,
        "passed": all(checks.values()),
        "oracle_rank": oracle_rank,
        "missing_options": missing_options,
        "gold_reward": reward.reward,
        "gold_reward_type": reward.reward_type,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--products",
        type=Path,
        default=SHOP_ENV / "data" / "items_eval_train.json",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=SHOP_ENV / "search_engine" / "environment_v2.sqlite3",
    )
    parser.add_argument("--task-id", type=int, action="append")
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    selected = set(args.task_id or [])
    searcher = MultiFieldBM25Searcher(args.index)
    rows = []
    try:
        for task_id, product, instruction in iter_tasks(
            iter_json_array(args.products)
        ):
            if selected and task_id not in selected:
                continue
            rows.append(audit_task(task_id, product, instruction, searcher))
            if args.max_tasks and len(rows) >= args.max_tasks:
                break
    finally:
        searcher.close()
    summary = {
        "audited_tasks": len(rows),
        "passed_tasks": sum(row["passed"] for row in rows),
        "failed_tasks": sum(not row["passed"] for row in rows),
        "rows": rows,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if summary["failed_tasks"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
