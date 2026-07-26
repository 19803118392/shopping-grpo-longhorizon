#!/usr/bin/env python3
"""Create the lightweight manifest frozen before Teacher Rollout."""

from __future__ import annotations

import argparse
from pathlib import Path

from shopping_grpo.environment_manifest import build_manifest, write_manifest


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shopsimulator-repository",
        type=Path,
        default=ROOT / "environments/ShopSimulator",
    )
    parser.add_argument(
        "--product-data",
        type=Path,
        default=ROOT
        / "environments/ShopSimulator/shop_env/data/items_eval_train.json",
    )
    parser.add_argument("--task-data", type=Path, required=True)
    parser.add_argument(
        "--environment-config",
        type=Path,
        default=ROOT
        / "environments/ShopSimulator/shop_env/configs/environment_v2.json",
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = build_manifest(
        shopsimulator_repository=args.shopsimulator_repository,
        shopping_grpo_repository=ROOT,
        product_data=args.product_data,
        task_data=args.task_data,
        environment_config=args.environment_config,
        seed=args.seed,
    )
    write_manifest(args.output, manifest)
    print(args.output)


if __name__ == "__main__":
    main()
