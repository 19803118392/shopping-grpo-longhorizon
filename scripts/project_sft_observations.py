#!/usr/bin/env python3
"""Render existing Action-only SFT messages with the online observation projector."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from transformers import AutoProcessor

from shopping_grpo.observation_projection import (
    PROJECTION_CONTRACT_VERSION,
    project_observation,
)
from shopping_grpo.sft_data import project_sft_messages


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=1536)
    parser.add_argument("--detail-token-budget", type=int, default=4096)
    parser.add_argument("--generic-token-budget", type=int, default=768)
    parser.add_argument("--search-top-k", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("--output must not overwrite --input")
    if args.metadata.resolve() in {args.input.resolve(), args.output.resolve()}:
        raise SystemExit("--metadata must be separate from --input and --output")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)

    def projector(tool_name, observation, parameters):
        visible, meta = project_observation(
            tool_name,
            observation,
            parameters=parameters,
            count_tokens=lambda text: len(
                tokenizer.encode(text, add_special_tokens=False)
            ),
            token_budget=args.token_budget,
            detail_token_budget=args.detail_token_budget,
            generic_token_budget=args.generic_token_budget,
            search_top_k=args.search_top_k,
        )
        return visible, meta.to_dict()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_rows = changed_rows = total_tools = truncated_tools = 0
    raw_tokens = visible_tokens = 0
    page_types = Counter()
    truncated_page_types = Counter()
    max_raw_tokens = max_visible_tokens = footer_failures = 0
    with args.input.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            messages, projections = project_sft_messages(row["messages"], projector)
            row["messages"] = messages
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_rows += 1
            changed = [item for item in projections if item["truncated"]]
            changed_rows += int(bool(changed))
            total_tools += len(projections)
            truncated_tools += len(changed)
            raw_tokens += sum(item["raw_tokens"] for item in projections)
            visible_tokens += sum(item["visible_tokens"] for item in projections)
            page_types.update(item["page_type"] for item in projections)
            truncated_page_types.update(item["page_type"] for item in changed)
            max_raw_tokens = max(
                [max_raw_tokens, *(item["raw_tokens"] for item in projections)]
            )
            max_visible_tokens = max(
                [max_visible_tokens, *(item["visible_tokens"] for item in projections)]
            )
            footer_failures += sum(
                not item["critical_footer_preserved"] for item in projections
            )

    metadata = {
        "input": str(args.input),
        "output": str(args.output),
        "model": args.model,
        "projection_contract": PROJECTION_CONTRACT_VERSION,
        "token_budget": args.token_budget,
        "detail_token_budget": args.detail_token_budget,
        "generic_token_budget": args.generic_token_budget,
        "search_top_k": args.search_top_k,
        "rows": total_rows,
        "changed_rows": changed_rows,
        "tool_observations": total_tools,
        "truncated_tool_observations": truncated_tools,
        "raw_tokens": raw_tokens,
        "visible_tokens": visible_tokens,
        "truncation_ratio": visible_tokens / raw_tokens if raw_tokens else 1.0,
        "max_raw_tokens": max_raw_tokens,
        "max_visible_tokens": max_visible_tokens,
        "critical_footer_failures": footer_failures,
        "page_types": dict(sorted(page_types.items())),
        "truncated_page_types": dict(sorted(truncated_page_types.items())),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
