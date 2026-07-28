"""Pure-CPU smoke checks for the repository's stable public contracts."""

from __future__ import annotations

import json

from shopping_grpo.evaluation.contracts import (
    JUDGE_DIMENSIONS,
    JUDGE_SCHEMA_VERSION,
    RUBRIC_SCHEMA_VERSION,
)
from shopping_grpo.evaluation.metrics import compute_deterministic_metrics
from shopping_grpo.evaluation.prompts import build_trajectory_judge_messages
from shopping_grpo.evaluation.results import assemble_task_evaluation
from shopping_grpo.evaluation.trajectory import normalize_trajectory
from shopping_grpo.sft_training import IGNORE_INDEX, build_supervised_example
from shopping_grpo.shop_tools import SHOP_TOOL_SCHEMAS_V2
from shopping_grpo.verl_dynamic_sampling import select_reward_varying_groups


class _CharacterTemplate:
    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=False,
        add_generation_prompt=False,
    ):
        del tools, tokenize
        text = ""
        for message in messages:
            text += f"<{message['role']}>"
            text += message.get("content") or ""
            for call in message.get("tool_calls") or []:
                text += f"[tool={call['function']['name']}]"
            text += f"</{message['role']}>"
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(character) for character in text]}


def _raw_sample() -> dict:
    search_call = {
        "id": "call-search",
        "type": "function",
        "function": {
            "name": "search_products",
            "arguments": '{"query":"白色保温杯"}',
        },
    }
    buy_call = {
        "id": "call-buy",
        "type": "function",
        "function": {"name": "buy_now", "arguments": "{}"},
    }
    return {
        "trajectory_id": "cpu-smoke-trajectory",
        "task_id": 900001,
        "attempt_index": 0,
        "status": "done",
        "done": True,
        "final_reward": 1.0,
        "messages": [
            {"role": "user", "content": "Instruction: 找白色保温杯。"},
            {"role": "assistant", "content": "", "tool_calls": [search_call]},
            {"role": "assistant", "content": "", "tool_calls": [buy_call]},
        ],
        "initial_result": {
            "instruction": "找白色保温杯。",
            "environment_version": "shopsimulator-environment-v2.1",
            "reward_version": "shopsimulator-reward-v3",
        },
        "steps": [
            {
                "tool_call": search_call,
                "tool_name": "search_products",
                "parameters": {"query": "白色保温杯"},
                "env_action": "search[白色保温杯]",
                "observation": "1|12345678|39.0|白色保温杯",
                "raw_observation": "audit-only",
                "projection": {
                    "truncated": False,
                    "raw_tokens": 10,
                    "visible_tokens": 10,
                    "critical_footer_preserved": True,
                },
                "reward": 0.0,
                "done": False,
            },
            {
                "tool_call": buy_call,
                "tool_name": "buy_now",
                "parameters": {},
                "env_action": "click[buy now]",
                "observation": "Environment terminated.",
                "reward": 1.0,
                "done": True,
            },
        ],
        "terminal_result": {
            "done": True,
            "over": True,
            "reward": 1.0,
            "reward_detail": {
                "reward_version": "shopsimulator-reward-v3",
                "reward_type": "gold_purchase",
                "reward_valid": True,
                "purchase_success": True,
                "termination_reason": "gold_purchase",
                "terminal_utility": 1.0,
                "weighted_score": 1.0,
                "hard_gates": {
                    "brand": {"required": "PRIVATE", "actual": "visible"}
                },
                "evidence": {"private": "MUST_NOT_REACH_JUDGE"},
            },
            "purchase": {
                "asin": "12345678",
                "price": 39.0,
                "options": {"颜色": "白色"},
                "attributes": ["PRIVATE_PURCHASE_ATTRIBUTE"],
            },
            "goal": {"asin": "PRIVATE_GOLD"},
        },
    }


def _rubric_bundle() -> dict:
    return {
        "schema_version": RUBRIC_SCHEMA_VERSION,
        "task_id": 900001,
        "query": "找白色保温杯。",
        "rubric_version": "cpu-smoke-rubric-v1",
        "generation": {
            "extractor_version": "cpu-smoke",
            "curator_model": "cpu-smoke",
            "curator_prompt_version": "cpu-smoke",
            "task_data_hash": "cpu-smoke",
            "query_hash": "cpu-smoke",
        },
        "rubrics": [],
    }


def _judge_result() -> dict:
    return {
        "schema_version": JUDGE_SCHEMA_VERSION,
        "task_id": 900001,
        "trajectory_id": "cpu-smoke-trajectory",
        "judge_status": "valid",
        "rubric_assessments": [],
        "dimension_scores": {
            dimension: {
                "score": 2,
                "reason": "CPU smoke",
                "evidence_event_ids": ["e0001"],
            }
            for dimension in JUDGE_DIMENSIONS
        },
        "errors": {
            "primary": None,
            "secondary": [],
            "evidence_event_ids": [],
        },
        "overall_diagnosis": "CPU smoke",
    }


def run_cpu_smoke() -> dict:
    checks = []
    tool_names = {
        schema["function"]["name"] for schema in SHOP_TOOL_SCHEMAS_V2
    }
    required_tools = {
        "search_products",
        "open_product",
        "select_option",
        "buy_now",
        "finish_without_purchase",
    }
    if not required_tools <= tool_names:
        raise AssertionError("shopping tool schema is incomplete")
    checks.append("action_schema")

    normalized = normalize_trajectory(_raw_sample())
    if len(normalized["events"]) != 2:
        raise AssertionError("trajectory normalization lost events")
    checks.append("trajectory_normalization")

    metrics = compute_deterministic_metrics(normalized)
    if not metrics["reward_and_outcome"]["strict_gold_success"]:
        raise AssertionError("Reward v3 strict-success sample failed")
    checks.append("reward_v3_sample")

    tokenizer = _CharacterTemplate()
    supervised = build_supervised_example(
        messages=[
            {"role": "user", "content": "buy a cup"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search_products",
                            "arguments": '{"query":"cup"}',
                        }
                    }
                ],
            },
            {"role": "tool", "content": "private observation"},
        ],
        tools=[],
        tokenizer=tokenizer,
        max_length=1000,
    )
    if supervised is None:
        raise AssertionError("SFT label-mask sample was rejected")
    labeled = [
        token for token in supervised["labels"] if token != IGNORE_INDEX
    ]
    if not labeled or len(labeled) == len(supervised["labels"]):
        raise AssertionError("SFT labels are not assistant-only")
    checks.append("sft_label_mask")

    messages = build_trajectory_judge_messages(
        normalized=normalized,
        rubric_bundle=_rubric_bundle(),
        deterministic_metrics=metrics,
    )
    rendered = messages[1]["content"]
    for secret in (
        "MUST_NOT_REACH_JUDGE",
        "PRIVATE_PURCHASE_ATTRIBUTE",
        "PRIVATE_GOLD",
    ):
        if secret in rendered:
            raise AssertionError("Judge prompt leaked hidden information")
    checks.append("judge_prompt_isolation")

    selected, diagnostics = select_reward_varying_groups(
        ["a", "a", "b", "b"],
        [0.0, 1.0, 0.5, 0.5],
    )
    if selected != [0, 1] or diagnostics["kept_group_count"] != 1:
        raise AssertionError("dynamic sampling grouping changed")
    checks.append("dynamic_sampling_grouping")

    assembled = assemble_task_evaluation(
        actor={"actor_id": "cpu-smoke", "model": "none"},
        normalized_trajectory=normalized,
        deterministic_metrics=metrics,
        rubric_bundle=_rubric_bundle(),
        judge_result=_judge_result(),
    )
    if "total_score" in json.dumps(assembled, ensure_ascii=False):
        raise AssertionError("evaluation assembly produced a composite score")
    checks.append("evaluation_assembly")
    return {
        "schema_version": "shopping-cpu-smoke-v1",
        "checks": checks,
    }
