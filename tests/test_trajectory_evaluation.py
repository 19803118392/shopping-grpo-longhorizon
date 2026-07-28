"""Tests for the isolated, offline trajectory evaluation package."""

import json
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from shopping_grpo.evaluation.artifacts import (
    ArtifactError,
    append_jsonl_fsync,
    index_jsonl,
    iter_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from shopping_grpo.evaluation.blind_guard import (
    guard_blind_final,
    validate_canonical_blind_asset,
)
from shopping_grpo.evaluation.comparison import compare_evaluation_runs
from shopping_grpo.evaluation.contracts import (
    JUDGE_DIMENSIONS,
    JUDGE_SCHEMA_VERSION,
    ContractValidationError,
    validate_curator_response,
    validate_judge_result,
)
from shopping_grpo.evaluation.manifest import build_run_manifest
from shopping_grpo.evaluation.metrics import compute_deterministic_metrics
from shopping_grpo.evaluation.model_client import (
    DEFAULT_PRO_MODEL,
    ModelResponseError,
    OpenAIJSONClient,
)
from shopping_grpo.evaluation.prompts import (
    TRAJECTORY_JUDGE_PROMPT_VERSION,
    actor_visible_trajectory,
    build_trajectory_judge_messages,
    judge_visible_metrics,
    sanitize_terminal_for_judge,
)
from shopping_grpo.evaluation.results import (
    assemble_task_evaluation,
    summarize_evaluations,
)
from shopping_grpo.evaluation.rubric import (
    build_task_facts,
    extract_price_candidates,
    extract_rubric_candidates,
    materialize_rubric_bundle,
    stable_hash,
)
from shopping_grpo.evaluation.task_facts import task_facts_from_environment
from shopping_grpo.evaluation.trajectory import normalize_trajectory


def _call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _raw_trajectory():
    calls = [
        _call("c0", "search_products", '{"query":"医用超声波洁牙机"}'),
        _call("g0", "open_product", '{"asin":"999999999999"}'),
        _call("c1", "search_products", '{"query":"医用超声波洁牙机"}'),
        _call("c2", "open_product", '{"asin":"860965673003"}'),
        _call("c3", "select_option", '{"value":"白色"}'),
        _call("c4", "buy_now", "{}"),
    ]
    messages = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": (
                "Instruction: 找医用级超声波洁牙机，白色，价格别超100。"
            ),
        },
    ]
    for index, call in enumerate(calls):
        messages.append(
            {
                "role": "assistant",
                "content": f"assistant-{index}",
                "tool_calls": [call],
            }
        )
    reward_detail = {
        "reward_version": "shopsimulator-reward-v3",
        "reward_type": "gold_purchase",
        "reward_valid": True,
        "purchase_success": True,
        "termination_reason": "gold_purchase",
        "terminal_utility": 1.0,
        "weighted_score": 1.0,
        "hard_gates": {
            "brand": {
                "required": "HIDDEN_REWARD_TARGET_BRAND",
                "actual": "actor-visible-brand",
            }
        },
        "dimension_scores": {
            "preference": {
                "evidence": "HIDDEN_REWARD_PREFERENCE_EVIDENCE"
            }
        },
        "evidence": {
            "target_asin_match": "HIDDEN_REWARD_TARGET_ASIN",
            "preference_scoring": {
                "expected": "HIDDEN_REWARD_EXPECTED_FEATURE"
            },
        },
        "target_asin_match": True,
    }
    return {
        "trajectory_id": "trajectory-1",
        "task_id": 16255,
        "attempt_index": 0,
        "created_at": "2026-07-28T00:00:00+00:00",
        "status": "done",
        "done": True,
        "final_reward": 1.0,
        "messages": messages,
        "initial_result": {
            "instruction": "找医用级超声波洁牙机，白色，价格别超100。",
            "environment_version": "shopsimulator-environment-v2.1",
            "user_persona": {"hidden": "must-not-leak"},
        },
        "steps": [
            {
                "step_index": 0,
                "tool_call": calls[0],
                "tool_name": "search_products",
                "parameters": {"query": "医用超声波洁牙机"},
                "env_action": "search[医用超声波洁牙机]",
                "observation": "1|860965673003|95.0|洁牙机",
                "raw_observation": "HIDDEN_RAW_SEARCH_CONTENT",
                "projection": {
                    "truncated": True,
                    "raw_tokens": 100,
                    "visible_tokens": 50,
                    "critical_footer_preserved": True,
                },
                "reward": 0.0,
                "done": False,
                "result": {},
            },
            {
                "step_index": 1,
                "tool_call": calls[2],
                "tool_name": "search_products",
                "parameters": {"query": "医用超声波洁牙机"},
                "env_action": "search[医用超声波洁牙机]",
                "observation": "1|860965673003|95.0|洁牙机",
                "reward": 0.0,
                "done": False,
                "result": {},
            },
            {
                "step_index": 2,
                "tool_call": calls[3],
                "tool_name": "open_product",
                "parameters": {"asin": "860965673003"},
                "env_action": "click[860965673003]",
                "observation": "price: 95; options: 白色",
                "reward": 0.0,
                "done": False,
                "result": {},
            },
            {
                "step_index": 3,
                "tool_call": calls[4],
                "tool_name": "select_option",
                "parameters": {"value": "白色"},
                "env_action": "click[白色]",
                "observation": "selected: 白色; price: 95",
                "reward": 0.0,
                "done": False,
                "result": {},
            },
            {
                "step_index": 4,
                "tool_call": calls[5],
                "tool_name": "buy_now",
                "parameters": {},
                "env_action": "click[buy now]",
                "observation": "done",
                "reward": 1.0,
                "done": True,
                "result": {},
            },
        ],
        "blocked_tool_calls": [
            {
                "step_index": 1,
                "tool_call": calls[1],
                "reason": "click_not_in_previous_observation",
                "consecutive_count": 1,
                "latest_observation_truncated": True,
            }
        ],
        "context_turn_tokens": [
            {"step_index": 0, "input_tokens": 1000},
            {"step_index": 1, "input_tokens": 1200},
        ],
        "context_compactions": [],
        "tool_call_truncations": [],
        "terminal_result": {
            "done": True,
            "over": True,
            "reward": 1.0,
            "reward_detail": reward_detail,
            "purchase": {
                "asin": "860965673003",
                "price": 95.0,
                "options": {"颜色分类": "白色"},
                "attributes": ["HIDDEN_PURCHASE_PRIVATE_ATTRIBUTE"],
                "instruction_text": "HIDDEN_PURCHASE_PRIVATE_INSTRUCTION",
            },
            "goal": {"asin": "hidden-gold", "price_upper": None},
        },
        "error": None,
        "release_error": None,
    }


def _task_facts():
    return build_task_facts(
        task_id=16255,
        query="找医用级超声波洁牙机，白色，价格别超100。",
        target_product={
            "asin": "860965673003",
            "category": "医疗器械›冲牙器／洗牙器／洁牙机",
            "title": "医用超声波洁牙机",
            "attribute": ["医用级", "超声波"],
            "customization_options": {
                "颜色分类": [{"value": "白色", "price": 95.0}]
            },
        },
        instruction_record={
            "instruction": "找医用级超声波洁牙机，白色，价格别超100。",
            "attributes": ["医用级", "超声波"],
            "instruction_options": ["白色"],
        },
        reward_goal={
            "category": "医疗器械›冲牙器／洗牙器／洁牙机",
            "expected_core_functions": ["医用级", "超声波"],
            "required_options_by_key": {
                "color": {
                    "value": "白色",
                    "source_axis": "颜色分类",
                    "source": "instruction.instruction_options",
                }
            },
            "price_upper": None,
        },
    )


def _rubric_bundle():
    facts = _task_facts()
    candidates = extract_rubric_candidates(facts)
    selected = []
    for candidate in candidates["candidates"]:
        if candidate["constraint_type"] in {
            "category",
            "core_function",
            "option",
            "budget_upper",
        }:
            quote = (
                candidate["query_spans"][0]["text"]
                if candidate["query_spans"]
                else ""
            )
            selected.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "description": candidate["description_hint"],
                    "hardness": candidate["hardness_hint"],
                    "query_quote": quote,
                    "selection_reason": "Query 明确表达该约束",
                }
            )
    return materialize_rubric_bundle(
        task_facts=facts,
        candidates=candidates,
        curator_response={"selected_constraints": selected},
        curator_model="deepseek-v4-flash",
        curator_prompt_version="rubric-curator-v1-draft",
        rubric_version="task-rubric-v1",
    )


def _judge(bundle, *, budget_status="violated"):
    assessments = []
    for item in bundle["rubrics"]:
        status = (
            budget_status
            if item["constraint_type"] == "budget_upper"
            else "satisfied"
        )
        assessments.append(
            {
                "rubric_id": item["rubric_id"],
                "status": status,
                "reason": "来自可见轨迹证据",
                "evidence_event_ids": ["e0006"],
            }
        )
    return {
        "schema_version": JUDGE_SCHEMA_VERSION,
        "task_id": 16255,
        "trajectory_id": "trajectory-1",
        "judge_status": "valid",
        "rubric_assessments": assessments,
        "dimension_scores": {
            name: {
                "score": 2,
                "reason": "行为合理",
                "evidence_event_ids": ["e0001", "e0006"],
            }
            for name in JUDGE_DIMENSIONS
        },
        "errors": {
            "primary": None,
            "secondary": [],
            "evidence_event_ids": [],
        },
        "overall_diagnosis": "完成了搜索、核验和购买。",
    }


class TrajectoryNormalizationTest(unittest.TestCase):
    def test_normalizer_merges_guard_attempts_and_hides_private_data(self):
        normalized = normalize_trajectory(_raw_trajectory())

        self.assertEqual(normalized["actor_query"], "找医用级超声波洁牙机，白色，价格别超100。")
        self.assertEqual(len(normalized["events"]), 6)
        self.assertEqual(
            [event["event_type"] for event in normalized["events"][:3]],
            ["tool_step", "guard_rejection", "tool_step"],
        )
        self.assertEqual(normalized["events"][1]["event_id"], "e0002")
        self.assertEqual(normalized["events"][1]["action_attempt_id"], "a0002")
        self.assertNotIn(
            "audit_only_raw_observation", normalized["events"][0]
        )
        self.assertNotIn("goal", normalized["terminal"])
        self.assertNotIn("user_persona", normalized)

    def test_audit_raw_observation_is_stripped_from_judge_view(self):
        normalized = normalize_trajectory(
            _raw_trajectory(),
            include_audit_raw_observations=True,
        )
        self.assertEqual(
            normalized["events"][0]["audit_only_raw_observation"],
            "HIDDEN_RAW_SEARCH_CONTENT",
        )
        rendered = actor_visible_trajectory(normalized)
        self.assertNotIn("audit_only_raw_observation", rendered["events"][0])


class DeterministicMetricsTest(unittest.TestCase):
    def test_metrics_keep_reward_and_efficiency_separate(self):
        metrics = compute_deterministic_metrics(
            normalize_trajectory(_raw_trajectory())
        )

        self.assertTrue(
            metrics["reward_and_outcome"]["strict_gold_success"]
        )
        self.assertEqual(
            metrics["actions_and_efficiency"]["executed_tool_steps"], 5
        )
        self.assertEqual(
            metrics["actions_and_efficiency"]["action_attempts"], 6
        )
        self.assertEqual(
            metrics["actions_and_efficiency"]["visible_search_candidate_count"],
            1,
        )
        self.assertEqual(
            metrics["repetition"]["duplicate_search_query_count"], 1
        )
        self.assertEqual(
            metrics["legality"]["guard_rejection_count"], 1
        )
        self.assertEqual(metrics["context"]["max_input_tokens"], 1200)
        self.assertIsNone(metrics["timing"]["trajectory_duration_seconds"])

    def test_visible_candidate_count_uses_canonical_8_to_12_digit_ids(self):
        raw = _raw_trajectory()
        observation = (
            "1|12345678|10|八位\n"
            "2|1234567890|20|十位\n"
            "3|123456789012|30|十二位\n"
            "ignore|1234567|七位\n"
            "ignore|1234567890123|十三位"
        )
        raw["steps"][0]["observation"] = observation
        raw["steps"][1]["observation"] = observation

        metrics = compute_deterministic_metrics(normalize_trajectory(raw))

        self.assertEqual(
            metrics["actions_and_efficiency"][
                "visible_search_candidate_count"
            ],
            3,
        )


class RubricTest(unittest.TestCase):
    def test_task_facts_mapping_uses_exact_goal_index(self):
        rows = task_facts_from_environment(
            task_ids=[1],
            goals=[
                {
                    "asin": "first",
                    "instruction_text": "第一条",
                    "attributes": ["旧属性"],
                    "goal_options": [],
                },
                {
                    "asin": "second",
                    "instruction_text": "第二条白色商品",
                    "attributes": ["第二条属性"],
                    "goal_options": ["白色"],
                    "required_options_by_key": {
                        "color": {
                            "value": "白色",
                            "source_axis": "颜色分类",
                        }
                    },
                },
            ],
            product_item_dict={
                "first": {"asin": "first", "category": "旧类目"},
                "second": {
                    "asin": "second",
                    "category": "目标类目",
                    "title": "第二个商品",
                },
            },
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["task_id"], 1)
        self.assertEqual(rows[0]["query"], "第二条白色商品")
        self.assertEqual(
            rows[0]["instruction"]["attributes"], ["第二条属性"]
        )
        self.assertEqual(
            rows[0]["target_product"]["asin"], "second"
        )

    def test_price_parser_covers_query_phrase_reward_missed(self):
        candidates = extract_price_candidates("白色款，价格别超100。")

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["operator"], "lte")
        self.assertEqual(candidates[0]["expected_value"]["value"], 100.0)
        self.assertEqual(candidates[0]["hardness_hint"], "hard")

    def test_price_parser_keeps_lower_bound_direction(self):
        candidates = extract_price_candidates("预算300元以上，品质优先。")

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["operator"], "gte")
        self.assertEqual(candidates[0]["expected_value"]["value"], 300.0)

    def test_candidate_extractor_resolves_option_without_creating_fields(self):
        candidates = extract_rubric_candidates(_task_facts())["candidates"]
        option = next(
            item for item in candidates if item["constraint_type"] == "option"
        )
        budget = next(
            item
            for item in candidates
            if item["constraint_type"] == "budget_upper"
        )

        self.assertEqual(option["field_path"], "purchase.options.color")
        self.assertEqual(option["expected_value"], "白色")
        self.assertEqual(budget["expected_value"]["value"], 100.0)

    def test_curator_cannot_reference_unknown_candidate(self):
        with self.assertRaises(ContractValidationError):
            validate_curator_response(
                {
                    "selected_constraints": [
                        {
                            "candidate_id": "invented",
                            "description": "自行创造",
                            "hardness": "hard",
                            "query_quote": "",
                            "selection_reason": "无",
                        }
                    ]
                },
                candidate_ids=["c0001"],
            )

    def test_materialized_bundle_keeps_code_owned_semantics(self):
        bundle = _rubric_bundle()
        budget = next(
            item
            for item in bundle["rubrics"]
            if item["constraint_type"] == "budget_upper"
        )

        self.assertEqual(budget["operator"], "lte")
        self.assertEqual(budget["expected_value"]["value"], 100.0)
        self.assertTrue(budget["query_spans"])


class JudgeAndResultsTest(unittest.TestCase):
    def test_judge_rejects_fabricated_event_reference_and_total_score(self):
        bundle = _rubric_bundle()
        result = _judge(bundle)
        result["dimension_scores"]["search_strategy"][
            "evidence_event_ids"
        ] = ["e9999"]
        with self.assertRaises(ContractValidationError):
            validate_judge_result(
                result,
                rubric_ids=[
                    item["rubric_id"] for item in bundle["rubrics"]
                ],
                allowed_event_ids=[f"e{index:04d}" for index in range(1, 7)],
            )

        result = _judge(bundle)
        result["errors"]["primary"] = "invented_error"
        with self.assertRaises(ContractValidationError):
            validate_judge_result(
                result,
                rubric_ids=[
                    item["rubric_id"] for item in bundle["rubrics"]
                ],
            )

        result = _judge(bundle)
        result["total_score"] = 10
        with self.assertRaises(ContractValidationError):
            validate_judge_result(
                result,
                rubric_ids=[
                    item["rubric_id"] for item in bundle["rubrics"]
                ],
            )

    def test_judge_prompt_enforces_data_level_reward_isolation(self):
        normalized = normalize_trajectory(
            _raw_trajectory(),
            include_audit_raw_observations=True,
        )
        messages = build_trajectory_judge_messages(
            normalized=normalized,
            rubric_bundle=_rubric_bundle(),
            deterministic_metrics=compute_deterministic_metrics(normalized),
        )

        rendered = messages[1]["content"]
        payload = json.loads(rendered)
        self.assertNotIn("HIDDEN_RAW_SEARCH_CONTENT", rendered)
        self.assertNotIn("hidden-gold", rendered)
        self.assertNotIn("HIDDEN_REWARD_TARGET_BRAND", rendered)
        self.assertNotIn("HIDDEN_REWARD_PREFERENCE_EVIDENCE", rendered)
        self.assertNotIn("HIDDEN_REWARD_TARGET_ASIN", rendered)
        self.assertNotIn("HIDDEN_REWARD_EXPECTED_FEATURE", rendered)
        self.assertNotIn("HIDDEN_PURCHASE_PRIVATE_ATTRIBUTE", rendered)
        self.assertNotIn("HIDDEN_PURCHASE_PRIVATE_INSTRUCTION", rendered)
        self.assertTrue(
            all(
                "reward" not in event
                for event in payload["actor_visible_trajectory"]["events"]
            )
        )
        self.assertNotIn(
            "reward_rubric_disagreement",
            payload["frozen_error_taxonomy"],
        )
        self.assertNotIn(
            "infrastructure_invalid",
            payload["frozen_error_taxonomy"],
        )
        self.assertEqual(
            set(payload["judge_visible_metrics"]),
            {
                "actions_and_efficiency",
                "repetition",
                "legality",
                "context",
            },
        )
        forbidden_keys = {
            "reward",
            "reward_detail",
            "weighted_score",
            "terminal_utility",
            "reward_type",
            "hard_gates",
            "dimension_scores",
            "evidence",
            "target_asin_match",
            "strict_gold_success",
            "purchase_success",
            "final_reward",
        }

        def collect_keys(value):
            if isinstance(value, dict):
                keys = set(value)
                for child in value.values():
                    keys.update(collect_keys(child))
                return keys
            if isinstance(value, list):
                keys = set()
                for child in value:
                    keys.update(collect_keys(child))
                return keys
            return set()

        judge_auxiliary_inputs = {
            "terminal_state": payload["terminal_state"],
            "judge_visible_metrics": payload["judge_visible_metrics"],
        }
        self.assertFalse(
            forbidden_keys & collect_keys(judge_auxiliary_inputs)
        )

    def test_terminal_and_metrics_have_explicit_judge_whitelists(self):
        normalized = normalize_trajectory(_raw_trajectory())
        terminal = sanitize_terminal_for_judge(normalized["terminal"])
        visible_metrics = judge_visible_metrics(
            compute_deterministic_metrics(normalized)
        )

        self.assertEqual(
            set(terminal),
            {"done", "over", "termination_reason", "purchase"},
        )
        self.assertEqual(
            set(terminal["purchase"]),
            {"asin", "price", "options"},
        )
        self.assertNotIn("reward_and_outcome", visible_metrics)
        self.assertNotIn("validity", visible_metrics)

    def test_assembly_records_reward_rubric_disagreement(self):
        normalized = normalize_trajectory(_raw_trajectory())
        metrics = compute_deterministic_metrics(normalized)
        bundle = _rubric_bundle()
        result = assemble_task_evaluation(
            actor={"actor_id": "sft", "model": "local-sft"},
            normalized_trajectory=normalized,
            deterministic_metrics=metrics,
            rubric_bundle=bundle,
            judge_result=_judge(bundle),
        )

        self.assertEqual(
            result["reward_and_terminal"]["metrics"]["reward_type"],
            "gold_purchase",
        )
        self.assertTrue(
            result["requirement_rubric"][
                "reward_rubric_disagreement"
            ]
        )
        self.assertNotIn("total_score", result["trajectory_quality"])

        summary = summarize_evaluations(
            expected_task_ids=[16255, 99999],
            evaluations=[result],
        )
        self.assertEqual(summary["expected_tasks"], 2)
        self.assertEqual(summary["missing_task_ids"], [99999])
        self.assertEqual(
            summary["reward_and_terminal"]["gold_purchase_rate"], 0.5
        )
        self.assertEqual(
            summary["trajectory_quality"]["judge_coverage_rate"], 0.5
        )
        self.assertNotIn("total_score", summary)

    def test_paired_comparison_keeps_all_sections_separate(self):
        normalized = normalize_trajectory(_raw_trajectory())
        metrics = compute_deterministic_metrics(normalized)
        bundle = _rubric_bundle()
        grpo = assemble_task_evaluation(
            actor={"actor_id": "grpo", "model": "local-grpo"},
            normalized_trajectory=normalized,
            deterministic_metrics=metrics,
            rubric_bundle=bundle,
            judge_result=_judge(bundle),
        )
        base = deepcopy(grpo)
        base["actor"] = {"actor_id": "base", "model": "local-base"}
        base_reward = base["reward_and_terminal"]["metrics"]
        base_reward["strict_gold_success"] = False
        base_reward["purchase_success"] = False
        base_reward["reward_type"] = "wrong_purchase"
        for score in base["trajectory_quality"][
            "dimension_scores"
        ].values():
            score["score"] = 1
        base["deterministic"]["actions_and_efficiency"][
            "executed_tool_steps"
        ] = 10

        comparison = compare_evaluation_runs(
            expected_task_ids=[16255],
            runs={"base": [base], "grpo": [grpo]},
        )
        paired = comparison["pairwise"]["base_to_grpo"]
        self.assertEqual(
            paired["reward_and_terminal"]["strict_success_transitions"][
                "failure_to_success"
            ],
            1,
        )
        self.assertEqual(
            paired["trajectory_quality"]["search_strategy"][
                "mean_delta_target_minus_source"
            ],
            1.0,
        )
        self.assertNotIn("total_score", comparison)


class RunManifestTest(unittest.TestCase):
    def test_manifest_refuses_secret_fields(self):
        with self.assertRaises(ValueError):
            build_run_manifest(
                run_id="run-1",
                actor={"model": "sft", "api_key": "must-not-write"},
                task_manifest={"path": "tasks.jsonl", "sha256": "x"},
                environment={"version": "v2.1"},
                protocol={"temperature": 0},
                code={"commit": "abc"},
                judge={"model": "deepseek-v4-pro"},
            )

    def test_manifest_contains_versions_without_credentials(self):
        manifest = build_run_manifest(
            run_id="run-1",
            actor={"model": "sft"},
            task_manifest={"path": "tasks.jsonl", "sha256": "x"},
            environment={"version": "v2.1"},
            protocol={"temperature": 0, "max_tokens": 512},
            code={"commit": "abc"},
            judge={"model": "deepseek-v4-pro"},
            created_at="2026-07-28T00:00:00+00:00",
        )

        self.assertIn("artifact_schemas", manifest)
        self.assertIn("prompt_versions", manifest)


class ArtifactAndCliTest(unittest.TestCase):
    def test_artifacts_refuse_duplicate_keys_and_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            write_jsonl_atomic(
                path,
                [{"task_id": 1}, {"task_id": 1}],
            )
            with self.assertRaises(ArtifactError):
                index_jsonl(path, key="task_id")
            with self.assertRaises(FileExistsError):
                write_jsonl_atomic(path, [{"task_id": 2}])
            append_path = Path(directory) / "append.jsonl"
            append_jsonl_fsync(append_path, {"trajectory_id": "one"})
            append_jsonl_fsync(append_path, {"trajectory_id": "two"})
            self.assertEqual(len(list(iter_jsonl(append_path))), 2)

    def test_offline_cli_preprocess_judge_input_and_assemble(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.jsonl"
            preprocessed = root / "preprocessed.jsonl"
            rubrics = root / "rubrics.jsonl"
            requests = root / "judge_requests.jsonl"
            judges = root / "judges.jsonl"
            actor = root / "actor.json"
            tasks = root / "tasks.jsonl"
            output = root / "evaluations.jsonl"
            summary = root / "summary.json"
            write_jsonl_atomic(raw, [_raw_trajectory()])
            write_jsonl_atomic(rubrics, [_rubric_bundle()])
            write_jsonl_atomic(judges, [_judge(_rubric_bundle())])
            write_jsonl_atomic(tasks, [{"task_id": 16255}])
            write_json_atomic(
                actor,
                {"actor_id": "sft", "model": "local-sft"},
            )
            script = (
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "build_trajectory_evaluation_artifacts.py"
            )

            def run(*arguments):
                return subprocess.run(
                    [sys.executable, str(script), *map(str, arguments)],
                    check=True,
                    capture_output=True,
                    text=True,
                )

            run(
                "preprocess",
                "--raw",
                raw,
                "--output",
                preprocessed,
            )
            preprocessed_row = next(iter_jsonl(preprocessed))
            self.assertEqual(
                preprocessed_row["deterministic_metrics"][
                    "actions_and_efficiency"
                ]["executed_tool_steps"],
                5,
            )

            run(
                "judge-inputs",
                "--preprocessed",
                preprocessed,
                "--rubrics",
                rubrics,
                "--output",
                requests,
            )
            request = next(iter_jsonl(requests))
            rendered = json.dumps(request, ensure_ascii=False)
            self.assertTrue(request["judge_required"])
            self.assertEqual(request["judge_model"], DEFAULT_PRO_MODEL)
            self.assertTrue(request["judge_request_hash"])
            request_material = {
                key: value
                for key, value in request.items()
                if key != "judge_request_hash"
            }
            self.assertEqual(
                request["judge_request_hash"],
                stable_hash(request_material),
            )
            changed_material = deepcopy(request_material)
            changed_material["messages"][1]["content"] += " "
            self.assertNotEqual(
                request["judge_request_hash"],
                stable_hash(changed_material),
            )
            self.assertTrue(request["rubric_ids"])
            self.assertEqual(
                request["allowed_event_ids"],
                [f"e{index:04d}" for index in range(1, 7)],
            )
            self.assertNotIn("HIDDEN_RAW_SEARCH_CONTENT", rendered)
            self.assertNotIn("hidden-gold", rendered)
            self.assertNotIn("HIDDEN_REWARD_TARGET_BRAND", rendered)

            run(
                "assemble",
                "--preprocessed",
                preprocessed,
                "--rubrics",
                rubrics,
                "--judges",
                judges,
                "--expected-tasks",
                tasks,
                "--actor",
                actor,
                "--output",
                output,
                "--summary",
                summary,
            )
            evaluation = next(iter_jsonl(output))
            summary_payload = json.loads(summary.read_text(encoding="utf-8"))
            self.assertTrue(
                evaluation["requirement_rubric"][
                    "reward_rubric_disagreement"
                ]
            )
            self.assertEqual(
                summary_payload["reward_and_terminal"][
                    "gold_purchase_rate"
                ],
                1.0,
            )

    def test_model_cli_resumes_not_judged_without_api_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests = root / "requests.jsonl"
            output = root / "judges.jsonl"
            not_judged = {
                "schema_version": JUDGE_SCHEMA_VERSION,
                "task_id": 7,
                "trajectory_id": "infra-7",
                "judge_status": "not_judged",
                "rubric_assessments": [],
                "dimension_scores": {},
                "errors": {
                    "primary": "infrastructure_invalid",
                    "secondary": [],
                    "evidence_event_ids": [],
                },
                "overall_diagnosis": "基础设施无效，未评分。",
            }
            request = {
                "schema_version": "shopping-judge-request-v2",
                "task_id": 7,
                "trajectory_id": "infra-7",
                "judge_required": False,
                "prompt_version": TRAJECTORY_JUDGE_PROMPT_VERSION,
                "judge_model": DEFAULT_PRO_MODEL,
                "not_judged_result": not_judged,
            }
            request["judge_request_hash"] = stable_hash(request)
            write_jsonl_atomic(requests, [request])
            script = (
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "run_trajectory_evaluation_models.py"
            )
            command = [
                sys.executable,
                str(script),
                "judge",
                "--requests",
                str(requests),
                "--output",
                str(output),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            subprocess.run(
                [*command, "--resume"],
                check=True,
                capture_output=True,
                text=True,
            )

            rows = list(iter_jsonl(output))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["judge_status"], "not_judged")
            self.assertEqual(
                rows[0]["judge_request_hash"],
                request["judge_request_hash"],
            )

            changed = deepcopy(request)
            changed["not_judged_result"]["overall_diagnosis"] = "内容已变化。"
            changed["judge_request_hash"] = stable_hash(
                {
                    key: value
                    for key, value in changed.items()
                    if key != "judge_request_hash"
                }
            )
            write_jsonl_atomic(requests, [changed], force=True)
            failed = subprocess.run(
                [*command, "--resume"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn(
                "cached Judge request hash mismatch",
                failed.stderr,
            )

    def test_blind_guard_uses_content_and_task_ids_not_filename(self):
        guard, task_ids = validate_canonical_blind_asset()
        self.assertEqual(guard["split_role"], "blind_final_test")
        self.assertEqual(len(task_ids), 200)

        source = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "benchmarks"
            / "shop_benchmark_reward_v3_final_200.jsonl"
        )
        with tempfile.TemporaryDirectory() as directory:
            renamed = Path(directory) / "innocent_name.jsonl"
            renamed.write_bytes(source.read_bytes())
            with self.assertRaisesRegex(
                ArtifactError,
                "frozen blind-final tasks",
            ):
                guard_blind_final([renamed], allowed=False)
            guard_blind_final([renamed], allowed=True)

            reserialized = Path(directory) / "unrelated_tasks.txt"
            rows = list(iter_jsonl(source))
            write_jsonl_atomic(reserialized, reversed(rows), force=True)
            self.assertNotEqual(
                reserialized.read_bytes(),
                source.read_bytes(),
            )
            with self.assertRaisesRegex(
                ArtifactError,
                "frozen blind-final tasks",
            ):
                guard_blind_final([reserialized], allowed=False)


class ModelClientTest(unittest.TestCase):
    def test_json_client_is_deterministic_and_does_not_return_credentials(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update(
                {
                    "url": url,
                    "payload": deepcopy(payload),
                    "headers": deepcopy(headers),
                    "timeout": timeout,
                }
            )
            return {
                "id": "request-1",
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "message": {
                            "content": '{"selected_constraints":[]}'
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }

        client = OpenAIJSONClient(
            model="deepseek-v4-flash",
            base_url="https://provider.example/v1",
            api_key="private-key",
            max_tokens=128,
            transport=transport,
        )
        response = client.complete_json(
            [{"role": "user", "content": "return json"}]
        )

        self.assertEqual(captured["payload"]["temperature"], 0.0)
        self.assertEqual(
            captured["payload"]["thinking"], {"type": "disabled"}
        )
        self.assertEqual(
            response["result"], {"selected_constraints": []}
        )
        self.assertEqual(response["metadata"]["usage"]["total_tokens"], 12)
        self.assertNotIn("private-key", json.dumps(response))

    def test_json_client_rejects_markdown_or_non_json_content(self):
        client = OpenAIJSONClient(
            model="deepseek-v4-pro",
            base_url="https://provider.example/v1",
            api_key="private-key",
            retries=0,
            transport=lambda *_: {
                "choices": [
                    {"message": {"content": "```json\\n{}\\n```"}}
                ]
            },
        )

        with self.assertRaises(ModelResponseError):
            client.complete_json(
                [{"role": "user", "content": "return json"}]
            )

    def test_json_client_retries_429_and_records_retry_audit(self):
        calls = []

        def transport(url, payload, headers, timeout):
            del payload, headers, timeout
            calls.append(url)
            if len(calls) == 1:
                raise HTTPError(
                    url,
                    429,
                    "rate limited",
                    {"Retry-After": "0"},
                    None,
                )
            return {
                "id": "request-after-retry",
                "choices": [{"message": {"content": "{}"}}],
            }

        client = OpenAIJSONClient(
            model="deepseek-v4-pro",
            base_url="https://provider.example/v1",
            api_key="private-key",
            retries=2,
            retry_delay_seconds=0,
            transport=transport,
        )
        response = client.complete_json(
            [{"role": "user", "content": "return json"}]
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(response["metadata"]["attempts"], 2)
        self.assertEqual(
            response["metadata"]["retry_http_statuses"],
            [429],
        )

    def test_json_client_fails_fast_for_nonretryable_http_status(self):
        calls = []

        def transport(url, payload, headers, timeout):
            del payload, headers, timeout
            calls.append(url)
            raise HTTPError(url, 401, "unauthorized", {}, None)

        client = OpenAIJSONClient(
            model="deepseek-v4-pro",
            base_url="https://provider.example/v1",
            api_key="private-key",
            retries=3,
            retry_delay_seconds=0,
            transport=transport,
        )
        with self.assertRaises(HTTPError):
            client.complete_json(
                [{"role": "user", "content": "return json"}]
            )
        self.assertEqual(len(calls), 1)

    def test_json_client_respects_retry_after_for_503(self):
        calls = []

        def transport(url, payload, headers, timeout):
            del payload, headers, timeout
            calls.append(url)
            if len(calls) == 1:
                raise HTTPError(
                    url,
                    503,
                    "unavailable",
                    {"Retry-After": "3"},
                    None,
                )
            return {"choices": [{"message": {"content": "{}"}}]}

        client = OpenAIJSONClient(
            model="deepseek-v4-pro",
            base_url="https://provider.example/v1",
            api_key="private-key",
            retries=1,
            retry_delay_seconds=1,
            transport=transport,
        )
        with patch(
            "shopping_grpo.evaluation.model_client.time.sleep"
        ) as sleep:
            response = client.complete_json(
                [{"role": "user", "content": "return json"}]
            )

        sleep.assert_called_once_with(3.0)
        self.assertEqual(
            response["metadata"]["retry_wait_seconds"],
            3.0,
        )


if __name__ == "__main__":
    unittest.main()
