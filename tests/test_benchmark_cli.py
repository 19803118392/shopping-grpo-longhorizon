"""验证 benchmark 清单与评测入口的最小 CLI 行为。"""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_shop_benchmark import (
    _sha256_file,
    _sha256_tree,
    parse_args,
    validate_frozen_candidate,
    validate_resumable_output,
)


class BenchmarkCliTest(unittest.TestCase):
    def test_evaluation_defaults_match_frozen_protocol(self):
        """Base、SFT、GRPO 必须默认使用同一 35 步上限。"""
        with patch.object(
            sys,
            "argv",
            [
                "evaluate_shop_benchmark.py",
                "--benchmark",
                "data/evaluation/tasks.jsonl",
                "--output",
                "outputs/eval/base/raw.jsonl",
                "--summary",
                "outputs/eval/base/summary.json",
                "--model",
                "Qwen/Qwen3.5-2B",
                "--llm-base-url",
                "http://127.0.0.1:8000/v1",
                "--api-key",
                "EMPTY",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.max_steps, 35)
        self.assertEqual(args.attempts_per_task, 1)
        self.assertIsNone(args.limit)
        self.assertEqual(args.seed, 2026)
        self.assertEqual(args.max_tokens, 512)
        self.assertEqual(args.temperature, 0.0)
        self.assertEqual(args.context_window, 24576)
        self.assertEqual(args.context_safety_margin, 512)
        self.assertFalse(args.context_compaction)
        self.assertEqual(args.observation_token_budget, 1536)
        self.assertEqual(args.observation_detail_token_budget, 2048)
        self.assertEqual(args.observation_generic_token_budget, 768)
        self.assertEqual(args.observation_search_top_k, 20)
        self.assertFalse(args.posthoc_final_200_repeated)

    def test_posthoc_final200x3_requires_explicit_distinct_mode(self):
        with patch.object(
            sys,
            "argv",
            [
                "evaluate_shop_benchmark.py",
                "--benchmark",
                "data/evaluation/tasks.jsonl",
                "--output",
                "outputs/posthoc/raw.jsonl",
                "--summary",
                "outputs/posthoc/summary.json",
                "--model",
                "shopping-agent",
                "--llm-base-url",
                "http://127.0.0.1:8000/v1",
                "--api-key",
                "EMPTY",
                "--posthoc-final-200-repeated",
                "--protocol",
                "posthoc-final200x3",
                "--attempts-per-task",
                "3",
                "--temperature",
                "0.7",
                "--top-p",
                "0.9",
                "--seed",
                "42",
            ],
        ):
            args = parse_args()

        self.assertTrue(args.posthoc_final_200_repeated)
        self.assertFalse(args.final_200)
        self.assertEqual(args.protocol, "posthoc-final200x3")
        self.assertEqual(args.attempts_per_task, 3)

    def test_final_candidate_is_rehashed_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "grpo.yaml"
            config.write_text("config", encoding="utf-8")
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "state.pt").write_bytes(b"checkpoint")
            model = root / "model"
            model.mkdir()
            (model / "model.safetensors").write_bytes(b"model")
            benchmark = root / "tasks.jsonl"
            benchmark.write_text('{"task_id":1}\n', encoding="utf-8")
            comparison = root / "comparison.json"
            comparison.write_text(
                json.dumps(
                    {
                        "schema_version": "shopping-paired-statistics-v1",
                        "baseline": {"attempt_coverage_rate": 1.0},
                        "candidate": {"attempt_coverage_rate": 1.0},
                        "failure_profiles": {
                            "baseline": {
                                "infrastructure_invalid_attempts": 0,
                                "critical_footer_failures": 0,
                            },
                            "candidate": {
                                "infrastructure_invalid_attempts": 0,
                                "critical_footer_failures": 0,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "freeze.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "shopping-final-candidate-v1",
                        "git_commit": "commit",
                        "config_path": str(config),
                        "config_sha256": _sha256_file(config),
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_sha256": _sha256_tree(checkpoint),
                        "model_path": str(model),
                        "model_sha256": _sha256_tree(model),
                        "served_model_name": "terminal-grpo",
                        "evaluation_report_path": str(comparison),
                        "evaluation_report_sha256": _sha256_file(comparison),
                        "final_benchmark_sha256": _sha256_file(benchmark),
                    }
                ),
                encoding="utf-8",
            )

            validated = validate_frozen_candidate(
                manifest,
                served_model_name="terminal-grpo",
                benchmark=benchmark,
                root=root,
                git_commit="commit",
            )
            self.assertEqual(validated["model_sha256"], _sha256_tree(model))
            (model / "model.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(SystemExit, "model hash mismatch"):
                validate_frozen_candidate(
                    manifest,
                    served_model_name="terminal-grpo",
                    benchmark=benchmark,
                    root=root,
                    git_commit="commit",
                )

    def test_posthoc_repeated_output_can_resume_after_protocol_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "raw.jsonl"
            task_id = 123
            attempt_index = 1
            base_seed = 42
            material = f"{base_seed}:{task_id}:{attempt_index}".encode()
            attempt_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**31)
            output.write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "attempt_index": attempt_index,
                        "actor_sampling": {
                            "model": "shopping-agent",
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "base_seed": base_seed,
                            "attempt_seed": attempt_seed,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            count = validate_resumable_output(
                output,
                expected_task_ids=[task_id],
                attempts_per_task=3,
                model="shopping-agent",
                temperature=0.7,
                top_p=0.9,
                seed=base_seed,
            )
            self.assertEqual(count, 1)

    def test_posthoc_repeated_output_rejects_protocol_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "raw.jsonl"
            output.write_text(
                json.dumps(
                    {
                        "task_id": 123,
                        "attempt_index": 0,
                        "actor_sampling": {
                            "model": "different-model",
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "base_seed": 42,
                            "attempt_seed": 0,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(SystemExit, "protocol mismatch"):
                validate_resumable_output(
                    output,
                    expected_task_ids=[123],
                    attempts_per_task=3,
                    model="shopping-agent",
                    temperature=0.7,
                    top_p=0.9,
                    seed=42,
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
