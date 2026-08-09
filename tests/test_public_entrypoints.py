"""Public CPU and parameterized GRPO entry-point tests."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.merge_lora_adapter import build_merge_manifest
from scripts.train_grpo import _sha256_file, _sha256_tree, build_command, parse_args
from scripts.train_grpo import main as train_grpo_main
from shopping_grpo.cli import main as cli_main
from shopping_grpo.smoke import run_cpu_smoke


class PublicEntrypointTest(unittest.TestCase):
    def test_grpo_merge_manifest_records_actor_checkpoint(self):
        manifest = build_merge_manifest(
            "intermediate",
            "intermediate/lora_adapter",
            "merged",
            "qwen3_5",
            source_checkpoint="global_step_30/actor",
        )

        self.assertEqual(manifest["operation"], "peft_merge_and_unload")
        self.assertEqual(
            manifest["source"]["grpo_actor_checkpoint"],
            "global_step_30/actor",
        )
        self.assertEqual(
            manifest["next_step"],
            "serve this standalone checkpoint directly",
        )

    def test_cpu_smoke_covers_public_contracts(self):
        result = run_cpu_smoke()

        self.assertEqual(
            result["checks"],
            [
                "action_schema",
                "trajectory_normalization",
                "reward_sample",
                "sft_label_mask",
                "dynamic_sampling_grouping",
            ],
        )

    def test_offline_example_cli_runs_without_models_or_environment(self):
        root = Path(__file__).resolve().parents[1]
        with (
            patch.object(
                sys,
                "argv",
                [
                    "shopping-grpo",
                    "evaluate",
                    str(root / "examples/trajectories.jsonl"),
                ],
            ),
            patch("builtins.print") as output,
        ):
            cli_main()

        summary = json.loads(output.call_args.args[0])
        self.assertEqual(summary["trajectory_count"], 3)
        self.assertEqual(summary["strict_gold_success_count"], 1)

    def test_public_grpo_launcher_accepts_sharded_weights_and_console(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors.index.json").write_text(
                "{}",
                encoding="utf-8",
            )
            train = temporary / "train.parquet"
            train.write_bytes(b"example")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"example")
            output = temporary / "output"
            with patch.object(
                sys,
                "argv",
                [
                    "train_grpo.py",
                    "--model",
                    str(model),
                    "--train-data",
                    str(train),
                    "--val-data",
                    str(validation),
                    "--output",
                    str(output),
                    "--config",
                    str(root / "configs/grpo.yaml"),
                    "--logger",
                    "console",
                    "--dry-run",
                ],
            ):
                args = parse_args()
            command, environment = build_command(args)

        self.assertIn("verl.trainer.main_ppo", command)
        self.assertEqual(environment["GRPO_MODEL_PATH"], str(model))
        self.assertEqual(environment["GRPO_TRAIN_FILE"], str(train))
        self.assertEqual(environment["GRPO_VAL_FILE"], str(validation))
        self.assertEqual(environment["SHOPPING_OPTIMIZATION_REWARD_PROFILE"], "environment_v3")
        self.assertIn("trainer.logger=[console]", command)

    def test_grpo_preflight_reuses_the_exact_training_overrides(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            train = temporary / "train.parquet"
            train.write_bytes(b"train")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"validation")
            output = temporary / "output"
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "train_grpo.py",
                        "--model",
                        str(model),
                        "--train-data",
                        str(train),
                        "--val-data",
                        str(validation),
                        "--output",
                        str(output),
                        "--config",
                        str(root / "configs/grpo.yaml"),
                        "--",
                        "trainer.total_training_steps=5",
                    ],
                ),
                patch("scripts.train_grpo.subprocess.call", side_effect=[0, 0]) as subprocess_call,
                patch("builtins.print"),
                self.assertRaisesRegex(SystemExit, "0"),
            ):
                train_grpo_main()

        preflight = subprocess_call.call_args_list[0].args[0]
        training = subprocess_call.call_args_list[1].args[0]
        expected_overrides = [
            "trainer.logger=[console]",
            "trainer.experiment_name=shopping-agent-grpo",
            "trainer.resume_mode=disable",
            "trainer.total_training_steps=5",
        ]
        self.assertEqual(preflight[2:], expected_overrides)
        self.assertEqual(training[-4:], expected_overrides)

    def test_resume_is_explicit_scoped_and_keeps_validation_frozen(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            train = temporary / "stage2.parquet"
            train.write_bytes(b"stage2")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"validation")
            output = temporary / "output"
            checkpoint = output / "global_step_5"
            (checkpoint / "actor").mkdir(parents=True)
            (checkpoint / "actor/model.bin").write_bytes(b"checkpoint")
            (checkpoint / "data.pt").write_bytes(b"loader")
            (checkpoint / "shopping_state.pt").write_bytes(b"adaptive")
            (output / "shopping_experiment_manifest.json").write_text(
                json.dumps(
                    {
                        "initial_train_sha256": hashlib.sha256(b"stage2").hexdigest(),
                        "validation_sha256": hashlib.sha256(b"validation").hexdigest(),
                        "config_sha256": _sha256_file(root / "configs/grpo.yaml"),
                        "model_sha256": _sha256_tree(model),
                        "frozen_settings": {
                            "model_path": str(model.resolve()),
                            "seed": 2026,
                            "optimization_reward": "v3",
                            "hydra_overrides": [],
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(
                sys,
                "argv",
                [
                    "train_grpo.py",
                    "--model",
                    str(model),
                    "--train-data",
                    str(train),
                    "--val-data",
                    str(validation),
                    "--output",
                    str(output),
                    "--resume-from",
                    str(checkpoint),
                    "--target-global-step",
                    "10",
                    "--config",
                    str(root / "configs/grpo.yaml"),
                    "--dry-run",
                ],
            ):
                args = parse_args()
            command, environment = build_command(args)

        self.assertIn("trainer.resume_mode=resume_path", command)
        self.assertIn(f"trainer.resume_from_path={checkpoint}", command)
        self.assertIn("trainer.total_training_steps=10", command)
        self.assertEqual(environment["SHOPPING_TRAINING_SEED"], "2026")

    def test_launcher_owned_hydra_fields_cannot_be_overridden(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            train = temporary / "train.parquet"
            train.write_bytes(b"train")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"validation")
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "train_grpo.py",
                        "--model",
                        str(model),
                        "--train-data",
                        str(train),
                        "--val-data",
                        str(validation),
                        "--output",
                        str(temporary / "output"),
                        "--config",
                        str(root / "configs/grpo.yaml"),
                        "--target-global-step",
                        "5",
                        "--dry-run",
                        "--",
                        "trainer.total_training_steps=999",
                    ],
                ),
                self.assertRaisesRegex(SystemExit, "owned by the experiment launcher"),
            ):
                build_command(parse_args())

    def test_resume_cannot_escape_the_experiment_output(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            train = temporary / "train.parquet"
            train.write_bytes(b"train")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"validation")
            output = temporary / "output"
            output.mkdir()
            foreign = temporary / "foreign/global_step_5"
            (foreign / "actor").mkdir(parents=True)
            (foreign / "data.pt").write_bytes(b"loader")
            (foreign / "shopping_state.pt").write_bytes(b"adaptive")
            with patch.object(
                sys,
                "argv",
                [
                    "train_grpo.py",
                    "--model",
                    str(model),
                    "--train-data",
                    str(train),
                    "--val-data",
                    str(validation),
                    "--output",
                    str(output),
                    "--resume-from",
                    str(foreign),
                    "--config",
                    str(root / "configs/grpo.yaml"),
                    "--dry-run",
                ],
            ):
                args = parse_args()
            with self.assertRaisesRegex(SystemExit, "direct child"):
                build_command(args)

    def test_preflight_only_does_not_create_output_or_start_training(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            model = temporary / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            train = temporary / "train.parquet"
            train.write_bytes(b"train")
            validation = temporary / "validation.parquet"
            validation.write_bytes(b"validation")
            output = temporary / "preflight-output"
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "train_grpo.py",
                        "--model",
                        str(model),
                        "--train-data",
                        str(train),
                        "--val-data",
                        str(validation),
                        "--output",
                        str(output),
                        "--config",
                        str(root / "configs/grpo.yaml"),
                        "--preflight-only",
                    ],
                ),
                patch("scripts.train_grpo.subprocess.call", return_value=0) as subprocess_call,
                patch("builtins.print"),
            ):
                train_grpo_main()

            self.assertEqual(subprocess_call.call_count, 1)
            self.assertIn("check_grpo_runtime.py", subprocess_call.call_args.args[0][1])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
