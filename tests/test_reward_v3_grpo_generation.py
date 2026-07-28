import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.prepare_grpo_reward_v3_fresh_v1 import (
    ENVIRONMENT_VERSION,
    REWARD_VERSION,
    proportional_targets,
    validate_probe_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def trajectory(task_id, steps, reward_type="gold_purchase"):
    terminal_utility = 1.0 if reward_type == "gold_purchase" else -0.5
    return {
        "task_id": task_id,
        "status": "done",
        "done": True,
        "initial_result": {"environment_version": ENVIRONMENT_VERSION},
        "steps": [{}] * steps,
        "terminal_result": {
            "termination_reason": reward_type,
            "reward_detail": {
                "reward_version": REWARD_VERSION,
                "reward_type": reward_type,
                "termination_reason": reward_type,
                "reward_valid": True,
                "purchase_success": reward_type == "gold_purchase",
                "terminal_utility": terminal_utility,
            },
        },
    }


class RewardV3GrpoGenerationTest(unittest.TestCase):
    def test_proportional_targets_are_exact_and_within_capacity(self):
        targets = proportional_targets(
            {"short": 400, "medium": 600, "long": 1000},
            1000,
        )
        self.assertEqual(targets, {"short": 200, "medium": 300, "long": 500})

    def test_probe_contract_requires_environment_v2_1(self):
        rows = [{"task_id": 1}]
        probes = [trajectory(1, 8)]
        report = validate_probe_contract(rows, probes)
        self.assertEqual(report["probe_task_count"], 1)
        probes[0]["initial_result"]["environment_version"] = "shopsimulator-environment-v1"
        with self.assertRaisesRegex(ValueError, "did not run"):
            validate_probe_contract(rows, probes)

    def test_probe_contract_requires_reward_v3_for_terminal_rows(self):
        rows = [{"task_id": 1}]
        probes = [trajectory(1, 8)]
        probes[0]["terminal_result"]["reward_detail"]["reward_version"] = (
            "shopsimulator-reward-v2"
        )
        with self.assertRaisesRegex(ValueError, "not Reward v3"):
            validate_probe_contract(rows, probes)

    def test_reward_v3_launcher_isolated_assets(self):
        launcher = ROOT / "scripts/run_grpo_reward_v3_fresh_v1.sh"
        result = subprocess.run(
            ["bash", str(launcher), "a1", "--dry-run"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Environment-v2.1/Reward-v3/fresh-v1", result.stdout)
        self.assertIn("vanilla_grpo_reward_v3_fresh_v1", result.stdout)
        self.assertIn("shop_tools_v2.json", result.stdout)
        self.assertIn("grpo_reward_v3_fresh_v1_train.parquet", result.stdout)
        self.assertNotIn("grpo_train_v1.parquet", result.stdout)

    def test_legacy_launcher_is_refused_by_default(self):
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/run_vanilla_grpo.sh"), "a0", "--dry-run"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("archived", result.stderr)

    def test_probe_service_launcher_pins_reward_v3_runtime(self):
        content = (
            ROOT / "scripts/start_reward_v3_probe_services.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("run_environment_v2_1.sh", content)
        self.assertIn("qwen35-2b-sft-v1-fresh-merged", content)
        self.assertIn("VLLM_USE_FLASHINFER_SAMPLER=0", content)
        self.assertIn("--tool-call-parser qwen3_xml", content)


if __name__ == "__main__":
    unittest.main()
