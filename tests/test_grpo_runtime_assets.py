"""正式 GRPO 入口必须把 Shopping 专用 AgentLoop 和 Vanilla 设置接通。"""

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class GrpoRuntimeAssetsTest(unittest.TestCase):
    def test_agent_loop_config_loads_project_wrapper(self):
        config = (ROOT / "configs/verl/shop_agent_loops.yaml").read_text(encoding="utf-8")
        self.assertIn("shopping_tool_agent", config)
        self.assertIn("shopping_grpo.verl_adapter.agent_loop.ShoppingToolAgentLoop", config)
        self.assertIn("reward_mode: ${oc.env:SHOPPING_REWARD_MODE,native}", config)

    def test_vanilla_config_uses_qwen_parser_and_environment_reward_only(self):
        config = (ROOT / "configs/verl/vanilla_grpo.yaml").read_text(encoding="utf-8")
        self.assertIn("adv_estimator: grpo", config)
        self.assertIn("format: qwen3_coder", config)
        self.assertIn("default_agent_loop: shopping_tool_agent", config)
        self.assertIn("use_remove_padding: false", config)
        self.assertIn("lora:\n      merge: true", config)
        self.assertIn("calculate_log_probs: true", config)
        self.assertIn("bypass_mode: true", config)
        self.assertIn("rollout_is: null", config)
        self.assertIn("rollout_rs: null", config)
        self.assertIn("loss_type: ppo_clip", config)
        self.assertIn("shopping_dynamic_sampling:", config)
        self.assertIn("enable: false", config)
        self.assertIn("metric: seq_reward", config)
        self.assertIn("max_num_gen_batches: 3", config)
        self.assertIn("reward_tolerance: 1.0e-8", config)
        self.assertIn("loss_agg_mode: token-mean", config)
        self.assertIn("clip_ratio_low: 0.20", config)
        self.assertIn("clip_ratio_high: 0.20", config)
        self.assertIn("calculate_entropy: false", config)
        self.assertIn("use_kl_loss: false", config)
        self.assertIn("use_kl_in_reward: false", config)
        self.assertIn(
            "worker_process_setup_hook: shopping_grpo.verl_compat.install_torch_padding_fallback",
            config,
        )
        self.assertIn("reward_model:\n  enable: false", config)
        self.assertNotIn("interaction_config_path", config)
        self.assertNotIn("prm", config.casefold())
        self.assertNotIn("lata", config.casefold())

    def test_a0_and_a1_are_parameterized_experiment_modes(self):
        launcher = ROOT / "scripts/run_vanilla_grpo.sh"

        a0 = subprocess.run(
            ["bash", str(launcher), "a0", "--dry-run"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        a1 = subprocess.run(
            ["bash", str(launcher), "a1", "--dry-run"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(a0.returncode, 0, a0.stderr)
        self.assertEqual(a1.returncode, 0, a1.stderr)
        self.assertIn("SHOPPING_GRPO_EXPERIMENT=a0", a0.stdout)
        self.assertIn("SHOPPING_REWARD_MODE=native", a0.stdout)
        self.assertIn("algorithm.norm_adv_by_std_in_grpo=true", a0.stdout)
        self.assertIn("shopping_dynamic_sampling.enable=false", a0.stdout)
        self.assertIn("SHOPPING_GRPO_EXPERIMENT=a1", a1.stdout)
        self.assertIn("SHOPPING_REWARD_MODE=constraint_aware", a1.stdout)
        self.assertIn("algorithm.norm_adv_by_std_in_grpo=false", a1.stdout)
        self.assertIn("shopping_dynamic_sampling.enable=true", a1.stdout)
        for output in (a0.stdout, a1.stdout):
            self.assertIn("actor_rollout_ref.actor.use_kl_loss=false", output)
            self.assertIn("actor_rollout_ref.actor.loss_agg_mode=token-mean", output)
            self.assertIn("actor_rollout_ref.actor.calculate_entropy=false", output)
            self.assertIn("actor_rollout_ref.actor.clip_ratio_low=0.20", output)
            self.assertIn("actor_rollout_ref.actor.clip_ratio_high=0.20", output)

    def test_server_launcher_uses_installed_verl_instead_of_reference_fork(self):
        launcher = ROOT / "scripts/run_vanilla_grpo.sh"
        self.assertTrue(launcher.is_file())
        content = launcher.read_text(encoding="utf-8")
        self.assertIn("verl.trainer.main_ppo", content)
        self.assertIn(
            'check_grpo_runtime.py" "${EXPERIMENT_OVERRIDES[@]}" "$@"',
            content,
        )
        self.assertNotIn("agentic-grpo-longhorizon", content)
        self.assertNotIn("shop_interaction.json", content)

    def test_runtime_setup_applies_the_numpy_override(self):
        setup = (ROOT / "docs/grpo-runtime-setup.md").read_text(encoding="utf-8")
        self.assertIn("--override requirements-grpo-overrides.txt", setup)
        self.assertIn("apply_verl_dynamic_sampling_patch.py", setup)
        self.assertNotIn("uv pip check", setup)

    def test_pinned_verl_patch_logs_comparable_group_metrics(self):
        patch = (
            ROOT / "patches/verl-0.8.0-shopping-dynamic-sampling.patch"
        ).read_text(encoding="utf-8")
        for metric in (
            "group/generated",
            "group/trained",
            "group/effective_ratio",
            "group/all_equal_ratio",
            "group/all_zero_semantic_ratio",
            "group/all_full_success_ratio",
            "group/infrastructure_invalid",
            "group/resample_batches",
            "rollout/generated_total",
        ):
            self.assertIn(metric, patch)

    def test_grpo_dependencies_pin_the_supported_runtime(self):
        requirements = (ROOT / "requirements-grpo.txt").read_text(encoding="utf-8")
        self.assertIn("verl==0.8.0", requirements)
        self.assertIn("vllm==0.25.1", requirements)
        self.assertIn("transformers==5.11.0", requirements)
        self.assertIn("tensordict==0.10.0", requirements)
        self.assertIn("numpy==2.2.6", requirements)
        override = (ROOT / "requirements-grpo-overrides.txt").read_text(encoding="utf-8")
        self.assertEqual(override.strip(), "numpy==2.2.6")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
