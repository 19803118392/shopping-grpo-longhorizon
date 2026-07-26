import unittest

from shopping_grpo.environment_manifest import (
    MANIFEST_VERSION,
    validate_manifest,
)


class EnvironmentManifestTest(unittest.TestCase):
    def test_environment_v2_contract_is_validated(self):
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "shopsimulator_commit": "a" * 40,
            "shopping_grpo_commit": "b" * 40,
            "product_data_sha256": "c" * 64,
            "task_data_sha256": "d" * 64,
            "search": {
                "version": "shopsimulator-multifield-bm25-v2",
                "page_size": 20,
            },
            "reward": {"version": "shopsimulator-reward-v2"},
            "observation_version": "shopping-observation-v2",
            "tool_version": "shopping-tools-v2",
            "max_steps": 35,
            "seed": 20260726,
        }
        self.assertIs(validate_manifest(manifest), manifest)

    def test_page_size_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_manifest({})

    def test_wrong_tool_contract_is_rejected(self):
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "shopsimulator_commit": "a" * 40,
            "shopping_grpo_commit": "b" * 40,
            "product_data_sha256": "c" * 64,
            "task_data_sha256": "d" * 64,
            "search": {
                "version": "shopsimulator-multifield-bm25-v2",
                "page_size": 20,
            },
            "reward": {"version": "shopsimulator-reward-v2"},
            "observation_version": "shopping-observation-v2",
            "tool_version": "shopping-tools-v1",
            "max_steps": 35,
            "seed": 20260726,
        }
        with self.assertRaisesRegex(ValueError, "Tool v2"):
            validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
