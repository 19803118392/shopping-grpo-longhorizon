import json
from pathlib import Path
import tempfile
import unittest

from web_agent_site.engine.environment_v2_config import (
    load_environment_v2_config,
    validate_environment_v2_config,
)


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "environment_v2.json"


class EnvironmentV2ConfigTest(unittest.TestCase):
    def test_repository_config_matches_runtime_contract(self):
        config = load_environment_v2_config(CONFIG)
        self.assertEqual(config["search"]["page_size"], 20)
        self.assertEqual(config["reward"]["gold_purchase"], 1.0)

    def test_reward_drift_is_rejected(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["reward"]["wrong_purchase"] = 0.2
        with self.assertRaisesRegex(ValueError, "reward values"):
            validate_environment_v2_config(config)

    def test_malformed_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot load"):
                load_environment_v2_config(path)


if __name__ == "__main__":
    unittest.main()
