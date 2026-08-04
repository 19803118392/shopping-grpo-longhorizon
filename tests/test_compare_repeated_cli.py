import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.compare_repeated_evaluations import main


def row(success):
    reward_type = "gold_purchase" if success else "wrong_purchase"
    return {
        "task_id": 1,
        "attempt_index": 0,
        "status": "done",
        "done": True,
        "terminal_result": {
            "done": True,
            "over": True,
            "reward_detail": {
                "reward_version": "shopsimulator-reward-v3",
                "reward_type": reward_type,
                "reward_valid": True,
                "purchase_success": bool(success),
                "termination_reason": reward_type,
            },
        },
    }


class CompareRepeatedCliTest(unittest.TestCase):
    def test_cli_writes_machine_readable_report_with_input_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "tasks.jsonl"
            baseline = root / "baseline.jsonl"
            candidate = root / "candidate.jsonl"
            output = root / "comparison.json"
            benchmark.write_text('{"task_id":1}\n', encoding="utf-8")
            baseline.write_text(json.dumps(row(False)) + "\n", encoding="utf-8")
            candidate.write_text(json.dumps(row(True)) + "\n", encoding="utf-8")
            with patch.object(
                sys,
                "argv",
                [
                    "compare_repeated_evaluations.py",
                    "--benchmark",
                    str(benchmark),
                    "--baseline",
                    str(baseline),
                    "--candidate",
                    str(candidate),
                    "--output",
                    str(output),
                    "--attempts-per-task",
                    "1",
                    "--bootstrap-samples",
                    "20",
                ],
            ), patch("builtins.print"):
                main()

            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(
            report["paired_task_delta"][
                "candidate_minus_baseline_percentage_points"
            ],
            100.0,
        )
        self.assertEqual(len(report["provenance"]["baseline_sha256"]), 64)
        self.assertEqual(report["provenance"]["bootstrap_samples"], 20)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
