"""End-to-end test for the public CLI from a non-editable wheel install."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import venv
import zipfile
from pathlib import Path


@unittest.skipUnless(
    importlib.util.find_spec("build"),
    "the wheel test requires the dev extra",
)
class WheelInstallTest(unittest.TestCase):
    def test_wheel_contains_blind_resources_and_public_cli_runs(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            dist = temporary / "dist"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--wheel",
                    "--outdir",
                    str(dist),
                    str(root),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            wheel = next(dist.glob("shopping_grpo-*.whl"))
            with zipfile.ZipFile(wheel) as archive:
                members = set(archive.namelist())
            self.assertIn(
                "shopping_grpo/resources/blind_guard.json",
                members,
            )
            self.assertIn(
                "shopping_grpo/resources/blind_final_task_ids.json",
                members,
            )

            environment = temporary / "venv"
            binary_directory = "Scripts" if os.name == "nt" else "bin"
            python = environment / binary_directory / "python"
            cli = environment / binary_directory / "shopping-grpo"
            uv = shutil.which("uv")
            if uv:
                subprocess.run(
                    [uv, "venv", str(environment)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    [
                        uv,
                        "pip",
                        "install",
                        "--python",
                        str(python),
                        "--no-deps",
                        str(wheel),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                venv.EnvBuilder(with_pip=True).create(environment)
                pip = environment / binary_directory / "pip"
                subprocess.run(
                    [str(pip), "install", "--no-deps", str(wheel)],
                    check=True,
                    capture_output=True,
                    text=True,
                )

            smoke = subprocess.run(
                [str(cli), "smoke", "--json"],
                cwd=temporary,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                len(json.loads(smoke.stdout)["checks"]),
                7,
            )

            trajectory = temporary / "non_blind_example.jsonl"
            trajectory.write_bytes(
                (root / "examples/trajectories.jsonl").read_bytes()
            )
            evaluated = subprocess.run(
                [str(cli), "evaluate", str(trajectory)],
                cwd=temporary,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                json.loads(evaluated.stdout)["trajectory_count"],
                3,
            )

            disguised_final = temporary / "renamed_input.txt"
            disguised_final.write_text(
                '{"task_id":8187,"trajectory_id":"blocked"}\n',
                encoding="utf-8",
            )
            blocked = subprocess.run(
                [str(cli), "evaluate", str(disguised_final)],
                cwd=temporary,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn("frozen blind-final tasks", blocked.stderr)


if __name__ == "__main__":
    unittest.main()
