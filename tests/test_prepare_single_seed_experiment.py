import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_single_seed_experiment import materialize, stratified_nested_order


def row(task_id: int, *, query: str, tool_steps: int) -> dict:
    messages = [{"role": "user", "content": f"Instruction: {query}"}]
    messages.extend({"role": "tool", "content": "observation"} for _ in range(tool_steps))
    messages.append({"role": "assistant", "content": "done"})
    return {
        "trajectory_id": f"trajectory-{task_id}",
        "task_id": task_id,
        "messages": messages,
        "tools": [],
    }


class PrepareSingleSeedExperimentTest(unittest.TestCase):
    def test_stratified_order_is_deterministic_and_nested(self):
        rows = [
            row(index, query=("红色杯子预算100" if index % 2 else "杯子"), tool_steps=index % 25)
            for index in range(379)
        ]
        first, _ = stratified_nested_order(rows, seed=42)
        second, _ = stratified_nested_order(list(reversed(rows)), seed=42)
        self.assertEqual(
            [item["task_id"] for item in first],
            [item["task_id"] for item in second],
        )
        self.assertTrue(
            {item["task_id"] for item in first[:95]}
            < {item["task_id"] for item in first[:190]}
        )

    def test_materialize_hashes_three_zero_overlap_subsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(
                "".join(
                    json.dumps(row(index, query="杯子", tool_steps=index % 4)) + "\n"
                    for index in range(379)
                ),
                encoding="utf-8",
            )
            evaluation = root / "evaluation.jsonl"
            evaluation.write_text(json.dumps({"task_id": 1000}) + "\n", encoding="utf-8")
            manifest = materialize(
                source,
                root / "output",
                evaluation_paths=(evaluation,),
                seed=42,
            )
            self.assertEqual(manifest["evaluation_overlap"], 0)
            self.assertEqual([manifest["subsets"][str(size)]["rows"] for size in (95, 190, 379)], [95, 190, 379])


if __name__ == "__main__":
    unittest.main()
