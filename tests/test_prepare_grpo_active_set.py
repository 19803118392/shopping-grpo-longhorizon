import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.prepare_grpo_active_set import build_screen, materialize_active_set


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _train_row(task_id: int) -> dict:
    return {
        "data_source": "shopsimulator",
        "prompt": [
            {"role": "system", "content": "shop safely"},
            {
                "role": "user",
                "content": ("Instruction: 想要耳机，必须黑色，需要蓝牙，最好降噪，预算不超过500元"),
            },
        ],
        "ability": "shopping",
        "reward_model": {"style": "rule", "ground_truth": None},
        "extra_info": {"split": "train", "index": task_id, "task_id": task_id},
    }


def _trajectory(task_id: int, attempt: int, reward: float, **overrides: object) -> dict:
    row = {
        "task_id": task_id,
        "attempt_index": attempt,
        "status": "done",
        "final_reward": reward,
        "terminal_result": {
            "reward_detail": {
                "reward": reward,
                "reward_valid": True,
                "reward_version": "shopsimulator-reward-v3",
                "reward_type": "gold_purchase" if reward == 1.0 else "wrong_purchase",
            }
        },
    }
    row.update(overrides)
    return row


class PrepareGrpoActiveSetTest(unittest.TestCase):
    def test_build_screen_balances_reference_length_buckets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.parquet"
            metadata = root / "train.jsonl"
            output = root / "screen.jsonl"
            rows = [_train_row(task_id) for task_id in range(1, 7)]
            pq.write_table(pa.Table.from_pylist(rows), train)
            _write_jsonl(
                metadata,
                [
                    {
                        "task_id": task_id,
                        "length_bucket": ("short", "medium", "long")[(task_id - 1) % 3],
                        "probe_steps": task_id,
                    }
                    for task_id in range(1, 7)
                ],
            )

            report = build_screen(
                train,
                metadata,
                output,
                screen_size=3,
                seed=42,
            )

            self.assertEqual(report["screen_size"], 3)
            self.assertEqual(report["length_buckets"], {"short": 1, "medium": 1, "long": 1})
            self.assertEqual(report["constraint_buckets"], {"4+": 3})
            self.assertEqual(report["option_tasks"], 3)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 3)

    def test_materialize_selects_only_complete_valid_varying_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.parquet"
            screening = root / "screening.jsonl"
            output = root / "active.parquet"
            pq.write_table(
                pa.Table.from_pylist([_train_row(task_id) for task_id in (10, 20, 30)]),
                train,
            )
            rows = []
            for attempt, reward in enumerate((0.0, 1.0, 0.0, 1.0)):
                rows.append(_trajectory(10, attempt, reward))
            for attempt in range(4):
                rows.append(_trajectory(20, attempt, 0.0))
            for attempt in range(4):
                rows.append(
                    _trajectory(30, attempt, 0.0, status="max_steps" if attempt == 3 else "done")
                )
            _write_jsonl(screening, rows)

            report = materialize_active_set(
                train,
                screening,
                output,
                attempts_per_task=4,
                tolerance=1.0e-8,
            )

            self.assertEqual(report["selected_task_ids"], [10])
            self.assertEqual(report["rejected"], {"constant_reward": 1, "non_terminal": 1})
            selected = pq.read_table(output).to_pylist()
            self.assertEqual([row["extra_info"]["task_id"] for row in selected], [10])

    def test_materialize_rejects_duplicate_task_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.parquet"
            screening = root / "screening.jsonl"
            output = root / "active.parquet"
            pq.write_table(pa.Table.from_pylist([_train_row(10)]), train)
            rows = [_trajectory(10, attempt, float(attempt % 2)) for attempt in range(4)]
            rows.append(_trajectory(10, 1, 1.0))
            _write_jsonl(screening, rows)

            with self.assertRaisesRegex(ValueError, "duplicate screening trajectory"):
                materialize_active_set(
                    train,
                    screening,
                    output,
                    attempts_per_task=4,
                    tolerance=1.0e-8,
                )

    def test_materialize_rejects_evaluation_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.parquet"
            screening = root / "screening.jsonl"
            evaluation = root / "evaluation.jsonl"
            output = root / "active.parquet"
            pq.write_table(pa.Table.from_pylist([_train_row(10)]), train)
            _write_jsonl(
                screening,
                [_trajectory(10, attempt, float(attempt % 2)) for attempt in range(4)],
            )
            _write_jsonl(evaluation, [{"task_id": 10}])

            with self.assertRaisesRegex(ValueError, "active-set/evaluation overlap"):
                materialize_active_set(
                    train,
                    screening,
                    output,
                    attempts_per_task=4,
                    tolerance=1.0e-8,
                    evaluation_task_files=(evaluation,),
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
