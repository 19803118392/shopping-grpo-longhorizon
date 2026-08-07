import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_grpo_runtime import validate_training_artifacts


class GrpoArtifactPreflightTest(unittest.TestCase):
    def test_hashes_model_and_accepts_matching_canonical_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            data = root / "data/grpo"
            data.mkdir(parents=True)
            train = data / "train.parquet"
            validation = data / "validation.parquet"
            train.write_bytes(b"train")
            validation.write_bytes(b"validation")
            metadata = {
                "train": {
                    "parquet": "data/grpo/train.parquet",
                    "parquet_sha256": hashlib.sha256(b"train").hexdigest(),
                },
                "validation": {
                    "parquet": "data/grpo/validation.parquet",
                    "parquet_sha256": hashlib.sha256(b"validation").hexdigest(),
                },
            }
            (data / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            audit = validate_training_artifacts(model, train, validation, root=root)

            self.assertEqual(len(audit["model"]["fingerprint_sha256"]), 64)
            self.assertEqual(
                audit["datasets"]["train"]["sha256"],
                metadata["train"]["parquet_sha256"],
            )

    def test_rejects_modified_canonical_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            data = root / "data/grpo"
            data.mkdir(parents=True)
            train = data / "train.parquet"
            validation = data / "validation.parquet"
            train.write_bytes(b"modified")
            validation.write_bytes(b"validation")
            metadata = {
                "train": {
                    "parquet": "data/grpo/train.parquet",
                    "parquet_sha256": "0" * 64,
                },
                "validation": {
                    "parquet": "data/grpo/validation.parquet",
                    "parquet_sha256": hashlib.sha256(b"validation").hexdigest(),
                },
            }
            (data / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "train parquet hash mismatch"):
                validate_training_artifacts(model, train, validation, root=root)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
