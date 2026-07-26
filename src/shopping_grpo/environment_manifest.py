"""Lightweight cross-repository contract for Environment v2 experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


MANIFEST_VERSION = "shopping-environment-manifest-v1"
EMBEDDED_SOURCE_FILE = "EMBEDDED_SOURCE.json"
REQUIRED_KEYS = {
    "manifest_version",
    "shopsimulator_commit",
    "shopping_grpo_commit",
    "product_data_sha256",
    "task_data_sha256",
    "search",
    "reward",
    "observation_version",
    "tool_version",
    "max_steps",
    "seed",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repository):
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def shopsimulator_source_commit(repository):
    repository = Path(repository)
    embedded_source = repository / EMBEDDED_SOURCE_FILE
    if not embedded_source.is_file():
        return git_commit(repository)
    try:
        metadata = json.loads(embedded_source.read_text(encoding="utf-8"))
        commit = metadata["environment_v2_commit"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid embedded ShopSimulator source metadata: {exc}") from exc
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("embedded ShopSimulator commit is not a lowercase Git SHA")
    return commit


def build_manifest(
    *,
    shopsimulator_repository,
    shopping_grpo_repository,
    product_data,
    task_data,
    environment_config,
    seed,
):
    config = json.loads(Path(environment_config).read_text(encoding="utf-8"))
    return {
        "manifest_version": MANIFEST_VERSION,
        "shopsimulator_commit": shopsimulator_source_commit(
            shopsimulator_repository
        ),
        "shopping_grpo_commit": git_commit(shopping_grpo_repository),
        "product_data_sha256": sha256_file(product_data),
        "task_data_sha256": sha256_file(task_data),
        "search": config["search"],
        "reward": config["reward"],
        "observation_version": config["observation_version"],
        "tool_version": config["tool_version"],
        "max_steps": int(config["termination"]["max_steps"]),
        "seed": int(seed),
    }


def validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("environment manifest must be an object")
    missing = REQUIRED_KEYS - set(manifest)
    if missing:
        raise ValueError(
            "environment manifest is missing: " + ", ".join(sorted(missing))
        )
    if manifest["manifest_version"] != MANIFEST_VERSION:
        raise ValueError("unsupported environment manifest version")
    if manifest["observation_version"] != "shopping-observation-v2":
        raise ValueError("manifest does not select Observation v2")
    if manifest["tool_version"] != "shopping-tools-v2":
        raise ValueError("manifest does not select Tool v2")
    if manifest["reward"].get("version") != "shopsimulator-reward-v2":
        raise ValueError("manifest does not select Reward v2")
    if manifest["search"].get("version") != "shopsimulator-multifield-bm25-v2":
        raise ValueError("manifest does not select multi-field BM25 v2")
    if int(manifest["search"].get("page_size", 0)) != 20:
        raise ValueError("Environment v2 page_size must equal 20")
    if int(manifest["max_steps"]) <= 0:
        raise ValueError("max_steps must be positive")
    for name in (
        "shopsimulator_commit",
        "shopping_grpo_commit",
        "product_data_sha256",
        "task_data_sha256",
    ):
        value = manifest[name]
        expected_length = 40 if name.endswith("_commit") else 64
        if (
            not isinstance(value, str)
            or len(value) != expected_length
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"manifest {name} is not a lowercase hexadecimal digest")
    return manifest


def write_manifest(path, manifest):
    validate_manifest(manifest)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite environment manifest: {output}")
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
