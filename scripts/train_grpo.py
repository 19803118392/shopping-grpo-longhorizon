#!/usr/bin/env python3
"""Run the repository's single supported Shopping Agent GRPO recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/grpo.yaml"
DEFAULT_AGENT_CONFIG = ROOT / "configs/agent_loop.yaml"
DEFAULT_TOOL_CONFIG = ROOT / "configs/tools.json"
DEFAULT_MANIFEST = ROOT / "data/environment.json"
DEFAULT_MODEL = ROOT / "outputs/models/sft-merged"
DEFAULT_TRAIN_DATA = ROOT / "data/grpo/train.parquet"
DEFAULT_VAL_DATA = ROOT / "data/grpo/validation.parquet"
EXPERIMENT_MANIFEST = "shopping_experiment_manifest.json"
OWNED_HYDRA_KEYS = {
    "actor_rollout_ref.actor.data_loader_seed",
    "actor_rollout_ref.actor.fsdp_config.full_determinism",
    "actor_rollout_ref.actor.fsdp_config.seed",
    "actor_rollout_ref.rollout.engine_kwargs.vllm.seed",
    "data.seed",
    "data.train_files",
    "data.val_files",
    "trainer.default_local_dir",
    "trainer.resume_from_path",
    "trainer.resume_mode",
}


def _model_has_weights(path: Path) -> bool:
    candidates = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any((path / name).is_file() for name in candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--val-data", type=Path, default=DEFAULT_VAL_DATA)
    parser.add_argument("--env-url", default="http://127.0.0.1:5700")
    parser.add_argument("--output", type=Path, default=Path("outputs/models/grpo"))
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="explicit global_step_N checkpoint inside --output",
    )
    parser.add_argument(
        "--target-global-step",
        type=int,
        help="set total_training_steps and save_freq to this cumulative step",
    )
    parser.add_argument(
        "--logger",
        choices=("console", "swanlab"),
        default="console",
    )
    parser.add_argument("--experiment-name", default="shopping-agent-grpo")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--optimization-reward",
        choices=("v3", "v4"),
        default="v3",
        help="v3 uses the environment utility; v4 uses the adapter-only ASIN-neutral objective",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="validate paths and print the veRL command without runtime checks",
    )
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="run the complete version/hash/CUDA preflight without starting training",
    )
    parser.add_argument(
        "hydra_overrides",
        nargs=argparse.REMAINDER,
        help="additional veRL Hydra overrides after --",
    )
    return parser.parse_args()


def _validated_path(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"{description} does not exist: {resolved}")
    return resolved


def _extra_overrides(args: argparse.Namespace) -> list[str]:
    extra = list(args.hydra_overrides)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    protected = set(OWNED_HYDRA_KEYS)
    if args.target_global_step is not None:
        protected.update({"trainer.save_freq", "trainer.total_training_steps"})
    for override in extra:
        key = override.split("=", 1)[0].lstrip("+~")
        if key in protected:
            raise SystemExit(f"Hydra override {key!r} is owned by the experiment launcher")
    return extra


def _training_overrides(args: argparse.Namespace) -> list[str]:
    logger_override = (
        "trainer.logger=[console,swanlab]"
        if args.logger == "swanlab"
        else "trainer.logger=[console]"
    )
    overrides = [
        logger_override,
        f"trainer.experiment_name={args.experiment_name}",
    ]
    if args.resume_from is not None:
        checkpoint = args.resume_from.expanduser().resolve()
        overrides.extend(
            [
                "trainer.resume_mode=resume_path",
                f"trainer.resume_from_path={checkpoint}",
            ]
        )
    else:
        overrides.append("trainer.resume_mode=disable")
    if args.target_global_step is not None:
        overrides.extend(
            [
                f"trainer.total_training_steps={args.target_global_step}",
                f"trainer.save_freq={args.target_global_step}",
            ]
        )
    overrides.extend(_extra_overrides(args))
    return overrides


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise SystemExit(f"artifact tree contains no files: {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"global_step_([1-9][0-9]*)", path.name)
    if not match:
        raise SystemExit("--resume-from must name a global_step_N checkpoint")
    return int(match.group(1))


def _frozen_settings(args: argparse.Namespace) -> dict:
    return {
        "model_path": str(args.model.expanduser().resolve()),
        "seed": int(args.seed),
        "optimization_reward": str(args.optimization_reward),
        "hydra_overrides": _extra_overrides(args),
    }


def _validate_resume(
    args: argparse.Namespace, output: Path, train_data: Path, val_data: Path
) -> None:
    if args.resume_from is None:
        return
    checkpoint = _validated_path(args.resume_from, "resume checkpoint")
    if not checkpoint.is_dir() or checkpoint.parent != output:
        raise SystemExit("--resume-from must be a direct child of the same --output directory")
    step = _checkpoint_step(checkpoint)
    if not (checkpoint / "actor").is_dir():
        raise SystemExit(f"resume checkpoint is missing actor state: {checkpoint / 'actor'}")
    if not (checkpoint / "data.pt").is_file():
        raise SystemExit(f"resume checkpoint is missing dataloader state: {checkpoint / 'data.pt'}")
    if not (checkpoint / "shopping_state.pt").is_file():
        raise SystemExit(
            f"resume checkpoint is missing adaptive state: {checkpoint / 'shopping_state.pt'}"
        )
    if args.target_global_step is not None and args.target_global_step <= step:
        raise SystemExit("--target-global-step must be greater than the resumed global step")

    manifest_path = output / EXPERIMENT_MANIFEST
    if not manifest_path.is_file():
        raise SystemExit(f"resume output is missing {EXPERIMENT_MANIFEST}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid resume experiment manifest: {exc}") from exc
    if manifest.get("validation_sha256") != _sha256_file(val_data):
        raise SystemExit("resume validation parquet differs from the frozen experiment validation")
    if manifest.get("config_sha256") != _sha256_file(args.config.expanduser().resolve()):
        raise SystemExit("resume GRPO config differs from the frozen experiment config")
    if manifest.get("model_sha256") != _sha256_tree(args.model.expanduser().resolve()):
        raise SystemExit("resume model differs from the frozen SFT initialization")
    if manifest.get("frozen_settings") != _frozen_settings(args):
        raise SystemExit("resume changed a frozen model/seed/Hydra setting")
    if manifest.get("initial_train_sha256") != _sha256_file(train_data):
        raise SystemExit("resume train parquet differs from the frozen experiment train data")


def _manifest_payload(
    args: argparse.Namespace, environment: dict[str, str], command: list[str]
) -> dict:
    train = Path(environment["GRPO_TRAIN_FILE"])
    validation = Path(environment["GRPO_VAL_FILE"])
    config = Path(environment["GRPO_CONFIG_PATH"])
    payload = {
        "schema_version": "shopping-grpo-experiment-manifest-v1",
        "model_path": environment["GRPO_MODEL_PATH"],
        "model_sha256": _sha256_tree(Path(environment["GRPO_MODEL_PATH"])),
        "initial_train_path": str(train),
        "initial_train_sha256": _sha256_file(train),
        "validation_path": str(validation),
        "validation_sha256": _sha256_file(validation),
        "config_path": str(config),
        "config_sha256": _sha256_file(config),
        "command_sha256": hashlib.sha256(
            json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "frozen_settings": _frozen_settings(args),
    }
    return payload


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    model = _validated_path(args.model, "model directory")
    if not model.is_dir() or not (model / "config.json").is_file():
        raise SystemExit(f"model directory is missing config.json: {model}")
    if not _model_has_weights(model):
        raise SystemExit(f"model directory has no supported weight file or sharded index: {model}")
    train_data = _validated_path(args.train_data, "train parquet")
    val_data = _validated_path(args.val_data, "validation parquet")
    config = _validated_path(args.config, "GRPO example config")
    output = args.output.expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise SystemExit(f"output must be a directory: {output}")
        if args.resume_from is None and any(output.iterdir()):
            raise SystemExit(f"output directory must be new or empty: {output}")
    elif args.resume_from is not None:
        raise SystemExit(f"resume output directory does not exist: {output}")
    _validate_resume(args, output, train_data, val_data)
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    if args.logger == "swanlab" and not os.environ.get("SWANLAB_API_KEY"):
        raise SystemExit("--logger swanlab requires SWANLAB_API_KEY")

    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "SHOPPING_GRPO_ROOT": str(ROOT),
            "SHOPPING_ENVIRONMENT_VERSION": "shopsimulator-environment-v2.1",
            "SHOPPING_ENV_MANIFEST": str(DEFAULT_MANIFEST),
            "GRPO_MODEL_PATH": str(model),
            "GRPO_TRAIN_FILE": str(train_data),
            "GRPO_VAL_FILE": str(val_data),
            "GRPO_OUTPUT_DIR": str(output),
            "SHOPSIM_BASE_URL": str(args.env_url),
            "SHOPPING_AGENT_LOOP_CONFIG": str(DEFAULT_AGENT_CONFIG),
            "SHOPPING_TOOL_CONFIG": str(DEFAULT_TOOL_CONFIG),
            "GRPO_CONFIG_NAME": config.stem,
            "GRPO_CONFIG_PATH": str(config),
            "SHOPPING_TRAINING_SEED": str(args.seed),
            "SHOPPING_OPTIMIZATION_REWARD_PROFILE": (
                "environment_v3"
                if args.optimization_reward == "v3"
                else "optimization_v4"
            ),
            # Ray 2.56 otherwise mirrors the outer `uv run` command into a
            # fresh worker environment that does not contain the pinned GRPO
            # runtime. Workers must inherit this already-preflighted venv.
            "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
            "PYTHONHASHSEED": str(args.seed),
        }
    )
    if args.logger == "swanlab":
        environment.update(
            {
                "SWANLAB_MODE": "online",
                "SWANLAB_LOG_DIR": str(output / "swanlab"),
            }
        )
    overrides = _training_overrides(args)
    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        f"--config-path={config.parent}",
        f"--config-name={config.stem}",
        *overrides,
    ]
    return command, environment


def main() -> None:
    args = parse_args()
    command, environment = build_command(args)
    audit = {
        "command": command,
        "model": environment["GRPO_MODEL_PATH"],
        "train_data": environment["GRPO_TRAIN_FILE"],
        "val_data": environment["GRPO_VAL_FILE"],
        "env_url": environment["SHOPSIM_BASE_URL"],
        "output": environment["GRPO_OUTPUT_DIR"],
        "logger": args.logger,
        "config": str(args.config.resolve()),
        "seed": args.seed,
        "optimization_reward": args.optimization_reward,
    }
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    preflight = [
        sys.executable,
        str(ROOT / "scripts/check_grpo_runtime.py"),
        *_training_overrides(args),
    ]
    preflight_status = subprocess.call(preflight, cwd=ROOT, env=environment)
    if preflight_status:
        raise SystemExit(preflight_status)
    if args.preflight_only:
        return
    output = Path(environment["GRPO_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / EXPERIMENT_MANIFEST
    if not manifest_path.exists():
        manifest_path.write_text(
            json.dumps(
                _manifest_payload(args, environment, command),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    status = subprocess.call(command, cwd=ROOT, env=environment)
    if status == 0 and args.target_global_step is not None:
        expected = output / f"global_step_{args.target_global_step}"
        if not expected.is_dir():
            raise SystemExit(f"training exited successfully without target checkpoint {expected}")
    raise SystemExit(status)


if __name__ == "__main__":
    main()
