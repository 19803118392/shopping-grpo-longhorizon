# Shopping GRPO

An end-to-end Shopping Agent post-training tutorial built on Qwen3.5-2B,
ShopSimulator and veRL:

```text
Qwen3.5-2B baseline → LoRA SFT → GRPO → held-out evaluation
```

The repository contains the ShopSimulator source snapshot, frozen SFT/GRPO/eval
datasets, the veRL integration and every launcher needed for the workflow.
There is one supported environment and one supported training recipe:
ShopSimulator Environment v2.1 with Reward v3.

## Results

One deterministic rollout per task on the same 200 held-out tasks:

| Model | Strict success | Mean reward |
|---|---:|---:|
| Qwen3.5-2B baseline | 0.0% | -0.1105 |
| LoRA SFT | 60.5% | 0.4729 |
| GRPO step 100 | 62.0% | 0.5158 |

See [`experiments/README.md`](experiments/README.md) for interpretation.

## Requirements

- Linux with an NVIDIA GPU and a compatible CUDA driver;
- [`uv`](https://docs.astral.sh/uv/);
- about 25 GB of disk space for dependencies, model weights and generated
  artifacts;
- SFT was designed for a 48 GB GPU;
- the provided GRPO configuration was validated on one 96 GB GPU.

Python 3.12 is used for training and Python 3.10 for the isolated
ShopSimulator service. `uv` resolves both environments.

## Repository layout

```text
configs/                 GRPO, AgentLoop and tool configuration
data/
  sft/                   ready-to-train SFT JSONL
  grpo/                  ready-to-train veRL JSONL and Parquet
  evaluation/            200 held-out tasks
environments/
  ShopSimulator/         embedded environment and product archive
experiments/             final reported experiment
scripts/                 the tutorial entrypoints
src/shopping_grpo/       environment, training and veRL integration code
tests/                   unit and packaging checks
```

Generated checkpoints, rollouts and logs are written only to `outputs/` and
are ignored by Git.

## 1. Install

From the repository root:

```bash
bash scripts/setup.sh
```

This command creates the main `.venv`, installs the pinned veRL/vLLM/SFT
dependencies, prepares the separate ShopSimulator environment, expands the
product archive and builds the search index.

## 2. Start ShopSimulator

Keep this terminal running:

```bash
bash scripts/start_environment.sh
```

The structured service listens on `http://127.0.0.1:5700`.

## 3. Baseline

In a second terminal, start the base model server:

```bash
bash scripts/serve_model.sh Qwen/Qwen3.5-2B
```

In a third terminal, collect and summarize the baseline:

```bash
bash scripts/baseline.sh
```

Outputs:

```text
outputs/evaluation/baseline/trajectories.jsonl
outputs/evaluation/baseline/summary.json
```

Stop the model server before training so it releases the GPU.

## 4. LoRA SFT

The checked-in dataset contains 379 training and 49 validation trajectories:

```bash
bash scripts/sft.sh
```

This trains the LoRA adapter and merges it into a standalone model:

```text
outputs/models/sft-lora/
outputs/models/sft-merged/
```

Evaluate the merged SFT model:

```bash
bash scripts/serve_model.sh outputs/models/sft-merged
bash scripts/evaluate.sh sft
```

Stop the model server again before GRPO.

## 5. GRPO

Inspect the resolved command without starting CUDA/Ray:

```bash
bash scripts/grpo.sh --dry-run
```

Start training:

```bash
bash scripts/grpo.sh
```

The default recipe uses the merged SFT model, the checked-in veRL Parquet
files, constraint-aware Reward v3 and bounded dynamic sampling. Checkpoints are
written under `outputs/models/grpo/`.

To enable SwanLab:

```bash
export SWANLAB_API_KEY=...
bash scripts/grpo.sh --logger swanlab
```

## 6. Export and evaluate GRPO

Choose a checkpoint using the GRPO validation metrics, then export its actor to
Hugging Face format:

```bash
bash scripts/export_grpo.sh \
  outputs/models/grpo/global_step_100/actor \
  outputs/models/grpo-merged
```

Serve and evaluate it:

```bash
bash scripts/serve_model.sh outputs/models/grpo-merged
bash scripts/evaluate.sh grpo
```

Compare:

```text
outputs/evaluation/baseline/summary.json
outputs/evaluation/sft/summary.json
outputs/evaluation/grpo/summary.json
```

The primary metric is `strict_success_rate`, which requires a valid Reward v3
gold purchase and a complete environment terminal state. Missing or failed
tasks remain in the denominator.

## Configuration

Most users only need environment variables:

| Variable | Default |
|---|---|
| `BASE_MODEL` | `Qwen/Qwen3.5-2B` |
| `SHOPSIM_BASE_URL` | `http://127.0.0.1:5700` |
| `LLM_BASE_URL` | `http://127.0.0.1:8000/v1` |
| `SERVED_MODEL_NAME` | `shopping-agent` |
| `SFT_ADAPTER_DIR` | `outputs/models/sft-lora` |
| `SFT_MERGED_DIR` | `outputs/models/sft-merged` |

Advanced GRPO overrides can be appended after the launcher arguments:

```bash
bash scripts/grpo.sh -- \
  trainer.total_training_steps=20 \
  trainer.save_freq=10
```

## Components

- ShopSimulator is embedded under [`environments/ShopSimulator/`](environments/ShopSimulator/).
- veRL 0.8.0 is pinned in `pyproject.toml`; the repository contains the
  Shopping AgentLoop, tool adapter and the small pinned dynamic-sampling patch.
- All training and evaluation datasets are versioned under [`data/`](data/).

## Acknowledgements

This project builds on
[ShopSimulator](https://github.com/ShopAgent-Team/ShopSimulator),
[veRL](https://github.com/verl-project/verl) and
[Qwen](https://github.com/QwenLM/Qwen3).
