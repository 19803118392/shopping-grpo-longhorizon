# LoRA SFT

## Purpose

The base model can speak naturally but does not reliably follow
ShopSimulator's action protocol. Supervised fine-tuning teaches the basic
policy: issue legal tool calls, use observations as evidence, select product
variants and terminate.

## Inputs

- Base model: `Qwen/Qwen3.5-2B`
- Train data: `data/sft/train.jsonl` (379 rows)
- Validation data: `data/sft/validation.jsonl` (49 rows)
- Target: assistant tokens only; user and tool-observation tokens are masked

The data provenance and hashes are recorded in
[`data/sft/metadata.json`](../data/sft/metadata.json).

## Run

After `bash scripts/setup.sh`:

```bash
bash scripts/sft.sh
```

The launcher trains a LoRA adapter and then merges it with the base model:

```text
outputs/models/sft-lora/
outputs/models/sft-merged/
```

Default recipe:

| Setting | Value |
|---|---|
| Maximum sequence length | 24,576 |
| Epochs | 3 |
| Per-device batch size | 1 |
| Gradient accumulation | 8 |
| Learning rate | `1e-4` |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Gradient checkpointing | enabled |
| Attention implementation | SDPA |

The long context is intentional: a training example includes the complete
multi-turn interaction. Shortening it may truncate the terminal decision or the
evidence that supports it.

## Evaluate

```bash
bash scripts/serve_model.sh outputs/models/sft-merged
bash scripts/evaluate.sh sft
```

The reported checkpoint completed 141 optimizer steps. Its validation loss was
0.3365 after epoch 1, 0.3189 after epoch 2 and 0.3147 after epoch 3. The frozen
result and reproduction config are in [`experiments/sft/`](../experiments/sft/).

## Additional SFT controls

The seed-42 experiment trained nested 95/190/379-row subsets for 144 optimizer
steps and continued the 379-row run to 288 steps. After 24,576-token overlength
filtering, the usable row counts were 95/189/376.

| Checkpoint | Validation-50×3 strict success | `pass@3` | `pass^3` | Loop rate | Mean steps |
|---|---:|---:|---:|---:|---:|
| `n95@144` | 62.0% | 74.0% | 46.0% | 10.7% | 12.18 |
| `n190@144` | 66.0% | 82.0% | 48.0% | 8.0% | 11.06 |
| `n379@144` | 66.7% | 80.0% | 56.0% | 18.0% | 10.87 |
| More-SFT `n379@288` | **70.0%** | **84.0%** | **58.0%** | **4.7%** | **9.91** |

The data-size and compute trends are positive, but every paired bootstrap
interval includes zero and only one training seed was run. More-SFT is therefore
a competitive control rather than a proven scaling law. In a later post-hoc
Final-200×3 comparison, More-SFT scored 65.7% versus Terminal-GRPO-30 at 63.2%
(+2.5 points, paired CI [-1.2,+6.2], `p=0.1756`) while using fewer steps and
producing fewer loops. This reused evaluation set cannot support a new final
claim, but it shows why extra supervised compute must be matched in RL studies.
See the [complete mechanism card](../experiments/single-seed-42/README.md).

## Output contract

GRPO starts from the merged model, not directly from the adapter:

```text
GRPO_MODEL_PATH=outputs/models/sft-merged
```

This boundary keeps the GRPO launcher independent of the SFT trainer process.
