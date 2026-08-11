# GRPO with veRL

## Purpose

SFT teaches the action format and a strong initial policy. GRPO then samples
fresh trajectories in ShopSimulator and optimizes the terminal Reward v3 signal.
The goal is to improve constraint satisfaction and termination behavior without
requiring a learned reward model.

## Integration boundary

veRL is installed from the pinned `verl==0.8.0` package. This repository does
not vendor the veRL source tree. Project-owned integration code lives in:

```text
src/shopping_grpo/training/grpo/
  adapter/              AgentLoop and ShopSimulator tools
  compat.py             narrow runtime compatibility hook
  dynamic_sampling.py   bounded non-zero-reward sampling
```

`scripts/setup.sh` applies one SHA-256-checked patch needed to connect the
bounded dynamic sampler to veRL 0.8.0. Setup fails rather than patching an
unknown veRL version.

## Inputs

- Initial policy: `outputs/models/sft-merged`
- Train set: `data/grpo/train.parquet` (1,000 tasks)
- Validation set: `data/grpo/validation.parquet` (50 tasks)
- Environment: ShopSimulator Environment v2.1
- Reward: Reward v3

Hashes are recorded in [`data/grpo/metadata.json`](../data/grpo/metadata.json).

## Run

Inspect the resolved command first:

```bash
bash scripts/grpo.sh --dry-run
```

Train:

```bash
bash scripts/grpo.sh
```

Important defaults:

| Setting | Value |
|---|---|
| Algorithm | GRPO |
| Rollouts per prompt | 4 |
| Rollout temperature / top-p | 0.7 / 0.9 |
| Train / validation batch | 2 / 2 |
| Policy learning rate | `1e-6` |
| LoRA rank / alpha | 16 / 32 |
| Maximum model length | 24,576 |
| Maximum training steps | 500 |
| Save / validation frequency | 50 / 50 |
| KL reward / KL loss | disabled / disabled |

Dynamic sampling can generate at most three batches to find a useful update and
permits at most ten consecutive skipped updates. These bounds prevent an
all-equal reward batch from causing an unbounded resampling loop.

## Optional reward-variance screen

For a budget-limited pilot, the training pool can be screened before GRPO. The
screen is balanced across the frozen short/medium/long metadata buckets and
prioritizes difficult, option-bearing Queries without reading gold-product
fields:

```bash
python scripts/prepare_grpo_active_set.py \
  --output outputs/active-screen/tasks.jsonl \
  --manifest outputs/active-screen/screen-manifest.json
```

Evaluate that JSONL with the normal benchmark runner and four attempts per task,
then materialize only complete, `reward_valid=true` groups whose Reward v3
outcomes vary:

```bash
python scripts/prepare_grpo_active_set.py \
  --screening outputs/active-screen/raw.jsonl \
  --attempts-per-task 4 \
  --output outputs/active-screen/train.parquet \
  --manifest outputs/active-screen/active-manifest.json
```

This is a sampling-efficiency mechanism, not a performance claim. Any resulting
checkpoint must still beat its SFT initialization under the paired validation
protocol before further training or held-out evaluation.

## What the GRPO experiments showed

| Experiment | Matched comparison | Strict-success difference | Interpretation |
|---|---|---:|---|
| Initial pipeline, deterministic Final-200×1 | GRPO step 100 vs pipeline SFT | +1.5pp | Descriptive; no sampling interval |
| Validation-50×3 selection | Terminal-GRPO-30 vs SFT | +8.0pp, CI [+2.0,+14.7] | Positive development result |
| Frozen-stage Final-200×1 | Terminal-GRPO-30 vs SFT | +1.5pp, CI [-2.0,+5.0] | Development effect did not reproduce |
| Active-set Validation-50×3 | Active-GRPO-10 vs More-SFT | +1.3pp, CI [-5.3,+8.0] | Failed the promotion gate |

The uniform seed-42 Reward-v3 attempt exposed the main efficiency problem. It
reached step 94 before interruption, with about 37.9% effective groups, 51.4%
all-equal groups, and 58 skipped updates; no final checkpoint from that run was
used as a performance result. Reward-v3 and Reward-v4 five-update runs and a
step-1→2 dynamic-state resume validated the integration only.

The active-set pilot selected 20 reward-varying tasks from a 48-task×4 screen.
It observed a 71.7% effective-group ratio and applied all ten updates, but
`pass@3` and `pass^3` stayed unchanged and the emphasized 4+ constraint stratum
regressed. Because the earlier uniform run used another initialization and run
length, the ratio difference is not a controlled causal estimate. The pilot did
not test turn-level credit assignment or demonstrate a policy gain.

The complete evidence ledger, including the stopped branches and protocol
limitations, is in [`experiments/comparison.md`](../experiments/comparison.md).

The canonical configuration is [`configs/grpo.yaml`](../configs/grpo.yaml).
Use explicit launcher arguments for the cumulative target and periodic
checkpoints:

```bash
bash scripts/grpo.sh --target-global-step 100 --checkpoint-every 10
```

Other advanced Hydra overrides may be appended after `--`. The launcher rejects
duplicate Hydra overrides for the training target and save frequency when the
explicit arguments are active.

## Export

veRL checkpoints are not directly served by the evaluation launcher. Export the
selected actor:

```bash
bash scripts/export_grpo.sh \
  outputs/models/grpo/global_step_100/actor \
  outputs/models/grpo-merged
```

The initial pipeline comparison uses step 100, while the later statistically
selected Terminal-GRPO checkpoint uses step 30. Select checkpoints using the
registered validation protocol rather than assuming that the final training
step is best.

The exporter first reconstructs veRL's FSDP state and then applies the emitted
LoRA adapter with `merge_and_unload`. The final directory is standalone and must
not contain a nested `lora_adapter/`; serving the intermediate veRL export would
silently evaluate the unchanged base weights.
