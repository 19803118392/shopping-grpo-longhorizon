# 96 GB GPU runbook

Use this checklist after provisioning a single 96 GB NVIDIA GPU. Keep training and
evaluation sequential: both workflows lease ShopSimulator slots and must not share
the service concurrently.

## Scope and first-session budget

The first rental should target environment setup, smoke tests, and an Evidence
Memory control/ablation—not the 500-step final run. Reserve at least 16 CPU cores,
64 GB RAM, and 100 GB persistent disk for environments, model caches, checkpoints,
rollouts, and logs. Set a provider-side spending alert and an initial 10-hour stop
limit. Record the actual hourly price from the chosen provider before starting.

Expected measured workloads from the existing report are approximately 3 hours for
SFT, 3–4 hours for the 100-step GRPO checkpoint, and 14 hours for 500 GRPO steps.
Repeated evaluation scales approximately with attempts per task and should be
budgeted separately.

## 1. Capture the machine contract

Save these outputs under a persistent run directory before installing anything:

```bash
mkdir -p outputs/run-audit
git rev-parse HEAD | tee outputs/run-audit/git-head.txt
nvidia-smi | tee outputs/run-audit/nvidia-smi.txt
nvidia-smi --query-gpu=name,memory.total,driver_version \
  --format=csv,noheader | tee outputs/run-audit/gpu.csv
python --version | tee outputs/run-audit/python.txt
df -h | tee outputs/run-audit/disk.txt
```

Do not put API keys in the repository or captured command logs.

## 2. Install and validate in order

```bash
bash scripts/setup.sh
bash scripts/start_environment.sh
python scripts/smoke_shop_env.py
python -m pytest -q
python -m compileall -q src scripts
shopping-grpo smoke --json
```

Before GRPO, ensure the merged SFT model exists and run the version/hash preflight
through a dry run:

```bash
python scripts/train_grpo.py \
  --model outputs/models/sft-merged \
  --output outputs/models/grpo-preflight \
  --dry-run
```

The dry run must point at an existing model with `config.json` and weights. It prints
the exact veRL command and environment paths without creating a training output.

## 3. Run the cheapest useful A/B first

Serve one frozen SFT checkpoint and collect the same tasks, sampling parameters,
attempt count, and seed policy twice:

1. control: Evidence Memory disabled;
2. candidate: `--evidence-memory` enabled.

Start with 20 benchmark tasks × 2 attempts. If action guards, Reward v3 validity,
and output coverage are clean, expand to 200 tasks × 5 attempts. Compare with
`scripts/compare_repeated_evaluations.py`. Do not interpret a positive point estimate
as a win unless the interval, task transitions, and failure taxonomy are coherent.

## 4. Modified GRPO smoke run

Enable Evidence Memory only for the modified run:

```bash
export SHOPPING_EVIDENCE_MEMORY_ENABLE=true
bash scripts/grpo.sh -- \
  trainer.total_training_steps=5 \
  trainer.save_freq=5 \
  trainer.test_freq=5 \
  trainer.experiment_name=shopping-evidence-memory-smoke
```

Inspect five updates before scheduling 100 steps. The control run must use the same
checkpoint, seed/config, task data, rollout count, and hardware, with
`SHOPPING_EVIDENCE_MEMORY_ENABLE=false`.

## 5. Stop conditions

Stop the run, preserve logs, and diagnose before spending more when any of these
conditions occurs:

- environment, observation, tool, or Reward version/hash preflight fails;
- `reward_valid=false`, infrastructure-invalid samples, or footer failures appear;
- repeated dynamic-sampling batches contain no usable reward variation;
- CUDA OOM occurs or peak allocation leaves too little serving headroom;
- disk free space falls below 20 GB;
- loss/reward becomes non-finite;
- the provider time or spending cap is reached.

After a failure, retain the command, resolved config, `nvidia-smi`, stdout/stderr,
latest valid checkpoint, and ShopSimulator logs before deleting the instance.

## 6. Artifacts to bring back

Download the resolved config, commands, environment versions, model/checkpoint hash,
raw JSONL rollouts, summaries, paired statistics report, training curves, failure
taxonomy, runtime, peak memory, and rental duration. These are the evidence needed
for both reproducibility and internship interviews.

