# Experiment comparison

## Current branch: frozen Final-200

The current SFT and Terminal-GRPO (30 updates) checkpoints were frozen before a
single deterministic pass over all 200 final tasks.

| Model | Strict success | Wilson 95% CI | Loop rate | Guard rejection | Mean steps |
|---|---:|---:|---:|---:|---:|
| SFT | 57.0% | [50.1%, 63.7%] | 11.5% | 28.0% | 11.34 |
| Terminal-GRPO (30 updates) | 58.5% | [51.6%, 65.1%] | 11.0% | 21.5% | 11.58 |

The paired delta is **+1.5 percentage points**, with task win/tie/loss
`9/185/6`, task-paired bootstrap 95% CI **[-2.0, +5.0] points**, and exact
McNemar `p=0.6072`. This is not statistically significant and does not establish
a reliable final GRPO gain. See the [Final-200 experiment card](final-200/README.md).

## Current branch: paired Validation-50×3

The current branch compares SFT and Terminal-GRPO after 30 updates on 50
development tasks with three paired attempts per task.

| Model | Strict success | Wilson 95% CI | `pass@3` | `pass^3` | Loop rate | Mean steps |
|---|---:|---:|---:|---:|---:|---:|
| SFT | 66.7% | [58.8%, 73.7%] | 78.0% | 54.0% | 10.0% | 10.95 |
| Terminal-GRPO (30 updates) | **74.7%** | **[67.2%, 81.0%]** | **84.0%** | **62.0%** | **8.7%** | **10.80** |

The paired delta was **+8.0 percentage points**, with task win/tie/loss `9/39/2`,
task-paired bootstrap 95% CI **[+2.0, +14.7] points**, and exact McNemar
`p=0.0118`. Final-200 did not reproduce that effect size, so this remains a
development-set ablation rather than the final algorithmic conclusion. See the
[experiment card](validation-50x3/README.md) for the full protocol and hashes.

## Imported upstream: single-pass Final-200

This is an imported upstream report, not a rerun of the current branch. The
record states that all three models were evaluated with one deterministic
rollout on the same 200 held-out tasks.

| Model | Done | Strict success | Purchase success | Mean reward | Mean steps | Guard rejections |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B baseline | 18.0% | 0.0% | 0.0% | -0.1105 | 5.875 | 752 |
| LoRA SFT | 96.5% | 60.5% | 60.5% | 0.4729 | 12.335 | 52 |
| GRPO step 100 | 96.5% | 62.0% | 62.5% | 0.5158 | 11.850 | 38 |

## Interpretation

SFT provides the dominant gain. It teaches the base model to use the action
protocol, continue through multi-step shopping tasks and reach valid terminal
states. GRPO then adds a smaller improvement: three additional strict successes,
one valid alternative purchase, fewer wrong purchases, fewer loops and fewer
guard rejections than SFT.

The result supports a practical training order: first establish reliable tool
behavior with SFT, then use online RL for constraint satisfaction and policy
refinement.

## Limits

- Each model has one rollout per task, so the table does not estimate sampling
  variance.
- The SFT and GRPO recipes target large single GPUs; results may differ with
  other distributed layouts or dependency versions.
- Seven SFT and GRPO tasks ended without a Reward v3 terminal record and remain
  in the denominator.
- Full model weights and rollout logs are generated artifacts, not committed
  repository files.

Machine-readable settings and summaries are stored beside each stage:

- [`baseline/`](baseline/)
- [`sft/`](sft/)
- [`grpo/`](grpo/)
