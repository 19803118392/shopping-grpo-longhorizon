# Experiment comparison

The project contains three consecutive experiment stages. Results are shown
together, while checkpoint and protocol boundaries remain explicit.

| Stage | Model and protocol | Strict success | vs stage-matched SFT | Statistical conclusion |
|---|---|---:|---:|---|
| 1. Pipeline benchmark, Final-200×1 | Qwen3.5-2B Base | 0/200 = 0.0% | — | One deterministic rollout |
| 1. Pipeline benchmark, Final-200×1 | LoRA SFT | 121/200 = 60.5% | — | One deterministic rollout |
| 1. Pipeline benchmark, Final-200×1 | GRPO step 100 | 124/200 = 62.0% | +1.5 points | One pass; sampling variance not estimated |
| 2. Method selection, Validation-50×3 | SFT | 100/150 = 66.7% | — | Wilson 95% CI [58.8%, 73.7%] |
| 2. Method selection, Validation-50×3 | Terminal-GRPO 30 | 112/150 = **74.7%** | **+8.0 points** | paired CI [+2.0,+14.7], `p=0.0118` |
| 3. Frozen confirmation, Final-200×1 | SFT | 114/200 = 57.0% | — | Wilson 95% CI [50.1%, 63.7%] |
| 3. Frozen confirmation, Final-200×1 | Terminal-GRPO 30 | 117/200 = **58.5%** | +1.5 points | paired CI [-2.0,+5.0], `p=0.6072` |

## Stage 1: establish the complete pipeline

The initial end-to-end benchmark evaluated Base, SFT, and GRPO step 100 with
one deterministic rollout on each of the same 200 held-out tasks.

| Model | Done | Strict success | Purchase success | Mean reward | Mean steps | Guard rejections |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B baseline | 18.0% | 0.0% | 0.0% | -0.1105 | 5.875 | 752 |
| LoRA SFT | 96.5% | 60.5% | 60.5% | 0.4729 | 12.335 | 52 |
| GRPO step 100 | 96.5% | 62.0% | 62.5% | 0.5158 | 11.850 | 38 |

SFT provides the dominant capability gain: it teaches the base model to follow
the action protocol, continue through long-horizon shopping tasks, and reach
valid terminal states. GRPO adds three strict successes, one valid alternative
purchase, and fewer Guard rejections, but a single rollout per task cannot
establish statistical reliability.

Machine-readable settings and summaries are stored in [`baseline/`](baseline/),
[`sft/`](sft/), and [`grpo/`](grpo/).

## Stage 2: repeated method selection

SFT and Terminal-GRPO after 30 updates were compared on 50 development tasks,
with three paired attempts per task.

| Model | Strict success | Wilson 95% CI | `pass@3` | `pass^3` | Loop rate | Mean steps |
|---|---:|---:|---:|---:|---:|---:|
| SFT | 66.7% | [58.8%, 73.7%] | 78.0% | 54.0% | 10.0% | 10.95 |
| Terminal-GRPO (30 updates) | **74.7%** | **[67.2%, 81.0%]** | **84.0%** | **62.0%** | **8.7%** | **10.80** |

The paired delta was **+8.0 percentage points**, with task win/tie/loss
`9/39/2`, task-paired bootstrap 95% CI **[+2.0, +14.7] points**, and exact
McNemar `p=0.0118`. This result selected the candidate for frozen confirmation;
it was not treated as the final algorithmic conclusion. See the
[Validation-50×3 experiment card](validation-50x3/README.md).

## Stage 3: frozen held-out confirmation

The SFT and Terminal-GRPO (30 updates) checkpoints, code, configuration, and
data hashes were frozen before one deterministic pass over all 200 tasks.

| Model | Strict success | Wilson 95% CI | Loop rate | Guard rejection | Mean steps |
|---|---:|---:|---:|---:|---:|---:|
| SFT | 57.0% | [50.1%, 63.7%] | 11.5% | 28.0% | 11.34 |
| Terminal-GRPO (30 updates) | 58.5% | [51.6%, 65.1%] | 11.0% | 21.5% | 11.58 |

The paired delta is **+1.5 percentage points**, with task win/tie/loss
`9/185/6`, task-paired bootstrap 95% CI **[-2.0, +5.0] points**, and exact
McNemar `p=0.6072`. The validation effect size did not reproduce, so this stage
does not establish a reliable final GRPO gain. See the
[Final-200 experiment card](final-200/README.md).

## Overall interpretation and limits

The full project supports the training order `Base → SFT → GRPO`: SFT creates
the usable tool policy, while GRPO shows smaller improvements in strict success
and Guard behavior. It does **not** yet support a claim that the GRPO improvement
is statistically reliable on the held-out set.

- The pipeline benchmark and frozen confirmation use different SFT/GRPO
  checkpoints and code snapshots; their absolute rates are not directly
  comparable training-effect estimates.
- Stage 1 has one rollout per task and does not estimate sampling variance.
- Stage 2 is a development-set experiment; its strata are exploratory.
- In Stage 1, seven SFT and GRPO tasks ended without a Reward v3 terminal record
  and remain in the denominator.
- The SFT and GRPO recipes target large single GPUs; results may differ with
  other distributed layouts or dependency versions.
- Full model weights and rollout logs are generated artifacts and are not
  committed to the repository.
