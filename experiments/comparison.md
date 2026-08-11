# Complete experiment ledger and analysis

This document is the canonical account of every reportable experiment in the
project. It combines results in one timeline while preserving checkpoint,
training-seed, dataset, and inference-protocol boundaries. A numerical
difference is described as a training effect only when the two rows form a
matched comparison under the same protocol.

## How to read the evidence

- **Strict success** requires a complete `gold_purchase` terminal result with
  `reward_valid=true`; missing, invalid, and infrastructure-failed attempts stay
  in the fixed denominator.
- **`pass@3`** is the fraction of tasks with at least one strict success in
  three attempts. **`pass^3`** is the fraction that succeed in all three.
- Wilson intervals describe uncertainty for one model's attempt-level rate.
  The paired bootstrap resamples tasks and is the primary interval for a model
  difference. McNemar uses paired attempt outcomes.
- Validation-50 is repeatedly used for method selection. Its results are
  development evidence, not held-out confirmation.
- Final-200 has zero overlap with training data, but the same 200 tasks appeared
  in the initial pipeline benchmark before the later frozen-confirmation run.
  It is therefore data-held-out but not an untouched, single-use test set over
  the entire project history.
- The post-hoc Final-200×3 run was performed after earlier Final-200 results were
  known. It estimates rollout variance and compares deployed checkpoints, but
  cannot upgrade a development decision into a new final claim.

## Unified result ledger

| Experiment | Model | Protocol | Strict success | Difference | Evidence status |
|---|---|---|---:|---:|---|
| Pipeline benchmark | Qwen3.5-2B Base | Final-200×1, deterministic | 0/200 = 0.0% | — | Descriptive single pass |
| Pipeline benchmark | LoRA SFT step 141 | Final-200×1, deterministic | 121/200 = 60.5% | — | Descriptive single pass |
| Pipeline benchmark | GRPO step 100 | Final-200×1, deterministic | 124/200 = 62.0% | +1.5pp vs pipeline SFT | No sampling interval |
| Method selection | SFT | Validation-50×3 | 100/150 = 66.7% | — | Development baseline |
| Method selection | Terminal-GRPO-30 | Validation-50×3 | 112/150 = 74.7% | +8.0pp | Paired CI [+2.0,+14.7], `p=0.0118` |
| Frozen confirmation | SFT | Final-200×1, deterministic | 114/200 = 57.0% | — | Frozen-stage baseline |
| Frozen confirmation | Terminal-GRPO-30 | Final-200×1, deterministic | 117/200 = 58.5% | +1.5pp | Paired CI [-2.0,+5.0], `p=0.6072` |
| SFT scaling | `n95@144` | Validation-50×3, seed root 42 | 93/150 = 62.0% | — | Single-training-seed development result |
| SFT scaling | `n190@144` | Validation-50×3, seed root 42 | 99/150 = 66.0% | +4.0pp vs `n95@144` | Paired CI [-3.3,+11.3] |
| SFT scaling | `n379@144` | Validation-50×3, seed root 42 | 100/150 = 66.7% | +4.7pp vs `n95@144` | Paired CI [-4.7,+14.0] |
| More-SFT | `n379@288` | Validation-50×3, seed root 42 | 105/150 = 70.0% | +3.3pp vs `n379@144` | Paired CI [-2.7,+10.0] |
| Post-hoc checkpoint comparison | Terminal-GRPO-30 | Final-200×3, stochastic | 379/600 = 63.2% | — | Reused benchmark |
| Post-hoc checkpoint comparison | More-SFT `n379@288` | Final-200×3, stochastic | 394/600 = 65.7% | +2.5pp | Paired CI [-1.2,+6.2], `p=0.1756` |
| Active-set pilot | More-SFT `n379@288` | Validation-50×3, seed root 42 | 105/150 = 70.0% | — | Matched pilot baseline |
| Active-set pilot | Active-GRPO-10 | Validation-50×3, seed root 42 | 107/150 = 71.3% | +1.3pp | Paired CI [-5.3,+8.0], `p=0.8238` |

These rows do not form one leaderboard. They represent several model families,
training recipes, inference seed roots, and stages of repeated benchmark use.

## Experiment 1: establish the Base → SFT → GRPO pipeline

The first experiment verified the complete workflow on one deterministic
rollout for each of 200 evaluation tasks.

| Model | Done | Strict success | Purchase success | Mean Reward v3 | Mean steps | Guard rejections |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B Base | 18.0% | 0.0% | 0.0% | -0.1105 | 5.88 | 752 |
| LoRA SFT | 96.5% | 60.5% | 60.5% | 0.4729 | 12.34 | 52 |
| GRPO step 100 | 96.5% | 62.0% | 62.5% | 0.5158 | 11.85 | 38 |

The base model mostly failed the action protocol: only 36/200 tasks reached a
valid environment terminal and 752 actions were rejected by the Guard. SFT is
the dominant intervention, raising strict success by 60.5 points and teaching
the model to continue through a multi-turn tool trajectory. GRPO added three
strict successes and reduced Guard rejections by 14, but one attempt per task
cannot distinguish a stable policy gain from decoding variance.

The exact settings and machine-readable summaries are in
[`baseline/`](baseline/), [`sft/`](sft/), and [`grpo/`](grpo/).

## Experiment 2: repeated Validation-50 method selection

The statistical upgrade evaluated SFT and Terminal-GRPO after 30 updates on all
50 development tasks with three paired attempts per task. Seeds were derived
from `(2026, task_id, attempt_index)` and replayed exactly for 10/10 checked
pairs. Temperature/top-p were `0.7/0.9`, the step cap was 35, and the fixed
denominator was 150 attempts per model.

| Model | Strict success | Wilson 95% CI | `pass@3` | `pass^3` | Loop rate | Mean steps |
|---|---:|---:|---:|---:|---:|---:|
| SFT | 100/150 = 66.7% | [58.8%, 73.7%] | 78.0% | 54.0% | 10.0% | 10.95 |
| Terminal-GRPO-30 | 112/150 = **74.7%** | [67.2%, 81.0%] | 84.0% | 62.0% | 8.7% | 10.80 |

The paired difference was +8.0pp, task win/tie/loss was `9/39/2`, the
task-bootstrap interval was [+2.0,+14.7]pp, and exact McNemar `p=0.0118`.
This was enough to promote the checkpoint to a confirmation run. It was not
enough to declare a final gain: the same development tasks selected the method,
and Guard rejection rate actually increased from 24.7% to 28.7%.

See the [Validation-50×3 card](validation-50x3/README.md).

## Experiment 3: frozen-stage Final-200 confirmation

The commit, model/checkpoint trees, configuration, validation report, and data
hashes were frozen before one deterministic attempt on each of 200 tasks.

| Model | Strict success | Wilson 95% CI | Loop rate | Guard rejection | Mean steps |
|---|---:|---:|---:|---:|---:|
| SFT | 114/200 = 57.0% | [50.1%, 63.7%] | 11.5% | 28.0% | 11.34 |
| Terminal-GRPO-30 | 117/200 = 58.5% | [51.6%, 65.1%] | 11.0% | 21.5% | 11.58 |

The difference shrank to +1.5pp. Task win/tie/loss was `9/185/6`, the paired
interval was [-2.0,+5.0]pp, and McNemar `p=0.6072`. Both models covered all 200
tasks with zero infrastructure-invalid attempts and zero critical footer
failures. Terminal-GRPO used slightly more steps and had six context-budget
errors versus three for SFT, while reducing Guard rejection by 6.5pp.

The +8.0pp development effect did not reproduce. The frozen-stage evidence
therefore supports only a small positive point estimate, not a reliable GRPO
improvement. Because these 200 tasks had also appeared in Experiment 1, this is
a rigorously frozen rerun on training-held-out data, not a first-ever look at a
pristine test set. See the [Final-200 card](final-200/README.md).

## Experiment 4: SFT data-size and compute controls

The seed-42 follow-up trained nested SFT subsets and evaluated all checkpoints
under one Validation-50×3 protocol. The nominal 95/190/379-row subsets produced
95/189/376 usable training examples after overlength filtering. All runs used
LoRA rank/alpha 16/32, learning rate `1e-4`, a 24,576-token window, and Assistant-
only loss. The `n379@288` run continued the same optimizer/scheduler trajectory
through the step-144 checkpoint.

| Configuration | Strict success | Wilson 95% CI | `pass@3` | `pass^3` | Loop rate | Mean steps |
|---|---:|---:|---:|---:|---:|---:|
| `n95@144` | 93/150 = 62.0% | [54.0%, 69.4%] | 74.0% | 46.0% | 10.7% | 12.18 |
| `n190@144` | 99/150 = 66.0% | [58.1%, 73.1%] | 82.0% | 48.0% | 8.0% | 11.06 |
| `n379@144` | 100/150 = 66.7% | [58.8%, 73.7%] | 80.0% | 56.0% | 18.0% | 10.87 |
| `n379@288` | 105/150 = **70.0%** | [62.2%, 76.8%] | **84.0%** | **58.0%** | **4.7%** | **9.91** |

The data-size trend is positive but saturating: `95→190` is +4.0pp, while
`190→379` is only +0.7pp. Doubling compute at 379 nominal rows adds +3.3pp and
reduces loops sharply, but all four paired intervals include zero. One training
seed and repeated use of the same 50 tasks prevent a general scaling claim.
The important experimental lesson is that More-SFT is a necessary, inexpensive
control whenever the claimed GRPO gain is only a few points.

## Experiment 5: post-hoc Final-200×3 checkpoint comparison

After earlier Final-200 results were already known, Terminal-GRPO-30 and the new
More-SFT checkpoint were each sampled three times on all 200 tasks with seed
root 42 and temperature/top-p `0.7/0.9`. This run estimates stochastic inference
behavior; `final_200=false` and `holdout_status=posthoc_reused` are recorded in
the raw summaries.

| Model | Strict success | Wilson 95% CI | `pass@3` | `pass^3` | Loop rate | Guard rejection | Mean steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Terminal-GRPO-30 | 379/600 = 63.2% | [59.2%, 66.9%] | 74.0% | 53.0% | 11.2% | 24.5% | 11.09 |
| More-SFT `n379@288` | 394/600 = **65.7%** | [61.8%, 69.4%] | **76.0%** | **55.5%** | **9.0%** | **11.5%** | **10.33** |

More-SFT has a +2.5pp point estimate, task win/tie/loss `32/146/22`, paired CI
[-1.2,+6.2]pp, and McNemar `p=0.1756`. It also shows fewer loops, fewer Guard
rejections, and shorter trajectories. However, the interval still includes
zero; the benchmark was reused; and the checkpoints do not form a clean
same-initialization, same-compute SFT-vs-GRPO training experiment. The supported
claim is that More-SFT is a stronger practical checkpoint and competitive
control—not that extra SFT has causally defeated GRPO.

## Experiment 6: Reward v4 signal audit and GRPO engineering checks

Reward v4 was implemented as an adapter-only, target-ASIN-bonus-free
optimization objective. It did not replace environment Reward v3 or the strict
success definition. Offline replay covered 700 trajectories from the repeated
validation and frozen-confirmation runs:

| Audit item | Result |
|---|---:|
| Trajectories | 700 |
| `reward_valid=true` | 664 |
| Reward v3 strict gold successes | 443 |
| Reward v4 constraint-complete successes | 444 |
| Additional differentiating successes | **1** |
| Target-ASIN invariance failures | 0 |

Only one trajectory changed the success signal. The project therefore stopped
before a matched 100-update v3-vs-v4 comparison. Five-update v3 and v4 runs
validated reward routing, and a separate checkpoint smoke restored model,
LoRA, optimizer, scheduler, RNG, global step, data position, and dynamic-sampler
state. These are engineering results, not model-quality results.

A uniform Reward-v3 long-run attempt from `n379@144` reached step 94 before
interruption but retained no final checkpoint/evaluation. Its log showed 58
skipped updates, an effective-group ratio of about 38%, and all-equal groups on
about 51% of sampled batches. A clean restart was intentionally stopped at step
5 after reproducing the low-signal behavior. Neither run appears in performance
tables; together they diagnose terminal-reward group variance as the main
training-efficiency bottleneck.

## Experiment 7: reward-varying active-set GRPO pilot

The minimum-cost follow-up screened 48 difficult training tasks with four
More-SFT rollouts each. Twenty tasks had complete, valid, non-constant Reward v3
groups and zero overlap with Validation-50 or Final-200. Ten GRPO updates were
then trained from More-SFT, including a verified step-5→10 resume.

| Training diagnostic | Uniform long-run log | Active-set pilot |
|---|---:|---:|
| Initialization | `n379@144` | `n379@288` More-SFT |
| Effective-group ratio | about 38% | **71.7%** |
| All-equal group ratio | about 51% | **14.2%** |
| Applied optimizer updates | incomplete run | 10/10 |
| Generated / nominal rollouts | not a matched cost comparison | 136/80 = 1.7× |

The two training columns are diagnostics, not a controlled ablation, because
their initial policies and run lengths differ. The paired policy evaluation is
the valid performance comparison:

| Model | Strict success | `pass@3` | `pass^3` | Loop rate | Mean steps |
|---|---:|---:|---:|---:|---:|
| More-SFT | 105/150 = 70.0% | 84.0% | 58.0% | 4.7% | 9.91 |
| Active-GRPO-10 | 107/150 = 71.3% | 84.0% | 58.0% | 4.0% | 10.30 |

The +1.3pp difference has task win/tie/loss `9/33/8`, paired CI
[-5.3,+8.0]pp, and McNemar `p=0.8238`. `pass@3` and `pass^3` are unchanged,
mean steps rise 3.9%, and the emphasized 4+ constraint stratum falls 4.8pp.
The experiment passes the sampling-efficiency objective but fails the registered
`+2pp` and non-negative-lower-bound promotion gates. Training stopped and no
additional Final-200 run was performed.

## Cross-experiment analysis

### 1. SFT creates the agent; RL is a marginal intervention

The largest robust change is Base→SFT: 0.0% to 60.5% in the initial pipeline,
with valid terminal behavior rising from 18.0% to 96.5%. This is not merely
product knowledge. SFT teaches the state-dependent tool grammar, evidence
navigation, option selection, and when to terminate.

### 2. The observed GRPO gain is unstable across evaluation stages

Terminal-GRPO-30 improves Validation-50 by +8.0pp with a positive paired
interval, but improves the frozen-stage Final-200 run by only +1.5pp with an
interval crossing zero. The likely explanations include development-set
selection, task-distribution differences, inference variance, and training-
seed variance; the current experiments cannot identify their individual
contributions. The evidence rules out a confident +8pp generalization claim.

### 3. More-SFT is the strongest low-cost control

More-SFT reaches 70.0% on seed-42 Validation-50×3 and has a +2.5pp post-hoc
Final-200×3 point estimate over Terminal-GRPO-30, together with better loop and
Guard metrics. Neither difference is statistically established. Still, these
results show that an RL comparison without matched extra-SFT compute can
misattribute ordinary supervised optimization gains to GRPO.

### 4. Terminal Reward v3 shows a within-group variance bottleneck

The uniform GRPO logs contain many all-equal groups and skipped updates. The
active-set pilot observed more than twice the effective-group ratio, but the
runs differ in initialization and length, so this is not a controlled causal
estimate of the screening mechanism. Policy success barely changes. The useful
signal may still be too sparse, too late, or poorly aligned with the complex 4+
constraint failures that matter most; turn-level credit assignment remains an
untested hypothesis.

### 5. Reward v4 did not add enough independent supervision

An optimization reward can only help if it changes ranking or credit assignment
on a meaningful number of trajectories. Reward v4 changed both the scalar
optimization reward and the success signal on only 1/700 audited trajectories.
It was therefore nearly equivalent to Reward v3 on the audited data. Stopping
after offline audit and integration smoke was the correct budget decision; this
does not resolve the separate turn-level credit-assignment hypothesis.

### 6. Efficiency improvements and capability improvements are different

The active-set run observed a higher fraction of batches that produced a
gradient; More-SFT reduced loops and trajectory length; Terminal-GRPO sometimes
reduced Guard errors. None of those engineering or behavior metrics alone
establishes strict-success improvement. The project reports them separately and
promotes a method only through paired task success.

## Status of planned or stopped branches

| Branch | Status | Why it is not a performance result |
|---|---|---|
| Seed replay and repeated-statistics protocol | Completed | Infrastructure/statistics validation; no model training effect |
| SFT checkpoint resume | Completed | Step continuity validation only |
| GRPO dynamic-state resume | Completed | State restoration validation only |
| Reward-v3 G4 and Reward-v4 G4 five-update runs | Completed smoke | Too short and not evaluated as a causal pair |
| Uniform Reward-v3 100-update rerun | Interrupted/stopped | No retained final checkpoint or paired evaluation; low effective-group signal |
| Reward-v4 100-update comparison | Not entered | Offline audit found only 1/700 differentiating trajectories |
| G2/G8, KL, new held-out branches | Not entered | Conditional gates were not met |
| Evidence Memory, length curriculum, Difficulty Curriculum | No reportable result in the current experiment contract | No retained complete paired artifact under the final protocol; no metric or claim is made |

The final defensible statement is:

> SFT establishes nearly all usable shopping-agent capability. GRPO can produce
> a positive development signal and active-set sampling can produce more
> effective groups in a pilot,
> but neither the frozen-stage confirmation, post-hoc repeated comparison, nor
> the active-set pilot establishes a reliable strict-success gain over a strong
> More-SFT control.
