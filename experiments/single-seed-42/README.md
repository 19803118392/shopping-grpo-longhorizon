# Single-seed SFT, More-SFT, and GRPO mechanism experiments

This is the compact, Git-tracked result card for the low-cost `seed=42`
mechanism experiments. Large adapters, checkpoints, logs, and raw trajectories
remain under the Git-ignored `outputs/single-seed-42/` and
`outputs/low-cost-grpo-pilot/` trees. Their repository-relative identities,
sizes, row counts, and SHA-256 values are frozen in the
[evidence manifest](evidence-manifest.json). A clone can audit the code and
aggregates; trajectory-level recomputation additionally requires matching local
artifacts or release assets.

## Protocol

- training seed: `42`;
- SFT data: nested, stratified subsets of 95, 190, and 379 strict-success
  trajectories, with zero overlap against validation and Final-200 tasks;
- SFT compute control: 144 optimizer steps for each data-size comparison;
- More-SFT control: the 379-trajectory run continued from step 144 to step 288,
  restoring model, LoRA, optimizer, scheduler, RNG, and global step;
- evaluation: all 50 tasks in `data/grpo/validation.jsonl`, three paired attempts
  per task, temperature/top-p `0.7/0.9`, maximum 35 environment steps, seed root
  `42`, and a fixed denominator of 150 attempts per model;
- strict success: complete `gold_purchase` with `reward_valid=true`;
- inference coverage: 150/150 attempts for every configuration;
- uncertainty: Wilson intervals, 10,000 task-paired bootstrap samples, and exact
  McNemar tests.

This protocol estimates task and rollout-sampling uncertainty for one training
seed. It does not estimate variance across training seeds.

Before the full runs, three five-task protocol smokes compared auto tool choice
without compaction, auto with compaction, and required tool choice with
compaction. Completion was 4/5, 5/5, and 5/5 respectively; Reward-valid coverage
was 3/5, 4/5, and 5/5. All three had zero strict successes on this tiny task
slice, so they were integration checks rather than a model-selection experiment.
The registered full protocol used auto tool choice with compaction.

## SFT construction and training compute

The nested subsets were defined before training. Overlength filtering at the
24,576-token model limit explains why the nominal 190/379 rows become 189/376
usable examples. The 379-row run emitted step 144 and then continued in the
same optimizer/scheduler trajectory to step 288.

| Run | Nominal / usable train rows | Target step | Wall time | Peak GPU memory | Final reported train loss |
|---|---:|---:|---:|---:|---:|
| `n95@144` | 95 / 95 | 144 | 92.5 min | 60.50 GiB | 0.1452 |
| `n190@144` | 190 / 189 | 144 | 95.3 min | 73.53 GiB | 0.2322 |
| `n379@144` | 379 / 376 | 144 | intermediate checkpoint | 73.53 GiB | not separately finalized |
| `n379@288` | 379 / 376 | 288 | 199.3 min total | 73.53 GiB | 0.2220 |

All runs used LoRA rank/alpha 16/32, learning rate `1e-4`, Assistant-only loss,
gradient accumulation 8, bf16 weights, and seed 42. Train losses should not be
compared as a data-scaling metric because the smaller subsets receive more
effective epochs at the same optimizer-step budget.

## SFT result

| Configuration | Strict success | Wilson 95% CI | `pass@3` | `pass^3` | Loop rate | Mean steps |
|---|---:|---:|---:|---:|---:|---:|
| `n95@144` | 93/150 = 62.0% | [54.0%, 69.4%] | 74.0% | 46.0% | 10.7% | 12.18 |
| `n190@144` | 99/150 = 66.0% | [58.1%, 73.1%] | 82.0% | 48.0% | 8.0% | 11.06 |
| `n379@144` | 100/150 = 66.7% | [58.8%, 73.7%] | 80.0% | 56.0% | 18.0% | 10.87 |
| `n379@288` | 105/150 = **70.0%** | [62.2%, 76.8%] | **84.0%** | **58.0%** | **4.7%** | **9.91** |

| Paired comparison | Delta | Task win/tie/loss | Bootstrap 95% CI | McNemar p |
|---|---:|---:|---:|---:|
| `n95@144 → n190@144` | +4.0 points | 13/31/6 | [-3.3,+11.3] | 0.3269 |
| `n190@144 → n379@144` | +0.7 points | 10/30/10 | [-6.0,+7.3] | 1.0000 |
| `n95@144 → n379@144` | +4.7 points | 12/32/6 | [-4.7,+14.0] | 0.3105 |
| `n379@144 → n379@288` | +3.3 points | 8/37/5 | [-2.7,+10.0] | 0.4421 |

The point estimates trend upward with more distinct trajectories and with more
SFT compute, while the mean trajectory length also falls. Every paired interval
includes zero. The defensible conclusion is therefore a positive development-
set trend, not a statistically established scaling gain.

**Direct interpretation:** simply doubling SFT from 144 to 288 optimizer steps
raised the development-set point estimate from 66.7% to 70.0% (+3.3 points),
while cutting the loop rate from 18.0% to 4.7%. This +3.3-point estimate is
larger than the project's +1.5-point frozen Final-200 GRPO estimate, but it is
not a same-protocol superiority result. On the matched Validation-50×3
comparison, Terminal-GRPO gained +8.0 points over SFT, which is larger than the
More-SFT gain. The correct claim is that additional SFT compute is a strong,
cheaper control that can explain or exceed a small GRPO gain—not that SFT has
been proven better than GRPO overall.

## Post-hoc Final-200×3: More-SFT versus Terminal-GRPO

After earlier Final-200 results were already visible, the More-SFT checkpoint
and Terminal-GRPO-30 were each sampled three times on all 200 evaluation tasks.
The protocol used seed root 42, temperature/top-p `0.7/0.9`, a 35-step cap, and
600 attempts per model. The summaries explicitly record `final_200=false` and
`holdout_status=posthoc_reused`.

| Model | Strict success | Wilson 95% CI | `pass@3` | `pass^3` | Loop rate | Guard rejection | Mean steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Terminal-GRPO-30 | 379/600 = 63.2% | [59.2%, 66.9%] | 74.0% | 53.0% | 11.2% | 24.5% | 11.09 |
| More-SFT `n379@288` | 394/600 = **65.7%** | [61.8%, 69.4%] | **76.0%** | **55.5%** | **9.0%** | **11.5%** | **10.33** |

The More-SFT point estimate is +2.5 points, task win/tie/loss is `32/146/22`,
the paired-bootstrap interval is `[-1.2,+6.2]` points, and exact McNemar
`p=0.1756`. More-SFT also has 13 fewer loop attempts, 78 fewer Guard-rejected
attempts, and shorter trajectories over the 600-attempt comparison.

This is useful evidence that More-SFT is a strong practical checkpoint and an
essential RL control. It is not a new final test or a clean causal proof that
SFT beats GRPO: the benchmark was reused after results were known, the models
come from different training runs/recipes, and the paired interval includes
zero.

## Trace-backed failure cases

The local, Git-ignored raw trajectories support concrete failure analysis rather
than generic labels. The table below records only actor-visible actions and
validated public Reward fields; it does not expose target-product fields. These
cases are diagnostic examples, not an additional evaluation metric.

| Task / attempt | Observed trajectory | Terminal outcome | Diagnosis |
|---|---|---|---|
| Terminal-GRPO `8187/0` | `search_products → open_product`, followed by three rejected clicks | `invalid_action_limit` | The policy kept selecting an action that was absent from the latest Observation; the Guard prevented stale-page execution. |
| Terminal-GRPO `9610/0` | 34 steps with repeated search, detail views, back navigation, and option selection | `repeat_loop` | The policy gathered evidence but failed to converge on a terminal decision. |
| Terminal-GRPO `11773/1` | Repeated search/open/view/back cycles for all 35 allowed steps | `max_steps` | Exploration lacked a stopping rule and exhausted the environment budget. |
| Terminal-GRPO `14666/0` | Selected an option and bought after eight steps | `wrong_purchase` | The final variant cost 49 against a budget of 44; category passed, but the budget hard gate failed. |
| Terminal-GRPO `20345/0` | Verified features, description, two option axes, reviews, then bought | `reward_unverifiable` | Multiple option axes affected price, so the environment could not derive one verifiable final variant price. This is a Reward/data-contract ambiguity, not merely a policy error. |

The corresponding source is
`outputs/single-seed-42/evaluation/posthoc-final200x3/terminal-grpo-30/raw.jsonl`.
It remains outside Git because of its size; the experiment summary and the
trace signatures above are the compact review artifacts. Together the cases
cover stale actions, loops, overlong search, hard-constraint violation, and
Reward unverifiability.

## Low-cost active-set GRPO pilot

The cheapest remaining GRPO hypothesis was that uniform sampling wasted most
rollouts on groups whose four Reward v3 outcomes were identical. Starting from
the `n379@288` More-SFT model, a fixed screen evaluated 48 difficult training
tasks four times each. Twenty tasks (41.7%) had four complete, valid trajectories
and non-constant terminal rewards; these tasks formed the active training set.
The selected task IDs have zero overlap with Validation-50 and Final-200.

The pilot then ran ten Reward-v3 GRPO updates with `G=4`, resuming once from
step 5 to step 10. All ten optimizer updates were applied. The mean effective
group ratio was 71.7%, the mean all-equal ratio was 14.2%, and 136 rollouts were
generated for 80 nominal training rollouts (1.7× sampling cost). These observed
ratios are higher than the earlier uniform-run diagnostic, but the runs differ
in initialization and length, so the difference is not a controlled estimate of
the effect of active screening.

The merged step-10 checkpoint and More-SFT control were compared on the same
Validation-50×3 attempts:

| Configuration | Strict success | Wilson 95% CI | `pass@3` | `pass^3` | Loop rate | Mean steps |
|---|---:|---:|---:|---:|---:|---:|
| More-SFT `n379@288` | 105/150 = 70.0% | [62.2%, 76.8%] | 84.0% | 58.0% | 4.7% | 9.91 |
| Active-GRPO-10 | 107/150 = **71.3%** | [63.6%, 78.0%] | 84.0% | 58.0% | 4.0% | 10.30 |

The paired delta is only **+1.3 points**, with task win/tie/loss `9/33/8`, a
paired-bootstrap 95% interval of `[-5.3,+8.0]` points, and exact McNemar
`p=0.8238`. Mean steps increased 3.9%; infrastructure invalid attempts and
critical footer failures were both zero. The exploratory 4+ constraint stratum,
which the active screen emphasized, fell 4.8 points rather than improving.

This fails the predeclared `+2pp` and non-negative-bootstrap-lower-bound gates,
while `pass@3` and `pass^3` are unchanged. Training therefore stopped at ten
updates and no Final-200 evaluation was run. The supported result is narrower:
the active-set run contained fewer low-information GRPO batches, but this ten-
update, single-seed pilot did **not** demonstrate either a causal sampling-
efficiency effect or an algorithmic gain over More-SFT.

## Reward v4 signal audit

Reward v4 was implemented only as an ASIN-neutral training objective; it does
not replace the environment's Reward v3 or the strict-success definition. An
offline replay covered 700 existing trajectories:

| Audit item | Result |
|---|---:|
| Total trajectories | 700 |
| `reward_valid=true` | 664 |
| Reward v3 strict gold successes | 443 |
| Reward v4 constraint-complete successes | 444 |
| Scalar optimization rewards changed | **1 / 700** |
| Additional differentiating successes | **1** |
| Target-ASIN invariance failures | 0 |

The only additional constraint-complete non-gold purchase was manually reviewed.
Because just 1/700 trajectories changed either the scalar optimization reward or
the success signal, the proposed v4 objective was too similar to Reward v3 to
justify a costly full causal comparison on the available budget.

## GRPO engineering validation and boundary

- Reward v3 and Reward v4 each reached five effective online GRPO updates from
  the same `n379@144` initialization; these runs validated reward routing and
  bounded dynamic resampling only.
- A checkpoint-resume smoke continued from global step 1 to 2 while restoring
  actor, optimizer, scheduler, RNG, data, and adaptive sampling state.
- Periodic resumable checkpoints are exposed through `--checkpoint-every`.
- No matched, completed Reward-v3-vs-v4 100-update pair exists. Interrupted or
  deliberately stopped long-run artifacts are excluded from all performance
  tables, so this experiment makes **no Reward v4 improvement claim**.
- The conditional G2/G8, KL, and new held-out branches were not entered.

A uniform Reward-v3 run from `n379@144` reached step 94 before interruption and
retained no final checkpoint suitable for evaluation. Across steps 1–94, the
log records about 37.9% effective groups, about 51.4% all-equal groups, 58
skipped updates, and 1,872 generated rollouts. A clean restart was stopped at
step 5 after reproducing the same low-signal pattern. These are sampler
diagnostics only; neither interrupted run is a model-quality result.

This diagnosis motivated the later active-set pilot. Its 71.7% effective-group
ratio is not a controlled replacement for the 37.9% number because it starts
from More-SFT rather than `n379@144` and runs for only ten updates. The valid
claim comes from the matched More-SFT versus Active-GRPO-10 evaluation above,
which failed its performance promotion gate.

These results do not change the project's frozen-stage Final-200 conclusion.
The later Final-200×3 comparison is explicitly post-hoc and is not used for
further tuning. See the [Final-200 card](../final-200/README.md),
[summary.json](summary.json), and [evidence manifest](evidence-manifest.json) for
the machine-readable values and reproducibility boundary above.
