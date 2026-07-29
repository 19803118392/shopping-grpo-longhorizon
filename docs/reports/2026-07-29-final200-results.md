# Final 200 Evaluation Results

## Scope and interpretation

This report summarizes a single deterministic rollout per task on 200 frozen
tasks. It is not a Pass@k experiment: there were no repeated samples per task.
The primary success statistic is therefore the task-level strict gold success
rate, reported separately from Reward, Rubric, trajectory quality and efficiency.

## Environment Reward and terminal outcomes

| Metric | Base | SFT | GRPO step 100 |
|---|---:|---:|---:|
| Strict gold successes | 0/200 | 121/200 | 124/200 |
| Strict gold success rate | 0.0% | 60.5% | 62.0% |
| Purchase success rate | 0.0% | 60.5% | 62.5% |
| Mean final Reward | -0.1105 | 0.4729 | 0.5158 |
| Mean weighted score | 0.0063 | 0.7168 | 0.7496 |
| Reward-valid tasks | 34 | 192 | 192 |
| Infrastructure-invalid tasks | 2 | 5 | 5 |

The GRPO-to-SFT paired transition table contains 12 failure-to-success changes
and 9 success-to-failure changes, for a net gain of 3 strict successes.

## Reward type distribution

| Reward type | Base | SFT | GRPO step 100 |
|---|---:|---:|---:|
| `gold_purchase` | 0 | 121 | 124 |
| `partial_alternative_purchase` | 0 | 32 | 34 |
| `valid_alternative_purchase` | 0 | 0 | 1 |
| `wrong_purchase` | 0 | 7 | 5 |
| `repeat_loop` | 34 | 27 | 25 |
| `max_steps` | 0 | 5 | 3 |
| `reward_unverifiable` | 2 | 1 | 1 |
| unknown / infrastructure | 164 | 7 | 7 |

## Five-dimensional trajectory quality

Scores are 0/1/2 and are averaged over valid Judge results only. The dimensions
are ordered independently; they are not weighted or summed.

| Dimension | Base | SFT | GRPO step 100 | GRPO − SFT |
|---|---:|---:|---:|---:|
| Candidate Utilization | 0.641 | 1.487 | 1.533 | +0.046 |
| Decision Quality | 0.056 | 1.503 | 1.574 | +0.072 |
| Evidence Verification | 0.212 | 1.226 | 1.251 | +0.026 |
| Search Strategy | 0.919 | 1.569 | 1.595 | +0.026 |
| Termination and Efficiency | 0.015 | 1.446 | 1.503 | +0.056 |

The mean score was higher for GRPO than SFT in all five dimensions. The paired
task table still contains both improvements and regressions; the largest mean
changes were in decision quality and termination efficiency.

## Requirement Rubric

GRPO had fewer aggregate hard-constraint violations than SFT. In the paired
comparison, 12 tasks improved on hard violations and 4 worsened. Reward-Rubric
disagreement was recorded independently: SFT had 25 disagreement tasks and GRPO
had 27. This disagreement is diagnostic and does not override either panel.

## Deterministic behavior and efficiency

| Metric | SFT | GRPO step 100 | Change |
|---|---:|---:|---:|
| Average action attempts | 12.595 | 12.040 | -0.555 |
| Average executed tool steps | 12.335 | 11.850 | -0.485 |
| Duplicate canonical actions | 718 | 657 | -61 |
| Duplicate search queries | 28 | 24 | -4 |
| Action Guard rejections | 52 | 38 | -14 |
| Repeat-loop tasks | 27 | 25 | -2 |
| Max-step tasks | 5 | 3 | -2 |
| Wrong-purchase tasks | 7 | 5 | -2 |

The lower step count is interpreted together with the higher success rate and
better termination Judge score. It should not be treated as a standalone
objective.

## Result files

- `outputs/eval/final_reward_v3_200_20260729/base/evaluations.jsonl`
- `outputs/eval/final_reward_v3_200_20260729/sft/evaluations.jsonl`
- `outputs/eval/final_reward_v3_200_20260729/grpo_step100/evaluations.jsonl`
- `outputs/eval/final_reward_v3_200_20260729/model_comparison.json`
