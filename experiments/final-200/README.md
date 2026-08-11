# Frozen Final-200 evaluation

This is the compact, Git-tracked experiment card for the one authorized
Final-200 run on the current branch. Raw trajectories, model weights, and full
freeze manifests remain under the Git-ignored `outputs/final-200/` tree.

## Frozen protocol

- frozen commit: `a48019f8c4cc3eb61361c88190a730c0267aa437`;
- benchmark: all 200 tasks in `data/evaluation/tasks.jsonl`;
- models: merged SFT and Terminal-GRPO after 30 updates;
- one deterministic attempt per task (`temperature=0`, `top_p=1`);
- maximum 35 environment steps;
- strict success requires `gold_purchase` and `reward_valid=true`;
- commit, checkpoint, model, config, validation report, and benchmark hashes were
  verified immediately before each run;
- results were not used to tune or rerun this frozen-stage decision;
- the 200 tasks have zero training overlap, but they had already appeared in the
  earlier pipeline benchmark, so they are not a never-before-seen test set over
  the full project history.

## Result

| Model | Strict success | Wilson 95% CI | Loop rate | Guard rejection | Mean steps |
|---|---:|---:|---:|---:|---:|
| SFT | 114/200 = 57.0% | [50.1%, 63.7%] | 11.5% | 28.0% | 11.34 |
| Terminal-GRPO (30 updates) | 117/200 = 58.5% | [51.6%, 65.1%] | 11.0% | 21.5% | 11.58 |

Terminal-GRPO minus SFT is **+1.5 percentage points**, with task win/tie/loss
`9/185/6`, task-paired bootstrap 95% CI **[-2.0, +5.0] points**, and exact
McNemar `p=0.6072388` (`candidate-only=9`, `baseline-only=6`). The confidence
interval includes zero, so this run does **not** establish a reliable final GRPO
improvement.

Both models cover 200/200 tasks with zero infrastructure-invalid attempts and
zero critical footer failures. SFT has three context-budget errors; Terminal-GRPO
has six. These remain in the fixed denominator.

## Interpretation

The earlier Validation-50×3 comparison showed +8.0 points with a positive
bootstrap interval. Final-200 did not reproduce that effect size. The defensible
conclusion is therefore:

> Terminal-GRPO has a small positive point estimate and fewer Guard rejections,
> but the frozen test does not provide statistical evidence of a reliable strict-
> success improvement over SFT.

No hyperparameter, prompt, reward, checkpoint, or inference setting may be
changed in response to this result and then presented as part of the same frozen
confirmation.

## Later post-hoc repeated evaluation

A later cost-controlled study sampled Terminal-GRPO-30 and More-SFT
`n379@288` three times on the same 200 tasks. It used stochastic decoding
(`temperature/top-p=0.7/0.9`) and seed root 42, after prior Final-200 results
were already known. Its summaries explicitly set `final_200=false` and
`holdout_status=posthoc_reused`.

| Model | Strict success | `pass@3` | `pass^3` | Loop rate | Mean steps |
|---|---:|---:|---:|---:|---:|
| Terminal-GRPO-30 | 379/600 = 63.2% | 74.0% | 53.0% | 11.2% | 11.09 |
| More-SFT `n379@288` | 394/600 = **65.7%** | **76.0%** | **55.5%** | **9.0%** | **10.33** |

More-SFT minus Terminal-GRPO is +2.5 points, task win/tie/loss `32/146/22`,
paired CI `[-1.2,+6.2]` points, and McNemar `p=0.1756`. This supports More-SFT
as a strong practical control but does not replace the frozen result: the
interval includes zero, the benchmark is reused, and the checkpoints do not
form a clean same-initialization training ablation. Full details are in the
[single-seed mechanism card](../single-seed-42/README.md).

Machine-readable provenance and metrics are in [run_config.json](run_config.json)
and [summary.json](summary.json).
