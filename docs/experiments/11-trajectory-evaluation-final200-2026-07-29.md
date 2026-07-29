# Experiment 11: Base/SFT/GRPO Trajectory Evaluation on Frozen Final 200

**Date:** 2026-07-29  
**Status:** completed  
**Checkpoint under test:** fresh GRPO `global_step_100`

## Objective

Compare the Base, fresh SFT and fresh GRPO actors on the same frozen ShopSimulator
task set. The evaluation is intended to diagnose shopping behavior, not to produce
a single composite score.

## Frozen assets

| Asset | Value |
|---|---|
| Benchmark | `data/benchmarks/shop_benchmark_reward_v3_final_200.jsonl` |
| Task count | 200 |
| Benchmark SHA256 | `2c4ff070e13ddc30796d38e85170210e7d3c211992425a62090f2419fe8e0208` |
| Metadata SHA256 | `42d7adc26ed48430da3def670453f44ee8a69f8ac7bbe5729a5cefa7bbd47b1b` |
| Contract | Environment v2.1 / Reward v3 / fresh-v1 |
| Rubric model | DeepSeek V4 Flash |
| Judge model | DeepSeek V4 Pro |
| Rubric count | 200 task bundles, 1,265 constraints |

The final benchmark was not used for prompt tuning, checkpoint selection or manual
calibration. All three actors used the same frozen Rubric bundles.

## Actor checkpoints

| Actor | Model path | Rollout count |
|---|---|---:|
| Base | local Qwen3.5-2B snapshot | 200 |
| SFT | `qwen35-2b-sft-v1-fresh-merged` | 200 |
| GRPO | exported `qwen35-2b-grpo-reward-v3-fresh-v1-step100-hf` | 200 |

## Shared rollout protocol

- one rollout per task;
- temperature `0`, top-p `1`;
- maximum 35 tool steps;
- maximum 512 generated tokens per model turn;
- context window 24,576 tokens with 512-token safety margin;
- observation projection budgets: 1,536 generic, 4,096 detail, 768 generic fallback;
- search observation top-k: 20;
- same system prompt, tool schema, Collector and ShopSimulator environment;
- no context compaction in the formal run;
- every task result was appended durably and could be resumed by `(task_id, attempt_index)`.

## Evaluation panels

1. Environment Reward and terminal result;
2. Query requirement Rubric status;
3. five-dimensional trajectory Judge;
4. deterministic efficiency, legality, repetition, context and validity metrics.

The Judge did not receive Reward scores, Reward subtype, strict success labels,
hidden gold fields or post-hoc purchase truth. No panel was converted into a total
score.

## Execution record

- Base, SFT and GRPO each produced 200 unique task trajectories.
- No rollout had a release error.
- Infrastructure-invalid trajectories were retained and excluded from ordinary
  success interpretation: Base 2, SFT 5, GRPO 5.
- Judge coverage: Base 198/200, SFT 195/200, GRPO 195/200.
- The Pro Judge was resumed after a controlled CPU handoff; cached rows were
  validated and not regenerated.

## Artifacts

The complete artifacts are stored outside Git under:

`outputs/eval/final_reward_v3_200_20260729/`

Important files are `base/evaluations.jsonl`, `sft/evaluations.jsonl`,
`grpo_step100/evaluations.jsonl`, their `evaluation_summary.json` files, and
`model_comparison.json`.
