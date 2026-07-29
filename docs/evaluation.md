# Evaluation

## Protocol

Baseline, SFT and GRPO are evaluated on the same 200 held-out tasks with no
training overlap. Each task receives one deterministic rollout:

| Setting | Value |
|---|---|
| Environment / reward | v2.1 / Reward v3 |
| Maximum environment steps | 35 |
| Maximum generated tokens per turn | 512 |
| Temperature / top-p | 0.0 / 1.0 |
| Context window / safety margin | 24,576 / 512 |
| Context compaction | disabled |
| Search observation budget | 1,536 tokens |
| Product-detail budget | 4,096 tokens |
| Generic fallback budget | 768 tokens |
| Visible search results | top 20 |

The task file is `data/evaluation/tasks.jsonl`; its SHA-256 is
`2c4ff070e13ddc30796d38e85170210e7d3c211992425a62090f2419fe8e0208`.

## Run one model

Keep ShopSimulator running, then start an OpenAI-compatible model endpoint:

```bash
bash scripts/serve_model.sh MODEL_OR_LOCAL_PATH
```

In another terminal:

```bash
bash scripts/evaluate.sh MODEL_NAME
```

For the untouched base model, `bash scripts/baseline.sh` is shorthand for
`bash scripts/evaluate.sh baseline`.

Outputs:

```text
outputs/evaluation/MODEL_NAME/trajectories.jsonl
outputs/evaluation/MODEL_NAME/summary.json
```

## Metrics

`strict_success_rate` is primary. A strict success requires:

1. a completed environment terminal state;
2. a valid Reward v3 result;
3. `gold_purchase`.

`purchase_success_rate` additionally accepts a fully matching
`valid_alternative_purchase`. Missing, failed and invalid tasks stay in the
200-task denominator. `mean_final_reward` also captures partial matches, loops,
wrong purchases and unverifiable outcomes.

The checked-in result summaries are compact audit records. Full trajectories
are generated artifacts and remain under the ignored `outputs/` directory.
See [`experiments/comparison.md`](../experiments/comparison.md).
