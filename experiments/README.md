# Experiments

All reported models use the same 200 held-out tasks, Environment v2.1 and
Reward v3. Each task receives one deterministic rollout.

| Model | Strict success | Mean reward |
|---|---:|---:|
| Qwen3.5-2B baseline | 0.0% | -0.1105 |
| LoRA SFT | 60.5% | 0.4729 |
| GRPO step 100 | 62.0% | 0.5158 |

The large capability gain comes from SFT. GRPO adds a smaller improvement in
success, legal actions and termination efficiency. The reproducible inputs are
[`data/`](../data/), the GRPO recipe is
[`configs/grpo.yaml`](../configs/grpo.yaml), and all generated checkpoints and
rollouts go under `outputs/`.
