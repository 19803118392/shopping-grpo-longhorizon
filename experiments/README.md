# Experiments

This directory contains compact, Git-tracked experiment artifacts. Large
checkpoints and complete trajectories are intentionally kept under the ignored
`outputs/` tree.

```text
validation-50x3/  current-branch paired repeated evaluation
baseline/   base-model evaluation config and summary
sft/        SFT training/evaluation config and summary
grpo/       GRPO training/evaluation config and summary
comparison.md
```

[`validation-50x3/`](validation-50x3/) is the only rerun performed under the
current statistical protocol. The `baseline/`, `sft/`, and `grpo/` directories
are imported upstream records from one deterministic pass over 200 held-out
tasks. See [comparison.md](comparison.md) for the two protocols and their
limitations.
