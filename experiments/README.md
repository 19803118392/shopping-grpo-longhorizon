# Experiments

This directory contains compact, Git-tracked experiment artifacts. Large
checkpoints and complete trajectories are intentionally kept under the ignored
`outputs/` tree.

```text
final-200/         current frozen one-pass final evaluation
validation-50x3/  current-branch paired repeated evaluation
baseline/   base-model evaluation config and summary
sft/        SFT training/evaluation config and summary
grpo/       GRPO training/evaluation config and summary
comparison.md
```

[`final-200/`](final-200/) is the frozen current-branch final evaluation;
[`validation-50x3/`](validation-50x3/) is the development-set repeated
comparison that preceded it. The `baseline/`, `sft/`, and `grpo/` directories
are imported upstream records. See [comparison.md](comparison.md) for the three
protocols and their limitations.
