# Experiments

This directory contains compact, Git-tracked experiment artifacts. Large
checkpoints and complete trajectories are intentionally kept under the ignored
`outputs/` tree.

```text
final-200/         frozen-confirmation one-pass evaluation
validation-50x3/  paired repeated method-selection evaluation
baseline/          pipeline benchmark config and summary
sft/               SFT training/evaluation config and summary
grpo/              GRPO training/evaluation config and summary
comparison.md
```

Together these directories record three stages of one project:
`baseline/`, `sft/`, and `grpo/` contain the initial pipeline benchmark;
[`validation-50x3/`](validation-50x3/) contains the repeated development-set
comparison; and [`final-200/`](final-200/) contains the frozen confirmation.
See [comparison.md](comparison.md) for the unified result table, protocol
boundaries, and limitations.
