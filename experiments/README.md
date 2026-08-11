# Experiments

This directory contains compact, Git-tracked experiment artifacts. Large
checkpoints and complete trajectories are intentionally kept under the ignored
`outputs/` tree.

```text
final-200/         frozen-confirmation one-pass evaluation
validation-50x3/  paired repeated method-selection evaluation
single-seed-42/   SFT scaling, More-SFT post-hoc evaluation, active-set GRPO, and Reward v4
baseline/          pipeline benchmark config and summary
sft/               SFT training/evaluation config and summary
grpo/              GRPO training/evaluation config and summary
comparison.md
```

Together these directories record three main stages and one later mechanism-
analysis phase of the same project:
`baseline/`, `sft/`, and `grpo/` contain the initial pipeline benchmark;
[`validation-50x3/`](validation-50x3/) contains the repeated development-set
comparison; and [`final-200/`](final-200/) contains the frozen confirmation.
[`single-seed-42/`](single-seed-42/) records the later low-cost SFT scaling
ablation, post-hoc Final-200×3 More-SFT comparison, active-set Reward-v3 GRPO
pilot, Reward v4 offline audit, and the explicit stop decisions for hypotheses
that did not pass their development-set promotion gates.
See [comparison.md](comparison.md) for the unified result table, protocol
boundaries, and limitations.
