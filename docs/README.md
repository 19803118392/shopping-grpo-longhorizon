# Documentation

Follow the guides in workflow order:

1. [Data collection](data-collection.md) explains how the checked-in SFT data
   was produced and audited.
2. [SFT](sft.md) trains the first useful shopping agent.
3. [GRPO](grpo.md) improves that model with online environment reward.
4. [Evaluation](evaluation.md) compares baseline, SFT and GRPO fairly.

[Reward v3](reward-v3.md) is the detailed specification shared by collection,
GRPO and evaluation.

For the current branch, [Statistical evaluation](local-upgrades.md) defines the
fixed-denominator repeated-sampling protocol, and the [GPU runbook](gpu-runbook.md)
contains the matched 50×3 execution commands. The compact verified result is
stored in the [Validation-50×3 experiment card](../experiments/validation-50x3/README.md).

The static [Final-200 dashboard](evaluation-dashboard.html) belongs to the
imported upstream single-pass report; it is not the current branch's 50×3 result.
