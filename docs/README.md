# Documentation

Follow the guides in workflow order:

1. [Data collection](data-collection.md) explains how the checked-in SFT data
   was produced and audited.
2. [SFT](sft.md) trains the first useful shopping agent.
3. [GRPO](grpo.md) improves that model with online environment reward.
4. [Evaluation](evaluation.md) compares baseline, SFT and GRPO fairly.

[Reward v3](reward-v3.md) is the detailed specification shared by collection,
GRPO and evaluation.

[Statistical evaluation](local-upgrades.md) defines the fixed-denominator
repeated-sampling protocol, and the [GPU runbook](gpu-runbook.md) contains the
matched 50×3 execution commands. The compact verified result is stored in the
[Validation-50×3 experiment card](../experiments/validation-50x3/README.md).

The subsequent one-pass frozen result is stored in the
[Final-200 experiment card](../experiments/final-200/README.md). It did not
establish a statistically reliable GRPO gain and may not be rerun for tuning.

The static [Final-200 dashboard](evaluation-dashboard.html) presents the initial
single-pass pipeline benchmark; it is separate from the later 50×3 method-
selection result and frozen confirmation.
