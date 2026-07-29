# GRPO Step 100 Analysis

## Main conclusion

On the frozen final 200, GRPO step 100 is better than fresh SFT on every major
reported panel, but the improvement in strict task success is small rather than
dramatic. The evidence supports “consistent, modest improvement,” not a claim of
large or statistically conclusive gain.

## What improved

The strict gold success rate increased from 60.5% to 62.0%. The paired task view
is more informative than the aggregate rate: GRPO converted 12 SFT failures into
successes and regressed 9 SFT successes, a net gain of 3 tasks.

The Reward increase (0.4729 to 0.5158) is consistent with the terminal result:
gold purchases increased by 3, wrong purchases decreased by 2, repeat loops
decreased by 2, and max-step terminations decreased by 2.

The Judge results show the largest relative behavioral gains in decision quality
and termination efficiency. This suggests that the GRPO update improved candidate
selection and convergence behavior more than it improved raw search strategy.
Evidence verification and search strategy also improved, but only slightly.

The deterministic data points in the same direction. GRPO used fewer action
attempts, made fewer illegal-action Guard rejections, and repeated fewer actions
and searches. Because GRPO also improved success and termination scores, the
reduction in steps is more plausibly improved convergence than premature stopping
in this run.

## What did not improve clearly

The result is not a universal win on every individual task. The paired table has
both improvements and regressions, and GRPO has 27 Reward-Rubric disagreement
tasks versus 25 for SFT. These disagreements are intentionally reported without
forcing one signal to replace the other.

The final benchmark uses one rollout per task. Therefore the result does not
measure pass@k, best-of-k behavior, sampling robustness or variance across random
rollouts. A 3-task net gain on 200 tasks should be treated as encouraging but
limited evidence.

## Base-to-SFT-to-GRPO interpretation

Base is substantially below both trained actors: it has no strict gold successes,
many invalid/repeating actions and a mean Reward below zero. SFT supplies the
large capability jump. GRPO step 100 adds a smaller refinement over SFT: better
decision quality, termination, legality and efficiency, with a modest additional
success gain.

This pattern is consistent with GRPO acting as behavioral refinement rather than
creating the core shopping capability from scratch.

## Metrics not available in this experiment

The Collector records input/context token statistics and observation projection
costs, but the formal run does not contain reliable end-to-end model latency,
tool latency, completion-token usage, TTFT, tokens-per-second, p50/p95 latency or
cost-per-successful-task. The `timing` fields in the final deterministic panels
are null. Consequently, this report makes no claim that GRPO is faster, cheaper
or has lower first-token latency than SFT.

## Limitations and next experiments

1. Repeat a selected subset or the full benchmark with multiple rollouts per task
   if pass@k or outcome variance is needed.
2. Add streaming timestamps and API usage capture for TTFT, completion tokens,
   end-to-end latency and cost.
3. Calibrate the Pro Judge with human labels and report agreement before treating
   small quality-score differences as definitive.
4. Build a fixed badcase regression set from the 12 GRPO gains, 9 regressions and
   the major error categories.

## Reproducibility

The benchmark SHA256, protocol and model paths are recorded in
`docs/experiments/11-trajectory-evaluation-final200-2026-07-29.md`. The raw and
assembled JSONL artifacts remain in the local `outputs/eval/` directory and are
not committed to Git because of their size and private trajectory content.
