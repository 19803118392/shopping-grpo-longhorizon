# Validation-50×3 paired evaluation

This is the compact, Git-tracked card for the project's repeated method-selection
experiment. Raw trajectories and model weights remain under the Git-ignored
`outputs/` tree.

> Final-200 subsequently measured SFT 57.0% versus Terminal-GRPO 58.5%, with a
> paired 95% CI of [-2.0, +5.0] points and McNemar p=0.6072. The validation gain
> therefore did not generalize as a statistically reliable final improvement.
> See the [Final-200 experiment card](../final-200/README.md).

## Protocol

- dataset: first 50 tasks in `data/grpo/validation.jsonl`;
- models: merged SFT and Terminal-GRPO after 30 updates;
- three attempts per task, for a fixed denominator of 150 attempts per model;
- paired seeds derived from `(2026, task_id, attempt_index)`;
- temperature/top-p `0.7/0.9`, maximum 35 environment steps;
- strict success requires `gold_purchase` and `reward_valid=true`;
- 10,000 task-paired bootstrap samples and an exact paired-attempt McNemar test.

The local vLLM seed replay matched all 10/10 task-attempt pairs after ignoring
timestamps and trajectory UUIDs, so the pairing is interpreted as common random
numbers rather than task pairing alone.

## Result

| Model | Strict success | Wilson 95% CI | `pass@3` | `pass^3` | Loop rate | Mean steps |
|---|---:|---:|---:|---:|---:|---:|
| SFT | 100/150 = 66.7% | [58.8%, 73.7%] | 78.0% | 54.0% | 10.0% | 10.95 |
| Terminal-GRPO (30 updates) | 112/150 = **74.7%** | **[67.2%, 81.0%]** | **84.0%** | **62.0%** | **8.7%** | **10.80** |

Terminal-GRPO minus SFT is **+8.0 percentage points**, with task-level
win/tie/loss `9/39/2`, paired-bootstrap 95% CI **[+2.0, +14.7] points**, and
exact McNemar `p=0.0118179` (`candidate-only=16`, `baseline-only=4`).

Both runs have 150/150 coverage, zero infrastructure-invalid attempts, and zero
critical footer failures. SFT has six context-budget errors and Terminal-GRPO
has three; these remain in the denominator. Guard rejection rate increases from
24.7% to 28.7%, so the result should not be described as an across-the-board
behavioral improvement.

## Scope

This is a development-set method comparison. Query-derived strata are
exploratory and have no multiple-comparison correction. The result is not a
Final-200 score and must not be directly subtracted from the project's earlier
single-pass 60.5%/62.0% pipeline benchmark because the checkpoints and protocols
differ.

Machine-readable protocol and result files are [run_config.json](run_config.json)
and [summary.json](summary.json).

`run_config.json` records both the config hash stored by the training manifest
and the current repository config hash. They differ because launcher reliability
edits were made after this 30-update checkpoint was trained; the model hash, not
the current config file, identifies the evaluated checkpoint.
