# Statistical evaluation upgrade

This branch evaluates SFT and GRPO as a paired repeated-measures experiment. It does
not treat a one-off percentage difference as evidence of an algorithmic gain.

The first completed 50×3 run reports SFT 66.7% versus Terminal-GRPO 74.7%, a
paired difference of +8.0 percentage points with bootstrap 95% CI
[+2.0, +14.7] and exact McNemar p=0.0118. See the
[experiment card](../experiments/validation-50x3/README.md); this is a validation
result, not Final-200.

## Repeated sampling

Each `(task_id, attempt_index)` has a deterministic actor seed derived from the base
seed, task ID and attempt index. A replay check compares two independently written
runs after removing timestamps, trajectory UUIDs and server-generated tool-call IDs.
The current local vLLM stack passed exact message/action replay, so matched attempts
are interpreted as common random numbers.

Missing attempts remain in the fixed denominator. Reports include:

- pooled strict success and Wilson 95% interval;
- success for every attempt index;
- empirical `pass@k`: at least one strict success in `k` attempts;
- empirical `pass^k`: strict success in all `k` attempts;
- task-level win/tie/loss and paired bootstrap interval;
- exact McNemar test over matched attempts;
- loop, no-purchase, Guard, Reward, infrastructure and Observation diagnostics.

## Difficulty and behaviour strata

Static strata use only the actor-visible Query. Gold ASIN, target-product fields and
Reward details are prohibited. The report groups tasks by:

- estimated constraint count: `1`, `2-3`, `4+`;
- explicit option/specification selection: yes/no;
- explicit price constraint: yes/no;
- frozen reference length (`short/medium/long`) when the benchmark supplies it.

Search-call count and executed trajectory length are model behaviours rather than
task attributes. They are therefore reported separately for each model and are not
used as causal paired strata. All subgroup results are exploratory and have no
multiple-comparison correction.

## Command

```bash
.venv/bin/python scripts/compare_repeated_evaluations.py \
  --benchmark data/grpo/validation.jsonl --limit 50 \
  --baseline outputs/eval/sft-50x3/raw.jsonl \
  --candidate outputs/eval/terminal-grpo-50x3/raw.jsonl \
  --attempts-per-task 3 --bootstrap-samples 10000 --seed 2026 \
  --baseline-label SFT --candidate-label Terminal-GRPO \
  --output outputs/eval/sft-vs-terminal-50x3/comparison.json \
  --markdown-output outputs/eval/sft-vs-terminal-50x3/report.md \
  --csv-output outputs/eval/sft-vs-terminal-50x3/report.csv
```

`scripts/render_statistical_report.py` can re-render an existing comparison JSON.
Final-200 remains frozen and deterministic; method selection uses only the 50-task
GRPO validation set.
