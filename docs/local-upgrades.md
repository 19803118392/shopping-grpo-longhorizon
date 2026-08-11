# Statistical evaluation upgrade

## Project basis and contribution boundary

This repository uses Qwen as the policy model, veRL as the GRPO framework, and
ShopSimulator as the stateful shopping environment. It does not claim original
authorship of those foundations. The project-owned upgrade line starts from the
fixed repository baseline `d99a0ac`; reviewers can inspect committed work with:

```bash
git log --oneline d99a0ac..HEAD
git diff --stat d99a0ac..HEAD
git diff d99a0ac..HEAD -- src scripts tests experiments docs
```

That range covers the evidence-bearing work below. Line counts are provenance
helpers, not resume achievements; the claims remain tied to code, tests, raw
rollouts, and machine-readable reports.

| Upgrade area | Implementation evidence | Result evidence |
|---|---|---|
| Paired repeated evaluation | `src/shopping_grpo/evaluation/statistics.py`, `scripts/compare_repeated_evaluations.py` | `experiments/validation-50x3/` |
| Frozen confirmation | `scripts/freeze_final_candidate.py`, `scripts/evaluate_shop_benchmark.py` | `experiments/final-200/` |
| Seed replay and fixed denominator | `scripts/verify_seed_replay.py`, `src/shopping_grpo/evaluation/rollout.py` | Validation run config and reports |
| GRPO resume and adaptive state | `scripts/train_grpo.py`, dynamic-sampling patch and tests | Resume smoke in `experiments/single-seed-42/` |
| Reward signal audit | `scripts/audit_reward_v4.py`, GRPO reward adapter | 700-trajectory audit in `experiments/single-seed-42/summary.json` |
| Reward-varying active set | `scripts/prepare_grpo_active_set.py` and its tests | Ten-update pilot and hashed artifact ledger in `experiments/single-seed-42/` |

## Claim-to-evidence contract

| Claim | Evidence | Safe wording | Unsupported wording |
|---|---|---|---|
| SFT creates the basic tool-use policy | Pipeline Base/SFT summaries: 0.0% vs 60.5% strict success | “In the initial deterministic pipeline benchmark, SFT reached 60.5%.” | “SFT is universally 60.5 points better.” |
| Terminal-GRPO has a positive development result | Validation-50×3: +8.0 points, CI [+2.0,+14.7] | “GRPO produced a positive development-set signal.” | “GRPO reliably improves the final policy by 8 points.” |
| The development effect did not reproduce | Frozen Final-200: +1.5 points, CI [-2.0,+5.0] | “The frozen stage did not establish a reliable gain.” | “Final-200 proves GRPO is better.” |
| More-SFT is a competitive control | Post-hoc Final-200×3: +2.5 points over GRPO, CI crosses zero | “More-SFT is a strong practical control.” | “More-SFT is proven superior to GRPO.” |
| Active-set pilot produces usable updates | Pilot effective-group ratio 71.7%, 10/10 updates applied | “The active-set run observed fewer low-information batches; causal attribution is untested.” | “Active sampling improves Agent capability or sampler efficiency.” |
| Reward v4 passed integration but not efficacy | One differentiating trajectory among 700; two five-update smokes | “Reward v4 was audited and stopped before a costly comparison.” | “Reward v4 improves training.” |

The statistical-evaluation stage compares SFT and GRPO as a paired repeated-
measures experiment. It does not treat a one-off percentage difference as
evidence of an algorithmic gain.

The first completed 50×3 run reports SFT 66.7% versus Terminal-GRPO 74.7%, a
paired difference of +8.0 percentage points with bootstrap 95% CI
[+2.0, +14.7] and exact McNemar p=0.0118. See the
[experiment card](../experiments/validation-50x3/README.md); this is a validation
result, not Final-200.

The later frozen Final-200 run reports 57.0% versus 58.5%, a paired difference
of +1.5 points with bootstrap 95% CI [-2.0, +5.0] and McNemar p=0.6072. It does
not establish a reliable final gain. The project retains both results and did
not tune the frozen-stage decision after that run. The benchmark had appeared
in the earlier pipeline stage, so it is training-held-out but not a pristine
first-look test set. See the
[final experiment card](../experiments/final-200/README.md).

Two later controls are explicitly post-selection evidence:

- Post-hoc Final-200×3 measured Terminal-GRPO-30 at 63.2% and More-SFT at
  65.7%; More-SFT minus GRPO was +2.5 points with CI [-1.2,+6.2] and
  `p=0.1756`.
- An active-set ten-update pilot measured More-SFT at 70.0% and Active-GRPO-10
  at 71.3% on Validation-50×3; the +1.3-point difference had CI [-5.3,+8.0]
  and `p=0.8238`.

Neither interval excludes zero. The first reuses Final-200 after earlier
results were known, and the second is development-set method selection. They
are retained as controls and negative evidence, not promoted to final gains.

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
The frozen-stage Final-200 decision remains closed. Later repeated use of the
same tasks is labeled post-hoc and cannot be presented as another frozen test;
method selection continues to use only the 50-task GRPO validation set.
