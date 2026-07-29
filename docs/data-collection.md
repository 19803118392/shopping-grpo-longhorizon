# Data collection

## Goal

The SFT stage needs complete examples of a shopping agent using tools correctly:
searching, opening products, inspecting evidence, choosing options and ending
with a valid purchase. The repository contains the accepted action-only
trajectories, not historical failed collection attempts.

## How the dataset was produced

The final collection used ShopSimulator Environment v2.1, Reward v3 and
`deepseek-v4-flash` as the teacher. Seven batches produced 604 raw trajectories.
Each trajectory was replayed and accepted only when the environment returned a
Reward v3 gold purchase.

Collection audit:

| Item | Value |
|---|---:|
| Raw trajectories | 604 |
| Unique task IDs | 604 |
| Accepted gold trajectories | 428 |
| Acceptance rate | 70.9% |
| Mean raw reward | 0.6121 |
| Mean steps | 11.3 |
| Guard violations | 0 |
| HTTP 400 responses | 0 |
| Collection errors | 4 |

The 428 accepted trajectories were split into 379 training and 49 validation
rows. Assistant reasoning was removed; the SFT target contains only the
observable action protocol. This keeps the training contract aligned with what
the environment can verify.

## Frozen deliverables

| File | Rows | SHA-256 |
|---|---:|---|
| `data/sft/train.jsonl` | 379 | `8cd1f72130b3c781d5ffe08fe3e399b2a9e45d204e3f3bd0d8e677d1b51c8ec5` |
| `data/sft/validation.jsonl` | 49 | `f8ae506d0fa9d1526342a9f717da24922c8a55776d076a296698abac4fde05b3` |

The aggregate raw collection had SHA-256
`b1db9e41673d285da7164e8352fa0a702f537157792fa137c94f7cf200435fa1`;
the accepted aggregate had SHA-256
`aab4d81f134dfcd40e67611f5a413142e4825d5cb6ea60b697536aec2c88fab7`.
Raw teacher responses are intentionally not part of the beginner repository.

## What a training row contains

Each JSONL row is a chat trajectory with:

- the shopping instruction;
- assistant tool calls;
- ShopSimulator tool observations;
- the final terminal action;
- metadata tying the row to Environment v2.1 and Reward v3.

During SFT, user and tool tokens are masked. Loss is computed only on assistant
actions. See [SFT](sft.md) for the exact training recipe.
