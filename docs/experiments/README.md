# 实验记录目录

这里存放按阶段拆分的实验记录。它们不是宣传材料：失败、无效结果和改动原因同样需要留下。主叙事见 [项目实验札记](../project-journal.md)。

## 已完成或正在进行

| 编号 | 记录 | 状态 | 说明 |
|---|---|---|---|
| 03 | [SFT v2：显存实验与 Action-only 决策](03-sft-v2-memory-and-action-only-2026-07-20.md) | 已完成 | Liger + SDPA + Action-only；benchmark v2_50 严格成功率 12% |
| 07 | [SFT v3：加入 100 条短成功轨迹后的重新冷启动](07-sft-v3-short-successes-2026-07-24.md) | 已完成训练、合并与 50 条评测 | 480 条 Action-only；购买率 28%，严格成功率 10%，记录 loss-only eval OOM 修复与早停 |
| 08 | [GRPO A0 五步稳定性与 actor 长序列 OOM](08-grpo-a0-memory-stability-2026-07-25.md) | 修复后 5-step 已通过 | 强制三条 log-prob/actor 路径 micro-batch=1，将 response/总序列预算降到 20K/24K |
| 09 | [GRPO A1 bounded dynamic sampling 五步稳定性](09-grpo-a1-dynamic-sampling-stability-2026-07-25.md) | 已通过 5 次非零更新 | 过滤常量/无 semantic signal group，3 次安全跳过，总计 168 条 rollout |
| 10 | [Shopping Agent 确定性上下文窗口](10-agent-context-window-2026-07-25.md) | 实现与固定 50 条复评完成 | HTTP 400 从 29 降为 0；严格成功仍为相同的 5/50 |

## 尚未启动：先放模板，禁止预填结果

| 编号 | 模板 | 什么时候使用 |
|---|---|---|
| 02 | [SFT 实验记录模板](archived/sft-experiment-template.md) | 已归档的空白模板 |
| 04 | [Vanilla GRPO 实验记录模板](archived/grpo-experiment-template.md) | 已归档的空白模板 |

## 已归档

以下记录保留在 [archived/](archived/) 中，仅供历史追溯：

- [实验 00：单轮轨迹采集与确定性验收](archived/00-data-collection-2026-07-20.md)
- [实验 01：Qwen3.5-2B Instruct 零样本基线](archived/01-qwen35-2b-instruct-baseline-2026-07-20.md)
- [实验 02：LoRA SFT v1 首轮冷启动](archived/02-sft-v1-pitfalls-2026-07-20.md)
- [实验 04：Vanilla GRPO v1 准备](archived/04-vanilla-grpo-v1-preparation-2026-07-21.md)
- [实验 05：GRPO 运行时与任务适配踩坑](archived/05-grpo-runtime-and-task-adaptation-pitfalls-2026-07-23.md)
- [实验 06：veRL 0.8 有限 reward-group 动态采样](archived/06-verl-080-bounded-dynamic-sampling-2026-07-23.md)

## 记录规则

1. 一条记录对应一个可以复现的实验问题，而不是一次随手运行。
2. 开头必须写清 code commit、模型/数据版本、环境和输出目录；密钥绝不写入文档。
3. 结果必须附固定 benchmark 清单和完整命令。运行失败也要记录失败位置与可复现证据。
4. “观察”与“结论”分开写：前者是日志/指标，后者是据此做出的下一步决定。
5. 还没有运行的段落填 `待运行`，不使用预计数值或“应该会提升”代替结果。
