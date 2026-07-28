# ShopSimulator benchmark v1

`shop_benchmark_v1.jsonl` 固定包含 200 个单轮 ShopSimulator `task_id`，用于公平比较 Base、SFT 和未来 GRPO 模型。

- 随机种子：`20260720`
- SFT 排除集：当前 `outputs/flash_accepted_500_parallel/sft.jsonl` 中的 380 个 task
- 推理协议：temperature=0、max_steps=35、每 task 一次 rollout
- 该 v1 文件属于 Reward v2 历史归档；其中的 `r_type`、`r_att`、`r_option`、`r_price` 口径不适用于当前 Environment v2.1。
- 当前及后续正式实验统一使用 Environment v2.1 / Reward v3；v1 结果不得与 Reward v3 结果直接混报。
- 清单 SHA-256：`9905ab9f4b8d9bbbc44adfd8cc4de2bce2797a63366dc94950015a5eed86655b`

不要把该清单中的 task 用于后续 SFT 或 GRPO 训练采集。若需要新增 benchmark，创建新的版本文件，不能静默改写 v1。

## benchmark v2_50（当前主指标）

`shop_benchmark_v2_50.jsonl` 是 v1 的**有序前 50 条**，不是重新随机抽样。这样已完成的 Base/SFT 评测可以直接缩小到成本可控的共同分母；其父清单 hash、选择方式和协议见同名 metadata。

- 训练、GRPO probe、GRPO online rollout 都必须排除这 50 个 task；
- v1 的 200 条历史结果保留作归档，不再与 v2_50 的指标直接混算；
- 若后续扩充 SFT 数据时出现与 v2_50 的 task 重叠，必须在新的 SFT 训练输入中排除这些 task，或新建 benchmark 后重跑 Base/SFT；不能拿已见题继续报告 v2_50。

### Reward v3 汇总标准

`shop_benchmark_v2_50` 及后续 benchmark 的汇总必须声明 `reward_contract: shopsimulator-reward-v3`，并从终局 `terminal_result.reward_detail` 读取以下字段：

- 严格成功（主指标）：轨迹正常结束，`reward_version == "shopsimulator-reward-v3"`，且 `reward_type == "gold_purchase"`、`reward_valid == true`、`purchase_success == true`、`termination_reason == "gold_purchase"`。
- `gold_purchase_rate`：严格成功数除以固定 benchmark 全部 task 数，未完成或缺失 task 计入分母。
- `purchase_success_rate`：`purchase_success == true` 的比例；它可以包含合法替代购买，不替代严格成功率。
- `reward_type_rates`：按 `gold_purchase`、`partial_alternative_purchase`、`valid_alternative_purchase`、`wrong_purchase`、`repeat_loop`、`max_steps`、`reward_unverifiable` 等 Reward v3 类型统计。
- `mean_final_reward` / `mean_terminal_utility`：全 benchmark 的终局奖励平均值，允许为负；同时记录 `total_final_reward`。
- `mean_weighted_score`：Reward v3 约束匹配分的平均值，只作为诊断指标，不能当作购买成功率。

禁止再用缺失于 Reward v3 的 `r_type`、`r_att`、`r_option`、`r_price` 计算正式 benchmark 主指标。
