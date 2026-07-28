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

## Reward v3 final test

`shop_benchmark_reward_v3_final_200.jsonl` 是 Reward v3 / fresh-v1 世代的盲测集。
它包含 200 个 task_id，不是训练 parquet，也不进入 GRPO validation。

相邻的 `shop_benchmark_reward_v3_final_200.guard.json` 是独立保护契约，冻结了
asset ID、split role、task/metadata SHA256、task 数量和 metadata 必需字段。离线
评测 wheel 内置该契约的最小副本和 200 个 task ID，并进一步比较输入 artifact 的
task ID；因此普通 wheel 安装不依赖仓库数据目录，复制、重命名或重新格式化 final
文件也不能绕过保护。正式评测前不得传 `--allow-blind-final`。

生成时排除了本机所有带 task_id 的历史输出、fresh SFT、完整 Reward v3 probe、
历史 GRPO split/probe 和旧 benchmark。完整排除 task 快照保存在相邻
`shop_benchmark_reward_v3_final_200.exclusions.jsonl`，来源文件及 SHA256 保存在
metadata。冻结统计为：全集 23,421 条、排除 6,720 条、可选 16,701 条；按种子
`20260728` 确定性随机无放回抽取 200 条。

该 final test 冻结后不得通过 SFT/GRPO rollout 再做选题或切分。GRPO checkpoint
只能由 validation 选择；最终选定 checkpoint 后，fresh merged SFT 与 GRPO 在同一
200 条上各运行一次固定 temperature=0 的配对评测。

- final test SHA256：
  `2c4ff070e13ddc30796d38e85170210e7d3c211992425a62090f2419fe8e0208`
- exclusions SHA256：
  `00030db1042a75c7c988c878bd957cfe39478e369beec1e9e41a03c52b2e9e88`
- metadata SHA256：
  `42d7adc26ed48430da3def670453f44ee8a69f8ac7bbe5729a5cefa7bbd47b1b`
