# Agent 实验契约

## 当前奖励标准

本仓库当前及后续正式后训练实验统一以 `Environment v2.1 / Reward v3` 为起点。Reward v2、旧 SFT、旧 GRPO 和旧 benchmark 只可作为历史归档，不能作为当前训练输入、恢复点或正式对比基线。

Reward v3 的正式严格成功定义是：

```text
reward_version == "shopsimulator-reward-v3"
reward_type == "gold_purchase"
reward_valid == true
purchase_success == true
termination_reason == "gold_purchase"
trajectory.status == "done"
trajectory.done == true
terminal_result.done == true
terminal_result.over == true
```

`gold_purchase_rate` 以固定 benchmark 的全部 task 为分母；缺失、异常、未完成 task 都计入分母。`purchase_success_rate` 可以包含合法替代购买，因此不能替代严格成功率。

正式汇总必须报告：

- `reward_contract`，固定为 `shopsimulator-reward-v3`；
- `gold_purchase_rate` 和严格成功 task IDs；
- `purchase_success_rate`；
- `reward_type_counts` / `reward_type_rates`；
- `total_final_reward`、`mean_final_reward`、`mean_terminal_utility`；
- `mean_weighted_score` 作为诊断指标；
- done rate、平均步数、guard rejection、context projection 和磁盘/日志信息。

当前正式 benchmark 禁止使用旧 Reward v2 的 `r_type`、`r_att`、`r_option`、`r_price` 作为主指标或 reward component 汇总。若发现这类字段出现在当前 benchmark 汇总中，应立即标记为口径错误，不得报告为正式结果。

## 代码与数据边界

- fresh-v1 Teacher/SFT 数据生成基线冻结于 commit
  `cb61a178f0cf5bcac4b0a1d5b475d0b882bd1634`。
- Reward v3 GRPO 准备与运行代码必须匹配
  `data/manifests/environment_v2_1_reward_v3_fresh_v1.json` 中记录的
  `shopping_grpo_commit`；不得仅凭旧 `cb61a17` 假定运行代码未变化。
- 当前 base model 必须使用固定本地 Qwen3.5-2B snapshot，不得联网解析新的 Hugging Face revision。
- 当前 SFT 只允许使用 fresh-v1 train/val 数据；历史数据和旧 adapter 只归档。
- benchmark task 不得进入 SFT 或 GRPO 训练数据。
- 修改 benchmark 汇总口径时必须同步修改测试、评测协议输出和本文件；不得只修一个 JSON 字段。

### fresh-v1 SFT 冻结哈希

- train：379 条，SHA256
  `8cd1f72130b3c781d5ffe08fe3e399b2a9e45d204e3f3bd0d8e677d1b51c8ec5`；
- validation：49 条，SHA256
  `f8ae506d0fa9d1526342a9f717da24922c8a55776d076a296698abac4fde05b3`。

旧契约中的 train SHA 少了 `a9`、只有 62 位，已经废止。24,576 token
模板预检的实际加载结果是 train 376/379、validation 47/49；“丢弃 3 条”
只指 train，不是 train+validation 总数。

## Reward v3 GRPO 实验世代

当前 GRPO 唯一正式世代是 `reward_v3_fresh_v1`：

- 初始 policy：
  `/root/autodl-tmp/checkpoints/qwen35-2b-sft-v1-fresh-merged`；
- candidate：
  `data/splits/grpo_reward_v3_fresh_v1_probe_pool.jsonl`；
- probe：
  `outputs/grpo_reward_v3_fresh_v1_probe/raw.jsonl`；
- train/validation：
  `data/splits/grpo_reward_v3_fresh_v1_{train,val}.jsonl`；
- veRL parquet：
  `data/verl/grpo_reward_v3_fresh_v1_{train,val}.parquet`；
- launcher：`scripts/run_grpo_reward_v3_fresh_v1.sh`；
- environment manifest：
  `data/manifests/environment_v2_1_reward_v3_fresh_v1.json`。

候选池必须排除 fresh-v1 全部 604 个 raw task、固定 benchmark、历史 Teacher
task 和历史 GRPO probe task。probe 必须由 fresh merged SFT 在 Environment
v2.1 / Reward v3 上重新执行；不得用旧 policy 的轨迹长度替代。train/validation
只按 fresh policy 的实际执行步数做确定性比例分层，不按 reward 或成功与否挑题。

`scripts/run_vanilla_grpo.sh`、`grpo_*_v1.jsonl` 和
`data/verl/grpo_*_v1.parquet` 已归入 legacy v1/v2 世代，默认禁止启动或作为本世代
输入。

## 运行规则

- 未经用户明确授权，不启动正式训练、GRPO、benchmark 重跑或模型合并。
- 只读诊断不得修改模型、数据、依赖或运行中的实验。
- 发现 OOM、NaN/Inf、CUDA 错误、模板失败、checkpoint 写入失败或 reward contract 不匹配时，停止并报告证据，不自动修复、降 batch 或重跑。
- Reward v3 probe 的任意 `status=error` 都是停止条件；不得静默过滤后切分。Environment v2.1 slot 只由 trajectory owner 显式释放，服务端 terminal 不自动回收，避免并发 double-release 导致状态串线。
