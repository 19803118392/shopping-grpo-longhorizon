# GRPO task generations

## Current: Reward v3 / fresh-v1

当前正式世代统一使用 `grpo_reward_v3_fresh_v1_*`：

- `grpo_reward_v3_fresh_v1_probe_pool.jsonl`：全新候选池；
- `grpo_reward_v3_fresh_v1_train.jsonl`：fresh merged SFT probe 后冻结的在线训练 task；
- `grpo_reward_v3_fresh_v1_val.jsonl`：同一 probe pool 中与 train 隔离的验证 task。

候选池排除 Environment v2.1 fresh-v1 的全部 604 个 raw task、两个固定
benchmark、历史 Teacher raw task 和历史 GRPO probe pool。train/validation
按 fresh merged policy 的实际工具步数分布做确定性比例分层，不按 reward 选题。
对应 veRL 输入使用 `data/verl/grpo_reward_v3_fresh_v1_{train,val}.parquet`。

## Archived: GRPO task split v1

这里的清单不是离线 RL 轨迹数据。它们只保存 `task_id`，实际训练时由当前 policy 在线进入 ShopSimulator rollout。

- `grpo_probe_pool_v1.jsonl`：2,000 个候选题。它与已发布的 Teacher raw rollout（757 个 task）及 `shop_benchmark_v2_50` 完全不重叠。
- `grpo_train_v1.jsonl`：1,000 个训练 task。使用冻结的
  `qwen35-2b-shopping-sft-v3-merged` policy，以 `temperature=0`、
  `max_steps=35`、`max_tokens=512` probe 了 1,207 个不同候选 task。
  其中 1,058 条状态有效，实际可用桶为 short 266、medium 53、long 739。
  由于 SFT v3 的 medium 桶远少于最初预估，最终按固定种子 `20260721`
  精确抽取 short 250、medium 50、long 700；149 条基础设施错误没有进入训练集。
- `grpo_val_v1.jsonl`：从候选池剩余 task 中以固定种子 `20260722`
  抽取的 50 个 validation task，与 train、SFT v3 和固定 benchmark 均不重叠。

对应 metadata 记录候选池、probe raw 和 train split 的 SHA-256。veRL 输入位于
`data/verl/grpo_train_v1.parquet` 与 `data/verl/grpo_val_v1.parquet`；
parquet 只含用户可见 instruction 和 task_id，不含隐藏 goal、标准答案或
`reward_detail`。

如果在已发布 raw snapshot 之外又新增并**冻结**了一批 SFT 数据，生成候选池时额外传入 `--exclude-sft path/to/sft.jsonl`。不要把仍在采集中的本地输出写进正式 manifest。

以上 `grpo_*_v1` 资产是旧 SFT v3 / Reward v1-v2 历史世代，只用于复现实验记录，
不得进入 Reward v3 / fresh-v1 正式训练。
