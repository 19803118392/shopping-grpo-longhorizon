# 实验 09：A1 bounded dynamic sampling 五步稳定性

## 目标

在 A0 已通过 5-step 显存稳定性测试后，验证 A1 是否能在不改变模型、数据和长度
预算的情况下：

- 过滤无 semantic signal 或组内 reward 完全相同的 group；
- 每个采样窗口最多生成 3 批；
- 采样不足时跳过而不是异常退出或执行假更新；
- 后续窗口能够继续采样并恢复真实 optimizer update；
- 连续完成 5 次非零梯度更新。

## 配置

- 代码：`1f0ee56b0118b75881d1459e67f396ab8c3aa57f`
- 模型：`checkpoints/qwen35-2b-shopping-sft-v3-merged`
- 数据：`data/verl/grpo_train_v1.parquet`
- 模式：`scripts/run_vanilla_grpo.sh a1`
- `train_batch_size=2`
- `rollout.n=4`
- `shopping_dynamic_sampling.max_num_gen_batches=3`
- `shopping_dynamic_sampling.max_consecutive_skipped_updates=10`
- response/总序列预算：20,480/24,576
- actor、rollout log-prob、Reference 均固定 micro-batch 1
- 输出：`outputs/grpo_v1_checks/a1_stability_5step_memory_safe_20260725/`
- W&B run：`pbhl8zhk`

Preflight 同时确认 veRL 0.8.0 补丁 marker、bounded sampling 配置、
rollout-log-prob bypass 和三条固定 micro-batch 路径。

## 动态采样行为

训练共经历 8 个采样窗口：

- 5 个窗口凑齐 2 个有效 group，执行 optimizer update；
- 3 个窗口在 3 批后只有 1 个有效 group，安全跳过；
- 最大连续跳过为 3，没有触及硬上限 10；
- 跳过时 `optimizer_updated=0`，`global_step` 不增加；
- 下一次凑齐 group 后连续跳过计数恢复为 0。

日志同时验证了两类过滤：

1. semantic reward 全零的 group 被标记为 `no_semantic_signal`；
2. 四条 semantic reward 都为 `0.066666...` 的正 reward group 被标记为
   `constant_reward`。

因此过滤依据不是 shaped reward 中行为惩罚造成的微小差异。采样不足时，已找到但
不足两个的单个有效 group 也不会被拿去执行小批次假更新。

## 五次真实更新

| step | generation batches | generated trajectories | semantic reward mean | actor/loss | grad norm | PPO KL | max response |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 24 | 0.055208 | 0.035485 | 0.022653 | 0.005571 | 19,797 |
| 2 | 2 | 16 | 0.641667 | 0.187974 | 0.175333 | 0.005831 | 20,376 |
| 3 | 2 | 16 | 0.650000 | 0.268435 | 0.097773 | 0.003475 | 19,865 |
| 4 | 2 | 16 | 0.036310 | 0.006681 | 0.008002 | 0.003872 | 20,254 |
| 5 | 3 | 24 | 0.006696 | 0.012995 | 0.005600 | 0.004201 | 20,432 |

五步均为有限且非零的 loss/grad norm，并记录
`training/optimizer_updated=1`；最终 `global_step=5`。

## 成本和资源

成功更新窗口合计：

- 12 个 generation batch；
- 24 个生成 group；
- 14 个 group 被过滤；
- 10 个 group、40 条轨迹进入训练；
- 96 条 rollout 被生成。

三个跳过窗口额外生成：

- 9 个 generation batch；
- 18 个 group；
- 72 条 rollout。

本次完整 smoke 因此总计 **21 个 generation batch、168 条 rollout**。这里的
`3 × 8 = 24` 是单个采样窗口的上限，不是一个 global step 在包含跳过重试后的总
上限。连续跳过上限 10 才是跨窗口的最终成本边界。

- 总耗时：983 秒（16 分 23 秒）
- 训练进度耗时：约 13 分 43 秒
- GPU 两秒轮询峰值：73,249 MiB
- actor PyTorch 峰值 allocated/reserved：51.982/69.213 GiB
- 顶层退出码：0

A1 虽然生成轨迹更多，但过滤后的 actor batch 仍固定为 8 条，所以显存没有高于 A0；
总耗时也没有增加，因为 actor forward/backward 是主要成本。

## 退出与环境回收

日志没有 CUDA OOM、HTTP 400、release error 或 infrastructure-invalid trajectory。
退出后：

- GPU 恢复为 3 MiB；
- Ray/vLLM 无残留；
- 同时租用 ShopSimulator 8 个 slot 得到 `0..7`；
- 8/8 slot 均成功释放。

训练达到 100% 后，Ray 清理阶段仍打印已知的 DataLoader worker killed traceback，
W&B 的 `atexit` 回调也打印 closed transport traceback。顶层退出码为 0，五步指标
和最终 step 已写出；这些 traceback 不影响 optimizer update 或环境回收，但正式训练
日志解析应把它们与训练期异常区分。

## 结论

A1 bounded dynamic sampling 的 5-step 稳定性门槛通过：过滤、有限补采、安全跳过、
恢复更新、非零梯度、显存稳定性和环境回收均有实际日志证据。

正式扩大时应启用 checkpoint；`resume_mode=auto` 已配置，但本 smoke 使用
`save_freq=-1`，未验证 checkpoint 写入与恢复。建议先以 50 step、`save_freq=5`
运行，并在 W&B 重点观察：

- `group/effective_ratio`
- `group/no_semantic_signal_ratio`
- `shopping_dynamic_sampling/skipped_updates_total`
- `shopping_dynamic_sampling/consecutive_skips`
- `rollout/generated_total`
- `actor/loss`、`actor/grad_norm`、`actor/ppo_kl`
- `response_length/max`

