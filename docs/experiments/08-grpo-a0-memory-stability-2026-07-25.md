# 实验 08：A0 五步稳定性与 actor 长序列 OOM

## 问题

在 RTX PRO 6000 Blackwell 96GB 上，A0 原生 reward 配置能否连续完成 5 次
Vanilla GRPO 更新？

## 首次运行

- 代码：`7d8b72c16372a430adcbbc6c0cd97590fa6e6487`
- 模型：`checkpoints/qwen35-2b-shopping-sft-v3-merged`
- 数据：`data/verl/grpo_train_v1.parquet`
- 命令：A0，`total_training_steps=5`，关闭训练前验证与 checkpoint
- 输出：`outputs/grpo_v1_checks/a0_stability_5step_20260725_v2/`
- W&B run：`kppv0qnt`

前四步完成，指标如下：

| step | actor/loss | actor/grad_norm | actor/ppo_kl | native reward |
|---:|---:|---:|---:|---:|
| 1 | 0.243791 | 0.306550 | 0.004960 | 0.045139 |
| 2 | 0 | 0 | 0.003927 | 0 |
| 3 | 0.051550 | 0.106998 | 0.005694 | 0.229167 |
| 4 | 0.034914 | 0.085604 | 0.004335 | 0.035714 |

第 5 步 rollout 已完成，但在
`update_actor → logprobs_from_logits_v2 → F.log_softmax` 处 OOM。traceback 显示
PyTorch 已分配 91.29 GiB，GPU 只剩 3.38 MiB。GPU 轮询峰值为 94,529 MiB。
没有 HTTP 400 或环境错误；退出后 8 个 ShopSimulator slot 均能重新获取并释放。

## 原因

旧配置允许：

```text
max_response_length = 24576
max_prompt_length = 4096
max_model_len = 28672
```

Qwen3.5 的词表为 248,320。长序列 actor forward 必须产生完整词表 logits；梯度、
激活和 log-softmax 临时空间叠加后，最坏 batch 会超过 96GB 卡实际可用的
94.97 GiB。

虽然配置写了 `ppo_micro_batch_size_per_gpu=1`，但同时启用
`use_dynamic_bsz=true`。veRL 会按照 token workload 重新组合 micro-batch，固定
micro-batch 值并不是实际硬边界。`ppo_max_token_len_per_gpu` 用于决定拆分数量，
不是超长轨迹过滤器。

## 第一阶段修复

只调整内存调度，不改变 reward、GRPO loss、模型、35-step 环境上限、group size 或
动态采样语义：

```text
max_response_length = 20480
max_prompt_length + max_response_length = 24576
actor.use_dynamic_bsz = false
actor.ppo_micro_batch_size_per_gpu = 1
ref.log_prob_use_dynamic_bsz = false
ref.log_prob_micro_batch_size_per_gpu = 1
rollout.log_prob_use_dynamic_bsz = false
rollout.log_prob_micro_batch_size_per_gpu = 1
```

Preflight 会拒绝恢复旧的 24K response 或重新打开 actor/reference dynamic batch。
这样可以在实际训练前发现不安全覆盖。

这一步验证通过前，不增加 post-rollout 长度过滤。若仍出现最坏长度 OOM，下一阶段才
在 reward 之后、Reference/advantage/update_actor 之前按实际 token 数丢弃整个 uid
group，并复用有限补采/跳过控制流；不能只删除 group 中单条 rollout。

## 断点恢复

veRL 默认 `resume_mode=auto`，checkpoint 包含 actor、optimizer、LR scheduler、RNG、
global step 和 StatefulDataLoader 状态。本次失败运行使用 `save_freq=-1`，因此前四次
内存中更新没有 checkpoint，不能恢复。稳定性验证通过后，初期正式训练应使用较短的
保存间隔，例如 `save_freq=5`，并保留最近两个 checkpoint。

## 复测结果

第一阶段修复在提交 `700a72b43fc88bf42ba07f02e925907079b025f5` 上完成
A0 5-step：

- 输出：`outputs/grpo_v1_checks/a0_stability_5step_memory_safe_20260725_v3/`
- W&B run：`pxltqpwd`
- 退出码：0
- 总耗时：1,073 秒（17 分 53 秒，训练进度本身约 15 分 6 秒）
- GPU 两秒轮询峰值：73,421 MiB；旧运行峰值为 94,529 MiB，下降 21,108 MiB
- PyTorch actor 峰值 allocated/reserved：52.434/69.383 GiB

| step | native reward | all-equal group ratio | actor/loss | grad norm | PPO KL | max response | step 秒 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.031250 | 0.5 | 0.067198 | 0.177972 | 0.006036 | 20,134 | 151.6 |
| 2 | 0 | 1.0 | 0 | 0 | 0.003764 | 20,480 | 191.1 |
| 3 | 0.114583 | 0.5 | 0.162117 | 0.087606 | 0.006901 | 20,075 | 186.4 |
| 4 | 0.053571 | 0.5 | 0.072168 | 0.115658 | 0.004207 | 19,312 | 167.5 |
| 5 | 0 | 1.0 | 0 | 0 | 0.005668 | 20,396 | 209.3 |

五步均记录 `training/optimizer_updated=1`，最终进入 `global_step=5`。第 5 步的
8 条 response 都在 19,104–20,396 token 之间，仍能完成 actor backward，说明本次
修复覆盖了此前失败的长序列场景。第 2、5 步的两组 native reward 全相同，因此
GRPO advantage、loss 和 grad norm 为 0；这是 A0 不过滤常量组的预期行为，不是数值
或执行错误。

日志没有 CUDA OOM、HTTP 400、基础设施无效轨迹或 release error。进程退出后 GPU
恢复为 3 MiB，Ray/vLLM 无残留；再次同时租用 8 个 ShopSimulator slot 得到
`0..7`，并成功释放 8/8。退出阶段 W&B 在 Ray actor 的 `atexit` 回调中打印了一段
closed Unix transport traceback，但顶层进程退出码仍为 0，五步指标已经同步，且不
影响训练或环境回收。

启动复测时还暴露出一项 veRL 配置约束：关闭 dynamic batch 后，
`rollout.calculate_log_probs=true` 的路径也必须显式配置
`rollout.log_prob_micro_batch_size_per_gpu=1`。首次尝试在系统 Python preflight
停止，第二次在 veRL 配置校验停止，两次都没有加载模型；补齐并测试后才执行上述唯一
一次 GPU 复测。失败启动日志分别保存在同名无后缀目录和 `_v2` 目录。

## 结论与后续边界

A0 五步稳定性门槛通过。当前没有证据需要再增加 post-rollout 长度过滤；先保留
20K response、24K 总序列和固定 micro-batch=1 作为正式实验边界。A0 的 5 步中有
2 步只产生常量 reward group，说明训练信号利用率仍是独立问题，应由已经实现的 A1
有限动态采样验证，不能通过重新放宽长度预算来解决。
