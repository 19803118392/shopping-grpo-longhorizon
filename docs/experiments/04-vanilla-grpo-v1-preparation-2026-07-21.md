# Vanilla GRPO v1：任务冻结与 veRL 接入准备

> 状态：代码、运行时 smoke 和正式 train/validation task 冻结已完成。正式训练尚未启动。

## 目标

SFT v2 已将 `benchmark_v2_50` 的严格成功率从 Base 的 0% 提升到 12%，但 68% 轨迹仍走满 35 步。下一步先跑最小 Vanilla GRPO：不引入 LLM Judge、Reward Model、PRM-Lite、长度塑形或 Persona，只优化 ShopSimulator 的原生终局 reward。

## 已冻结的边界

| 资产 | 决策 |
|---|---|
| 评测 | `data/benchmarks/shop_benchmark_v2_50.jsonl`，严格取 v1 的前 50 条 |
| RL 候选池 | `data/splits/grpo_probe_pool_v1.jsonl`，2,000 个未见 task |
| 排除集 | 已发布 Teacher raw rollout 全部 757 个 task + benchmark v2_50 |
| 最终训练集 | 1,000 个新 task，不是 1,000 条离线轨迹 |
| 长度分层 | SFT v3 实测后冻结为 short/medium/long = 250/50/700，详见下方记录 |
| reward | 仅正常 `done && over` 时的 ShopSimulator `final_reward`；其他终止均为 0 |

为什么不按 Teacher 轨迹长度分层：GRPO 面对的是 SFT policy 的真实困难度；Teacher 的探索方式与它不同。最初计划 probe 2,000 个候选并按 300/450/250 抽取，但 SFT v3 的真实轨迹分布与旧预估不同，因此以服务器实测结果为准，不制造“表面均衡”的数据。

### SFT v3 正式 probe 与冻结结果（2026-07-25）

按成本停止条件，正式 probe 在 8 个并发 worker 收尾后停于 1,207 条：

- 1,207 个 task_id 全部唯一，且全部来自 2,000 条候选池；
- 1,058 条状态有效：short 266、medium 53、long 739；
- 149 条 `error` 不进入任何训练桶；
- 309 条执行购买，290 条 reward 大于 0，108 条 reward 等于 1；
- probe 中共有 35 个不同 reward 值，证明不同任务存在非全零训练候选；
- 无环境 release 失败。

旧的 medium=450 配额在该 policy 下客观不可满足。最终不按 reward 选题，只通过
现有分层选择器和固定种子 `20260721` 抽取 short 250、medium 50、long 700，
合计 1,000 个训练 task。validation 使用种子 `20260722` 从剩余候选中抽取
50 个 task。train、validation、SFT v3 的 480 个 task 以及固定 benchmark 之间
均无 task_id 重叠。

## SFT → GRPO 的权重边界

保留原始 Base 与 SFT adapter 不动。先将 SFT LoRA 合并到一个新目录，再以这个 merged checkpoint 作为 GRPO 初始策略，并挂载一枚新的 GRPO LoRA。这样 SFT 与 RL 的贡献可分别保存、回滚和比较。

```bash
PYTHONPATH=src python3 scripts/merge_lora_adapter.py \
  --base-model /path/to/Qwen3.5-2B \
  --adapter checkpoints/qwen35-2b-shopping-lora-v2 \
  --output checkpoints/qwen35-2b-shopping-sft-v2-merged \
  --bf16
```

## veRL 最小适配层

`src/shopping_grpo/verl_adapter/` 只做四件事：

1. `ShoppingToolAgentLoop.run` 对每条 rollout 调 `reset(task_id)`，取得独占 HTTP 环境；
2. `Tool` 复用仓库唯一的 `SHOP_TOOL_SCHEMAS`、`tool_call_to_action` 和动作守卫；
3. AgentLoop 只把环境原生终局 reward 写入 veRL 输出；
4. AgentLoop 在 `finally` 中调用 `release()`；释放失败会中止训练并保留租约诊断，避免静默耗尽 env slot。

运行时固定为 pip `veRL 0.8.0`。它已经内置 Qwen3.5 的 `qwen3_coder` parser，因此删除项目重复 parser；它也不再提供 reference fork 的 `verl.interactions`，环境生命周期由项目 AgentLoop 直接管理。完整版本矩阵和服务器安装步骤见 `docs/grpo-runtime-setup.md`。

veRL 的 prompt 必须在训练开始前准备成 parquet。`task_id` 本身不含用户需求；`scripts/prepare_verl_grpo_dataset.py` 会先 reset 一次，只提取用户可见 instruction 写入 prompt，绝不写入 goal、标准答案或 reward_detail。训练时 AgentLoop 会再次 reset 同一 task，保证每个 group sample 独占环境。

## 服务器执行顺序

1. 用 merged SFT v3 模型 probe 候选池，沿用 `evaluate_shop_benchmark.py`、`temperature=0`、`max_steps=35`、`max_tokens=512`，保存 raw probe；
2. 分层冻结 1,000 个 task：

```bash
PYTHONPATH=src python3 scripts/prepare_grpo_tasks.py select \
  --probes outputs/grpo_probe_sft_v3/raw.jsonl \
  --short 250 \
  --medium 50 \
  --long 700 \
  --seed 20260721
```

3. 启动 ShopSimulator 后转 parquet：

```bash
PYTHONPATH=src python3 scripts/prepare_verl_grpo_dataset.py \
  --tasks data/splits/grpo_train_v1.jsonl \
  --output data/verl/grpo_train_v1.parquet \
  --base-url http://127.0.0.1:5700
```

4. 由候选池剩余 task 冻结 validation，避免拿最终 benchmark 调参：

```bash
PYTHONPATH=src python3 scripts/prepare_grpo_tasks.py validation
```

5. 生成 train/val parquet 后执行 `scripts/run_vanilla_grpo.sh`。运行时严格执行 35 个工具步；`max_assistant_turns=40` 只是防御性外层上限，避免第 35 个合法调用被 veRL 的“先递增再判断”逻辑提前丢弃。连续三次动作守卫拒绝、工具异常或 35 步未完成都会立即以 0 reward 终止。

## 仍待服务器验证

- 固定的 veRL 0.8.0 / vLLM 0.25.1 / Transformers 5.11.0 能否在目标 GPU 上完成一次 Qwen3.5 policy update；项目预检会在加载权重前校验版本、导入来源、AgentLoop API 和 veRL 内置 `qwen3_coder` parser；
- 组内 reward 方差、全 0 group 比例与单 GPU 可承受的 rollout 并发；
- 合并 checkpoint 后的固定 benchmark v2_50 复测，应与 adapter 推理一致。
