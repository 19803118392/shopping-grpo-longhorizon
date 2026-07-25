# Shopping GRPO v1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task.

**Goal:** 在现有 veRL 0.8.0 与 ShopSimulator 多步环境链路上，实现一版能够产生有效学习信号、成本受控、可与原始基线公平对比的 Shopping GRPO。

**Architecture:** 保留现有 AgentLoop、工具 schema、动作守卫、veRL 官方训练入口和本地动态采样补丁。只修改终局奖励、有效组筛选、优势归一化配置和必要日志，不引入 Value Model、PRM、LLM Judge 或工具步骤级信用分配。

**Tech Stack:** Python 3.12、PyTorch 2.11、veRL 0.8.0、vLLM 0.25.1、Ray 2.56.1、Qwen3.5-2B、ShopSimulator、LoRA。

---

## 1. 固定范围

训练流程固定为：

```text
Qwen3.5-2B Base
→ Action-only LoRA SFT
→ 合并 SFT 权重
→ 从合并后的 SFT 模型新建 GRPO LoRA
→ veRL 在线调用 ShopSimulator
→ 固定 50 条 benchmark 评测
```

第一版只实现：

1. 约束感知终局奖励；
2. Dr.GRPO 式优势计算；
3. 无效 reward group 动态过滤与补采；
4. Token-Mean Policy Gradient Loss；
5. 固定对称 clip；
6. 最小必要监控；
7. A0、A1 两组实验。

第一版明确不实现：

- Reference Model 与 KL Loss；
- Value Model；
- PRM、LLM Judge、Rubric；
- step-level reward 或 step-level advantage；
- Clip-Higher；
- 新的 Agent 框架；
- 修改 ShopSimulator 原生评分逻辑；
- 增大 group size；
- 并行维护第二套动态采样器。

## 2. 数据与任务隔离

使用以下固定数据：

```text
SFT 初始化模型：
GRPO_MODEL_PATH 指向通过 readiness audit 的最新 merged checkpoint

GRPO 训练任务：
data/splits/grpo_train_v1.jsonl
data/verl/grpo_train_v1.parquet

GRPO 验证任务：
data/splits/grpo_val_v1.jsonl
data/verl/grpo_val_v1.parquet

固定评测：
data/benchmarks/shop_benchmark_v2_50.jsonl
```

当前优先使用已经完成训练、合并和 50 条 benchmark 的
`qwen35-2b-shopping-sft-v3-merged`。后续只有新 checkpoint 通过同一 readiness
audit 后，才能替换 `GRPO_MODEL_PATH`。

若尚未冻结 GRPO train/validation，先指定当前 SFT policy 的 probe 原始轨迹：

```bash
export GRPO_PROBE_RAW=/absolute/path/to/frozen-sft-policy-probes.jsonl

PYTHONPATH=src python3 scripts/prepare_grpo_tasks.py select \
  --probes "$GRPO_PROBE_RAW"

PYTHONPATH=src python3 scripts/prepare_grpo_tasks.py validation
```

随后在 ShopSimulator 已启动时生成 parquet：

```bash
PYTHONPATH=src python3 scripts/prepare_verl_grpo_dataset.py \
  --tasks data/splits/grpo_train_v1.jsonl \
  --output data/verl/grpo_train_v1.parquet \
  --split train \
  --base-url "$SHOPSIM_BASE_URL"

PYTHONPATH=src python3 scripts/prepare_verl_grpo_dataset.py \
  --tasks data/splits/grpo_val_v1.jsonl \
  --output data/verl/grpo_val_v1.parquet \
  --split validation \
  --base-url "$SHOPSIM_BASE_URL"
```

生成后冻结 JSONL、metadata 和 parquet，不得重新抽样或改写 task_id。

任务隔离要求：

- SFT task_id 不进入 GRPO train；
- benchmark 50 条不进入 SFT 或 GRPO train；
- readiness audit、cheap probe 使用过的 task 不进入 benchmark；
- A0 与 A1 使用完全相同的 GRPO train、validation 和 benchmark；
- A0 与 A1 使用相同 seed 和任务顺序。

## 3. 终局奖励

### 3.1 输入字段

只从 ShopSimulator 终局结果读取：

```text
native_reward
reward_detail.r_type
reward_detail.r_att
reward_detail.r_option
reward_detail.r_price
done
over
```

四个 reward component 必须：

- 字段完整；
- 能转换为有限数值；
- 数值位于 `[0, 1]`；
- 只用于训练端计分和日志；
- 不得写入模型可见的 messages、observation 或 tool response。

终局购买轨迹若缺少任一 component，标记为无效轨迹，不使用默认值补齐。

### 3.2 奖励公式

定义：

```text
R_full =
    1，当 r_type = r_att = r_option = r_price = 1
    0，其他情况

R_strict =
    r_type × r_att × r_option × r_price

R_loose =
    ShopSimulator native_reward

R_semantic =
    R_full + 0.5 × R_strict + 0.2 × R_loose
```

设：

```text
T = 当前轨迹已消耗的工具执行步数
H = 35
```

成功效率奖励：

```text
R_efficiency =
    0.05 × R_full × max(0, 1 - T / H)
```

失败超长惩罚：

```text
P_overlong =
    0，T <= 28
    0.05 × (1 - R_full) × (T - 28) / 7，28 < T <= 35
```

`P_overlong` 上限固定为 `0.05`。

未完成惩罚：

```text
P_unfinished =
    0.05，当 termination_reason = assistant_finished_without_environment_done
    0，其他情况
```

重复动作惩罚：

```text
P_repeat =
    0.03 × N_repeat / max(N_action_attempts, 1)
```

最终训练奖励：

```text
R =
    R_semantic
    + R_efficiency
    - P_overlong
    - P_unfinished
    - P_repeat
```

不再额外增加固定 `max_steps` 惩罚，避免和 `P_overlong` 重复扣分。

### 3.3 步数口径

`T` 沿用当前环境的 max-step 计数口径：

- 已执行的环境工具计入；
- `think` 若当前实现占用环境步骤，继续计入；
- 动作守卫拒绝、且未执行到环境的调用不计入 `T`；
- `T` 最大为 35。

不在本阶段改变 max-step 的既有语义。

### 3.4 重复动作口径

每次模型尝试调用环境工具时，生成动作签名：

```text
(
  tool_name,
  canonical_parameters,
  current_observation_fingerprint
)
```

规则：

- 参数使用标准 JSON key 排序后序列化；
- observation fingerprint 使用 Python 标准库稳定哈希输入；
- `think` 不参与重复统计；
- 动作守卫检查前记录尝试，因此被守卫拒绝的重复尝试也能计数；
- 当前签名若在之前 3 次环境动作尝试中出现，`N_repeat += 1`；
- 不增加第三方依赖。

### 3.5 错误分类

以下属于基础设施无效轨迹，不进入训练：

- HTTP 非 2xx；
- ShopSimulator reset/release 失败；
- 环境槽位分配失败；
- tool response 无法解析；
- reward_detail 缺失或非法；
- vLLM/Ray/CUDA 运行时异常。

以下属于模型行为，可作为负样本参与组内比较：

- `max_steps`；
- 未购买便自行结束；
- 连续动作守卫拒绝；
- 非法并行工具调用；
- 购买了部分满足或错误商品。

一组中只要出现基础设施无效轨迹，整组丢弃并补采，不能把基础设施失败当作零奖励训练。

## 4. 优势与 Loss

### 4.1 优势计算

A1 使用：

```text
A_i = R_i - mean(R_group)
```

配置：

```yaml
algorithm:
  norm_adv_by_std_in_grpo: false
```

不引入 Value Model，不修改 advantage estimator 的其他部分。

### 4.2 Token-Mean Loss

配置：

```yaml
actor_rollout_ref:
  actor:
    loss_agg_mode: token-mean
```

参与 loss 的 token：

- assistant 生成的正文；
- assistant 生成的工具调用。

不参与 loss 的 token：

- system；
- user；
- ShopSimulator tool observation；
- padding。

继续使用轨迹级 advantage，并广播到该轨迹全部有效 assistant token。

### 4.3 Clip

第一版固定：

```yaml
clip_ratio_low: 0.20
clip_ratio_high: 0.20
```

不得在 A1 同时启用 Clip-Higher。

### 4.4 KL 与 Reference Model

A0、A1 都使用：

```yaml
actor_rollout_ref:
  actor:
    use_kl_loss: false

algorithm:
  use_kl_in_reward: false
```

保留：

```yaml
actor_rollout_ref:
  rollout:
    calculate_log_probs: true
    rollout_correction:
      bypass_mode: true
      bypass_mode_error_on_nonzero: false
```

不启动 Reference Model，不计算 KL Loss。

## 5. 动态采样

### 5.1 有效组

group size 固定为 `n = 4`。

一组必须同时满足以下条件才进入 policy update：

1. 恰好包含 4 条 rollout；
2. 4 个最终 shaped reward 都是有限数；
3. 不含基础设施无效轨迹；
4. `max(R_group) - min(R_group) > 1e-8`；
5. 至少一条轨迹满足 `R_semantic > 0`。

全零、全成功、其他 reward 完全相同的组均不参与更新。

### 5.2 补采

执行顺序：

```text
生成一批 task groups
→ 运行环境 rollout
→ 计算 shaped reward
→ 过滤无效组
→ 有效组不足时补采新 task
→ 凑够 train_batch_size 个有效组
→ 执行一次 optimizer update
```

固定配置：

```yaml
shopping_dynamic_sampling:
  enable: true
  max_num_gen_batches: 3
  reward_tolerance: 1.0e-8
```

`max_num_gen_batches=3` 包含首次生成。

达到上限仍不足一个完整训练 batch 时：

- 不执行部分 batch 更新；
- 不填充重复 group；
- 立即停止当前训练；
- 输出每种过滤原因和已消耗轨迹数量。

继续复用：

```text
src/shopping_grpo/verl_dynamic_sampling.py
patches/verl-0.8.0-shopping-dynamic-sampling.patch
scripts/apply_verl_dynamic_sampling_patch.py
```

不得另写第二套 trainer 或 sampler。

## 6. 实验配置

### 6.1 共同配置

A0、A1 固定相同：

```text
初始化 checkpoint
GRPO LoRA r/alpha/dropout
train/validation task
task 顺序
seed
group size = 4
learning rate = 1e-6
rollout temperature = 0.7
rollout top_p = 0.9
max environment steps = 35
train batch size
optimizer update 数
50 条 benchmark
```

96 GB GPU 上第一轮不得增加 group size。

### 6.2 A0：最小对照组

```text
Reward：ShopSimulator native reward
KL：关闭
Reference Model：关闭
优势：组内均值中心化 + 标准差归一化
Dynamic Sampling：关闭
Loss：token-mean
Clip：0.20 / 0.20
```

运行覆盖项：

```bash
bash scripts/run_vanilla_grpo.sh a0
```

### 6.3 A1：Shopping GRPO v1

```text
Reward：本文约束感知 shaped reward
KL：关闭
Reference Model：关闭
优势：只做组内均值中心化
Dynamic Sampling：开启
Loss：token-mean
Clip：0.20 / 0.20
```

运行覆盖项：

```bash
bash scripts/run_vanilla_grpo.sh a1
```

正式运行前必须额外传入：

```bash
source .venv-grpo-v080/bin/activate

export GRPO_MODEL_PATH=/root/autodl-tmp/shopping-grpo-longhorizon/checkpoints/qwen35-2b-shopping-sft-v3-merged
export GRPO_TRAIN_FILE=/root/autodl-tmp/shopping-grpo-longhorizon/data/verl/grpo_train_v1.parquet
export GRPO_VAL_FILE=/root/autodl-tmp/shopping-grpo-longhorizon/data/verl/grpo_val_v1.parquet
export GRPO_OUTPUT_DIR=/root/autodl-tmp/shopping-grpo-longhorizon/checkpoints/grpo
export SHOPSIM_BASE_URL=http://127.0.0.1:5700
```

实际 checkpoint 名称若不同，只允许修改 `GRPO_MODEL_PATH`，不得临时改变实验配置。

## 7. 必要监控

### 7.1 Reward

每个训练 step 记录：

```text
reward/full_mean
reward/strict_mean
reward/native_mean
reward/shaped_min
reward/shaped_mean
reward/shaped_max
reward/efficiency_mean
penalty/overlong_mean
penalty/unfinished_mean
penalty/repeat_mean
component/r_type_mean
component/r_att_mean
component/r_option_mean
component/r_price_mean
```

### 7.2 Group

```text
group/generated
group/trained
group/effective_ratio
group/all_equal_ratio
group/all_zero_semantic_ratio
group/all_full_success_ratio
group/infrastructure_invalid
group/resample_batches
rollout/generated_total
```

### 7.3 Policy 与运行时

```text
actor/loss
actor/grad_norm
actor/ppo_kl
actor/clip_fraction
response_length/mean
trajectory/average_steps
trajectory/max_steps_rate
trajectory/done_rate
trajectory/repeat_action_rate
runtime/gpu_peak_memory_mib
runtime/step_seconds
```

第一版不重新开启完整 logits entropy 计算。

veRL 同时启用 `console` 与 `wandb` logger。二者接收完全相同的每步 metrics
字典；本地日志用于完整追溯，W&B online 用于查看 reward、group、policy 和
runtime 曲线。当前 veRL 原生字段中，clip fraction、GPU 显存和 step 耗时分别
记录为 `actor/pg_clipfrac`、`actor/perf/max_memory_allocated_gb` 和
`timing_s/step`。

### 7.4 离线多样性审计

每个 checkpoint 评测完成后，从保存轨迹离线统计：

```text
unique search query 数
unique ASIN 数
unique tool sequence 数
同 task 的工具序列重复率
```

## 8. 实现任务

### Task 1：终局奖励数据闭环

**Files:**

- Modify: `src/shopping_grpo/verl_adapter/runtime.py`
- Modify: `src/shopping_grpo/verl_adapter/tools.py`
- Modify: `src/shopping_grpo/verl_adapter/agent_loop.py`
- Create: `tests/test_shopping_reward.py`
- Modify: `tests/test_verl_adapter.py`

**执行：**

1. 先添加 reward component、公式边界和隐藏字段不泄露测试；
2. 在终局工具结果中保存经过校验的 `reward_detail`；
3. 在 `runtime.py` 实现纯函数式 reward breakdown；
4. `terminal_reward()` 根据 `reward_mode` 选择 native 或 constraint-aware；
5. 把 breakdown 放入 `output.extra_fields["shopping"]`；
6. 不把 reward_detail 写回模型消息；
7. 运行相关单元测试。

**验收：**

- 满分 8 步轨迹奖励为：

```text
1.7 + 0.05 × (1 - 8 / 35)
```

- 满分 35 步轨迹奖励为 `1.7`；
- 未购买便结束、且不超长的轨迹奖励包含 `-0.05`；
- 终局 reward_detail 缺字段时轨迹被标记为 infrastructure invalid；
- 模型可见 messages 中不存在 `reward_detail`、goal、标准答案或 purchase。

### Task 2：重复动作统计

**Files:**

- Modify: `src/shopping_grpo/verl_adapter/runtime.py`
- Modify: `src/shopping_grpo/verl_adapter/tools.py`
- Modify: `tests/test_shopping_reward.py`

**执行：**

1. 在 state 增加动作尝试计数与最近 3 个签名；
2. 在动作守卫前记录签名；
3. 使用标准库规范化参数和 observation fingerprint；
4. `think` 不参与；
5. 将 `N_repeat` 和重复率加入 reward breakdown。

**验收：**

- 3 步窗口内重复签名被计数；
- 相同工具但不同参数不计为重复；
- 相同工具和参数、但页面不同不计为重复；
- 守卫拒绝的重复调用仍计数；
- 计数不会改变环境实际 step 数。

### Task 3：有效组过滤

**Files:**

- Modify: `src/shopping_grpo/verl_dynamic_sampling.py`
- Modify: `tests/test_verl_dynamic_sampling.py`
- Modify: `patches/verl-0.8.0-shopping-dynamic-sampling.patch`
- Modify: `scripts/apply_verl_dynamic_sampling_patch.py`
- Modify: `scripts/check_grpo_runtime.py`
- Modify: `tests/test_verl_dynamic_sampling_patch.py`

**执行：**

1. 先为 5 条有效组规则补测试；
2. 扩展现有选择函数，使其同时检查 shaped reward、semantic reward 和 invalid 标记；
3. 修改现有 veRL 补丁传递这些字段；
4. 保持最多 3 批、完整 batch、失败即停；
5. 更新补丁原始 hash 与目标 hash；
6. 更新 preflight 以验证补丁确实生效。

**验收：**

- `[0,0,0,0]` 丢弃；
- `[1,1,1,1]` 丢弃；
- `[0.2,0.2,0.2,0.2]` 丢弃；
- shaped reward 有差异但 semantic reward 全为 0 时丢弃；
- `[0,0.2,0,0]` 且含 semantic positive 时保留；
- 包含 infrastructure invalid 成员时整组丢弃；
- 第 3 批后仍不足完整 batch 时清楚报错，不进入 optimizer。

### Task 4：配置固定

**Files:**

- Modify: `configs/verl/shop_agent_loops.yaml`
- Modify: `configs/verl/vanilla_grpo.yaml`
- Modify: `tests/test_verl_configs.py`
- Modify: `README.md`

**执行：**

1. AgentLoop 增加 `reward_mode`，默认由 `SHOPPING_REWARD_MODE` 读取；
2. 显式写入 `loss_agg_mode=token-mean`；
3. 显式固定 clip 为 `0.20/0.20`；
4. 保持 `entropy_coeff=0`，不得新增或开启完整 logits entropy 计算；
5. 保持 group size、学习率、temperature、top_p、max_steps 不变；
6. README 只增加 A0/A1 smoke 入口，不复制整份技术文档。

**验收：**

- `SHOPPING_REWARD_MODE=native` 使用原生 reward；
- `SHOPPING_REWARD_MODE=constraint_aware` 使用 shaped reward；
- 未设置环境变量时保持 native，兼容历史命令；
- Hydra 最终解析配置与本文件一致。

### Task 5：最小日志

**Files:**

- Modify: `src/shopping_grpo/verl_adapter/agent_loop.py`
- Modify: `patches/verl-0.8.0-shopping-dynamic-sampling.patch`
- Modify: `tests/test_verl_adapter.py`
- Modify: `tests/test_verl_dynamic_sampling_patch.py`

**执行：**

1. 输出第 7 节列出的 reward 与 group 指标；
2. 复用 veRL 已有 tracker，不增加监控依赖；
3. 每个 step 聚合后写一次，不逐 token 写日志；
4. 轨迹级 breakdown 保留在 `extra_fields`，方便审计。

**验收：**

- 训练日志能解释一个 group 为什么被过滤；
- 能同时看到 native reward 与 shaped reward；
- 不打印隐藏 goal 或 reward_detail 原文；
- 不显著增加 rollout 输出体积。

## 9. 本地测试门禁

实现后依次执行：

```bash
cd /Users/yyhdbl/Documents/算法/agent-rl-grpo/shopping-grpo-longhorizon

PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_shopping_reward.py' -v
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_verl_adapter.py' -v
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_verl_dynamic_sampling.py' -v
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_verl_dynamic_sampling_patch.py' -v
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_verl_configs.py' -v
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 -m compileall -q src scripts tests
git diff --check
```

任一检查失败，不进入服务器 smoke。

## 10. 服务器验证顺序

### 10.1 Preflight

```bash
source .venv-grpo-v080/bin/activate
python scripts/check_grpo_runtime.py
```

同时验证：

- ShopSimulator 为 8 slots；
- 连续两轮 8-slot reset/release 成功；
- GPU 无残留 Ray/vLLM 进程；
- 动态采样补丁 hash 正确；
- SFT merged checkpoint 可读取。

### 10.2 1-step A1 signal smoke

固定：

```text
train prompts = 2
group size = 4
最多生成 3 批
total_training_steps = 1
val_before_train = false
save_freq = -1
test_freq = -1
```

运行 A1 命令，并追加：

```bash
trainer.total_training_steps=1 \
trainer.val_before_train=false \
trainer.save_freq=-1 \
trainer.test_freq=-1
```

通过条件：

- exit code 0；
- `global_step=1`；
- 至少一个训练组 reward 有差异；
- 至少一个训练组存在 `R_semantic > 0`；
- `actor/loss` 为有限数且不为 0；
- `actor/grad_norm` 为有限数且大于 0；
- 无 OOM；
- 无 HTTP 非 2xx；
- 8 个环境全部释放；
- 结束后 GPU、Ray、vLLM 无残留。

若 3 批内找不到完整有效 batch，按任务可学性或 SFT readiness 问题停止，不绕过动态采样。

### 10.3 3-step A1 smoke

1-step 通过后再运行：

```bash
trainer.total_training_steps=3 \
trainer.val_before_train=false \
trainer.save_freq=-1 \
trainer.test_freq=-1
```

通过条件：

- 3 step 全部完成；
- 至少一个 step 的 loss 和 grad norm 非零；
- 所有 loss、grad norm、clip fraction 为有限数；
- 无 OOM、HTTP 错误和环境泄漏；
- 记录实际生成轨迹数、过滤率、GPU 峰值和总耗时；
- GPU 峰值超过显存容量 95% 时，不增加 group size。

### 10.4 短对比实验

先各运行相同的少量 optimizer steps：

```text
A0：KL-free native reward
A1：Shopping GRPO v1
```

比较：

- 有效学习 step 比例；
- 生成一条有效训练轨迹的成本；
- full success；
- max_steps；
- repeat action；
- actor loss、grad norm、clip fraction；
- GPU 小时。

短实验没有证明 A1 至少产生稳定非零学习信号时，不启动正式训练。

## 11. Benchmark

固定使用：

```text
data/benchmarks/shop_benchmark_v2_50.jsonl
```

评测配置：

```text
temperature = 0
n = 1
max environment steps = 35
同一 ShopSimulator 版本
同一 tool schema
同一 action guard
```

每次至少评测：

1. SFT merged 初始化模型；
2. A0 最终 checkpoint；
3. A1 最终 checkpoint。

主指标：

```text
full_success_rate
```

其中 full success 必须满足：

```text
r_type = r_att = r_option = r_price = 1
```

辅助指标：

```text
mean_native_reward
mean_strict_reward
mean_shaped_reward
r_type_mean
r_att_mean
r_option_mean
r_price_mean
done_rate
max_steps_rate
average_steps
repeat_action_rate
```

不得用 shaped reward 替代 full success 作为最终结论。

## 12. 实验记录

每次服务器实验在 `docs/experiments/` 新建一份记录，必须包含：

```text
Git commit
模型 checkpoint
数据文件及 SHA256
完整命令
最终 Hydra 配置
运行时版本
GPU 型号与峰值显存
optimizer steps
生成轨迹数
有效组数
过滤原因统计
reward min/mean/max
loss、grad norm、clip fraction
环境 reset/release 数
HTTP 错误数
耗时与 GPU 小时
benchmark 结果
结论与下一步
```

A0 与 A1 必须按相同 optimizer update 数比较，同时报告各自实际生成轨迹数和 GPU 小时。

## 13. 最终放大门禁

只有全部满足以下条件，才允许扩大训练：

- 本地全部测试通过；
- preflight 通过；
- 1-step A1 产生非零学习信号；
- 3-step A1 稳定完成；
- 没有基础设施错误被当作训练 reward；
- 环境 release 成功率 100%；
- 没有 OOM；
- 动态采样不会执行部分 batch 更新；
- A1 短实验的有效组比例和 full success 不劣于 A0；
- benchmark 仍为冻结的 50 条；
- 实验记录完整。

放大时第一轮只增加 `trainer.total_training_steps`，不同时修改 group size、学习率、clip、reward 权重或最大步数。
