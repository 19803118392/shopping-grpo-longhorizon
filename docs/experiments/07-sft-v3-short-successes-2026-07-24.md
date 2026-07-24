# 实验 07：SFT v3 — 加入 100 条短成功轨迹后的重新冷启动

> 状态：已完成训练、合并与固定 50 条 benchmark｜日期：2026-07-24

## 摘要

本实验没有在 SFT v2 merged 模型上续训，也没有叠加第二个 LoRA。我们从原始
Qwen3.5-2B Instruct 重新训练一套 LoRA，把原有 380 条 Action-only 成功轨迹与
新增 100 条短成功轨迹合并，希望缓解 SFT v2「会操作但长时间不购买」的问题。

训练在 RTX 4090 48GB 上使用 BF16、Liger、SDPA、20,480 token、micro batch 1
和 gradient accumulation 8。计划训练 3 epoch；实际在 step 142/159
（2.679 epoch）按人工决定早停。epoch 1 的验证损失最低，因此最终合并的是
`checkpoint-53`，不是最后一次更新。

固定 50 条 benchmark 上，购买完成率从 SFT v2 的 20% 提升到 28%，平均步数从
30.12 降到 27.58，平均 reward 从 0.1489 小幅升到 0.1540；但严格成功率从 12%
降到 10%。因此 v3 的结论是「更愿意购买、轨迹略短」，而不是全面优于 v2。

---

## 1. 实验身份与边界

| 项目 | 值 |
|---|---|
| 分支 | `agent/vanilla-grpo-runtime` |
| 代码基线 | `f145a2358054e6c26846a1674eafca5de313bcee` |
| GPU | NVIDIA GeForce RTX 4090，48GB |
| 原始 Base | `Qwen/Qwen3.5-2B` |
| 本地 Base revision | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| LoRA 输出 | `checkpoints/qwen35-2b-shopping-sft-v3-lora` |
| 最终 merged 输出 | `checkpoints/qwen35-2b-shopping-sft-v3-merged` |
| SwanLab run | [qwen35-2b-shopping-sft-v3-4090](https://swanlab.cn/@yyhdbl/shopping/runs/xsp3rzdr) |

Base 的服务器绝对路径为：

```text
/root/autodl-tmp/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc
```

其 `config.json` SHA-256 为
`ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4`，
权重 SHA-256 为
`aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`。

本实验明确没有使用：

- `qwen35-2b-shopping-sft-v2-merged`；
- SFT v2 adapter 或 checkpoint；
- 在 merged 模型上继续套 LoRA；
- benchmark task 作为训练或 validation 数据。

---

## 2. 数据构造

### 2.1 来源

旧数据不能直接使用历史 `sft.jsonl`，因为其中可能仍是 Full-CoT 格式。实际做法是
从旧 `raw.jsonl.gz` 用当前 builder 重新生成 Action-only 数据：

| 来源 | 原始行数 | 可用 Action-only 行数 | SHA-256 |
|---|---:|---:|---|
| `outputs/flash_accepted_500_parallel/raw.jsonl.gz` | 757 | 380 | `91a0b488dbd7d7c1baf9d3f53460131bed8489e7f3cf7a1cda2c4d330772aed0` |
| `outputs/sft_v3_short_100_retry/sft.jsonl` | 100 | 100 | `2cc5f11ec8e5526ac07503275f4175aadfc674d7f19b21e48f203150131ea21b` |

合并按 `task_id` 去重，并排除
`data/benchmarks/shop_benchmark_v2_50.jsonl` 的全部 task。划分使用 seed 42、
validation ratio 0.05，并按 task_id 隔离。

```bash
gzip -cd outputs/flash_accepted_500_parallel/raw.jsonl.gz \
  > outputs/sft_v3_combined/old_rebuilt/raw.jsonl

PYTHONPATH=src .venv-sft/bin/python scripts/build_sft_data.py \
  --raw outputs/sft_v3_combined/old_rebuilt/raw.jsonl \
  --accepted outputs/sft_v3_combined/old_rebuilt/accepted.jsonl \
  --rejected outputs/sft_v3_combined/old_rebuilt/rejected.jsonl \
  --stats outputs/sft_v3_combined/old_rebuilt/reject_stats.json \
  --sft outputs/sft_v3_combined/old_rebuilt/sft_action_only.jsonl

.venv-sft/bin/python scripts/merge_sft_data.py \
  --input outputs/sft_v3_combined/old_rebuilt/sft_action_only.jsonl \
  --input outputs/sft_v3_short_100_retry/sft.jsonl \
  --benchmark data/benchmarks/shop_benchmark_v2_50.jsonl \
  --output outputs/sft_v3_combined/sft_all.jsonl

PYTHONPATH=src .venv-sft/bin/python scripts/split_sft_data.py \
  --input outputs/sft_v3_combined/sft_all.jsonl \
  --train outputs/sft_v3_combined/train.jsonl \
  --validation outputs/sft_v3_combined/validation.jsonl \
  --validation-ratio 0.05 \
  --seed 42
```

### 2.2 合并与隔离结果

| 检查 | 结果 |
|---|---:|
| 合并输入 | 480 |
| 重复 task 删除 | 0 |
| benchmark task 排除 | 0 |
| 合并输出 | 480 |
| train / validation | 456 / 24 |
| 新旧 task 重叠 | 0 |
| train / validation task 重叠 | 0 |
| benchmark task 重叠 | 0 |

文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `sft_all.jsonl` | `8de67f5c7a50a8e14131bfcf283ad5cd3b96173a7a82cc1aaba5264ff00a229f` |
| `train.jsonl` | `01731d59e0abfae376bd71ec5c7fc3edee816bb991418393534011554ca51c3d` |
| `validation.jsonl` | `1a970d83c3a7942f9a98a4bd864e54f25d6f5944a6f7645afd6c9b77b82262bd` |

### 2.3 数据质量

| 检查 | 结果 |
|---|---:|
| 含 `reasoning_content` | 0 |
| 含 `<think>` | 0 |
| 含 `runtime_action_guard` | 0 |
| 含 reward detail、隐藏 goal 或标准答案 | 0 |
| tool schema entries | 5,760 |
| `additionalProperties=false` entries | 5,760 |
| Qwen3.5 Processor 可渲染 | 480/480 |
| chat-template 渲染错误 | 0 |

唯一工具 schema 的 SHA-256 为
`527be42797fd8524f7c239007922c56a9415628059258f7b03daf198263fb1e4`。

### 2.4 Token 分布与实际利用率

| 分位 | token 数 |
|---|---:|
| min | 2,968 |
| p25 | 5,184 |
| p50 | 8,157 |
| p75 | 12,492 |
| p90 | 17,942 |
| p95 | 22,065 |
| max | 40,924 |
| mean | 9,880.81 |

20,480 token 限制下：

- 全量保留 447/480（93.13%）；
- train 保留 424/456（92.98%）；
- validation 保留 23/24（95.83%）。

训练器实际看到 424 条训练样本和 23 条验证样本。33 条过长样本被丢弃，没有截断
后混入训练。

---

## 3. 显存问题与修复

### 3.1 第一次 smoke 为什么在 96GB Blackwell 上仍然 OOM

第一次 mini-batch smoke 的训练更新本身成功：

- train loss：`0.1894527`；
- grad norm：`1.456`；
- optimizer update：1 次。

OOM 出现在随后的 validation，而不是训练 forward/backward。验证路径试图为一条
16,160-token 样本物化 `[sequence, vocab]` 的 float32 logits；Qwen3.5 的词表为
248,320，单个张量预计约 16.05GB。当时 allocator 仅剩约 12.11GB，因此即便显卡
总容量为约 96GB，仍无法再分配这块连续临时张量。

这说明问题不在「只有一条特别长的训练样本」，而在 validation 为长序列保留完整
词表 logits。修复是让 Liger 验证路径只返回 loss，不把无用 logits 交给 Trainer
累计；没有安装 FlashAttention，也没有改变数据、模型或 loss。

### 3.2 为什么最终在 48GB 上反而能训练

采用 loss-only eval 后，训练与验证都不再物化上述 16GB 临时 logits。配合：

- Liger fused loss；
- PyTorch SDPA；
- gradient checkpointing；
- micro batch 1；
- 20,480 token 长度过滤；

RTX 4090 上训练脚本记录的峰值 allocated memory 仅为 14.67 GiB。显存未占满不是
配置错误：micro batch 1 是长序列稳定性的边界，gradient accumulation 8 用来保持
有效 batch 8；盲目提高 micro batch 会让最坏长度 batch 重新 OOM。

---

## 4. 训练配置与执行

核心参数：

| 参数 | 值 |
|---|---|
| precision | BF16 |
| max length | 20,480 |
| epochs | 3（计划） |
| train / eval micro batch | 1 / 1 |
| gradient accumulation | 8 |
| effective train batch | 8 |
| learning rate | `1e-4` |
| scheduler | linear |
| warmup ratio | 0.03 |
| max grad norm | 1.0 |
| LoRA r / alpha / dropout | 16 / 32 / 0.05 |
| attention | SDPA |
| fused loss | Liger |
| gradient checkpointing | 开启 |
| seed | 42 |
| logging | 每 step；SwanLab online |
| checkpoint | 每 epoch，最多 3 个 |

LoRA 覆盖普通 attention、MLP 和 Qwen3.5 Gated DeltaNet 投影：

```text
q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
in_proj_qkv in_proj_z in_proj_b in_proj_a out_proj
```

训练命令：

```bash
BASE=/root/autodl-tmp/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc

.venv-sft/bin/python scripts/train_lora_sft.py \
  --model "$BASE" \
  --train outputs/sft_v3_combined/train.jsonl \
  --validation outputs/sft_v3_combined/validation.jsonl \
  --output checkpoints/qwen35-2b-shopping-sft-v3-lora \
  --max-length 20480 \
  --epochs 3 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --warmup-ratio 0.03 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --bf16 \
  --liger-kernel \
  --attention-implementation sdpa \
  --gradient-checkpointing \
  --logging-steps 1 \
  --save-total-limit 3 \
  --seed 42 \
  --swanlab \
  --swanlab-project shopping \
  --swanlab-run-name qwen35-2b-shopping-sft-v3-4090 \
  --swanlab-mode online
```

可训练参数为 16,819,200 / 2,230,060,864（0.7542%），仍然是单层 LoRA。

---

## 5. 收敛、早停与 checkpoint 选择

| 阶段 | 平均 step loss | eval loss | checkpoint |
|---|---:|---:|---|
| epoch 1 | 0.10884 | **0.0749436** | `checkpoint-53` |
| epoch 2 | 0.07206 | 0.0754158 | `checkpoint-106` |
| epoch 3 partial | 最后记录 0.0668 | 未评测 | 未保存 |

观察：

1. step loss 从首步 0.1901 明显下降，说明训练 loss 在收敛；
2. epoch 2 的平均训练 loss 继续下降，但 eval loss 比 epoch 1 略差约 0.00047；
3. 到 step 142 时已完成 2.679 epoch，用户决定早停，SIGINT 在 backward 中被正常
   捕获；partial epoch 3 没有形成新 checkpoint；
4. 由于 epoch 1 的 eval loss 最低，手工选择 `checkpoint-53`，没有默认使用最后
   checkpoint。

实际训练到早停共约 2 小时 33 分，峰值 allocated GPU memory 为 14.67 GiB。
训练 loss 和 grad norm 始终为有限值，没有训练 OOM。

---

## 6. 合并为独立模型

```bash
.venv-sft/bin/python scripts/merge_lora_adapter.py \
  --base-model "$BASE" \
  --adapter checkpoints/qwen35-2b-shopping-sft-v3-lora/checkpoint-53 \
  --output checkpoints/qwen35-2b-shopping-sft-v3-merged \
  --bf16
```

合并采用 `peft_merge_and_unload`：

- merged `model.safetensors` SHA-256：
  `fa189989178b6d3aa7fdad82433399b9e5b39f65d141eb02a7e7c4fddff07865`；
- `merge_manifest.json` SHA-256：
  `f27267247a2c5d691e85c659d32f6189f14cef7b3d7538990335fc69f946d187`。

该目录是独立的 BF16 checkpoint，可作为后续 GRPO 的 base；不会覆盖 v2。

---

## 7. Benchmark

统一协议：

```text
benchmark = data/benchmarks/shop_benchmark_v2_50.jsonl
temperature = 0
top_p = 1
max_steps = 35
max_tokens = 512
ShopSimulator slots = 8
vLLM max_model_len = 32768
tool parser = qwen3_coder
```

先运行固定前 10 条 smoke：

| 指标 | 结果 |
|---|---:|
| done | 5/10 |
| strict success | 1/10 |
| mean reward | 0.18667 |
| average steps | 21.7 |
| error | 0 |

确认模型具备购买能力后，复用这 10 条 raw，仅补跑后 40 条。补跑耗时 689 秒，
完整 50 条结果如下：

| 指标 | SFT v2 | SFT v3 | 变化 |
|---|---:|---:|---:|
| done rate | 20% | **28%** | +8 pp |
| strict success rate | **12%** | 10% | -2 pp |
| mean final reward | 0.14889 | **0.15404** | +0.00515 |
| average steps | 30.12 | **27.58** | -2.54 |
| max_steps | 34/50 | **31/50** | -3 |
| r_type | 20% | **28%** | +8 pp |
| r_att | **12%** | 10% | -2 pp |
| r_option | **14%** | 12% | -2 pp |
| r_price | 18% | **22%** | +4 pp |
| error | 6/50 | **5/50** | -1 |

SFT v3 的严格成功 task 为：

```text
2716, 3049, 4918, 17971, 20047
```

### 7.1 HTTP 400 限制

完整评测有 5 条轨迹在第 33–34 步收到 vLLM HTTP 400：

```text
3963, 19593, 3362, 19759, 838
```

本轮没有改变协议或重跑这些任务。评测器记录失败后继续，最终 50/50 task 都有
一条 raw 记录。没有 OOM、ShopSimulator 槽耗尽或环境泄漏；50 条记录的
`release_error` 全部为 `null`。因此 50 条指标可用于与同样含错误轨迹的 v2 粗略
比较，但若要做高精度模型排名，应先单独定位这些长上下文 HTTP 400。

结果校验和：

| 文件 | SHA-256 |
|---|---|
| `outputs/eval/qwen35_2b_sft_v3_50/raw.jsonl` | `a72e2ccf42b5ce3954abbbe78857ba020316e74d60df5ed6db6b39a46d9b8bb0` |
| `outputs/eval/qwen35_2b_sft_v3_50/summary.json` | `589666f0a2d7147b8387e650b2e6abd182e00e0eb7a3af856b02155cef6baa69` |

---

## 8. 结论与下一步

### 已证明

- 新增短成功轨迹后，模型更容易执行购买：done 20% → 28%；
- max-step 轨迹减少，平均步数下降，方向符合「减少犹豫」的目标；
- 原始 Instruct → 单个新 LoRA → 独立 merged checkpoint 的训练边界正确；
- loss-only validation 解决了与训练无关的长序列完整 logits OOM；
- v3 能稳定提供非零购买 reward，可继续作为 GRPO 起点候选。

### 没有证明

- strict success 没有超过 v2，反而从 12% 变为 10%；
- 属性与选项 reward 分量没有改善；
- 50 条样本较小且有 5 条 HTTP 400，不能据此宣称统计显著提升；
- 早停发生在 2.679 epoch，本实验不是完整跑完 3 epoch 的结果。

### 理性建议

保留 v2 和 v3 两套 checkpoint。后续若做 GRPO，可优先用 v3 做最小 signal smoke，
因为它的购买率更高，更容易在同 task 的 4 条 rollout 中产生 reward 差异；但正式
比较必须继续同时监控 strict success，不能只看 done rate。

---

## 9. 产物

- 数据与 metadata：`outputs/sft_v3_combined/`
- 训练日志：`outputs/sft_v3_combined/logs/sft-v3-4090-full-train.log`
- merge 日志：`outputs/sft_v3_combined/logs/sft-v3-merge-best.log`
- LoRA checkpoint：`checkpoints/qwen35-2b-shopping-sft-v3-lora/checkpoint-53`
- merged 模型：`checkpoints/qwen35-2b-shopping-sft-v3-merged`
- 10 条 smoke：`outputs/eval/qwen35_2b_sft_v3_smoke10/`
- 50 条 benchmark：`outputs/eval/qwen35_2b_sft_v3_50/`

这些大文件和运行产物保留在服务器，不提交到 Git；Git 只保存可审计的实验记录。
