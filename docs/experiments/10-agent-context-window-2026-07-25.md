# 实验 10：Shopping Agent 上下文窗口与工具级状态投影

## 问题

GRPO A1 `global_step=50` 导出模型在固定 50 条 benchmark 的第一次有效工具调用评测中，
出现了 29 条 HTTP 400。复现 task `11075` 后，vLLM 返回的错误不是网络或
ShopSimulator 故障，而是：

```text
maximum context length: 24576
input tokens: at least 24065
requested output tokens: 512
total: at least 24577
```

模型并非输出长篇自然语言，而是在多步工具交互中反复搜索或翻页。每次
ShopSimulator 返回的商品列表和页面状态都会进入后续请求，旧 observation 累积后
耗尽 24K 上下文。SFT v3 的 5 条同类错误集中在第 33--34 步；GRPO A1 的 29 条错误
多发生在第 23--29 步，说明 GRPO 策略产生了更多上下文成本较高的工具轨迹。

这些 HTTP 400 不能算作模型决策失败，因此原 GRPO A1 50 条结果不能与 SFT v3
直接比较。

## 第一阶段（历史）：应急滑动窗口

不引入摘要模型和新依赖，使用确定性 token-aware 滑动窗口：

1. 始终保留 system prompt、原始任务和最新 tool observation；
2. 使用实际 Qwen/vLLM chat template 计算 token 数；
3. 超过预算时，从最早的完整
   `assistant tool_call + tool observation` 组开始删除；
4. 不拆开 tool call 与 `tool_call_id` 对应的 tool message；
5. 完整轨迹继续写入 raw 日志，仅压缩发送给模型的运行时视图；
6. GRPO 中同步删除旧 token 对应的 response mask 和 rollout log-prob，使 PPO
   字段保持严格对齐；
7. 若固定 prompt 与最新页面仍无法放入预算，将该轨迹标为基础设施无效并安全停止，
   不生成假 reward 或假更新。

该方法消除了 HTTP 400，但它会从在线 GRPO 的 token 轨迹中删除旧动作，因此不再作为
正式 Vanilla GRPO 的默认路径。`context_compaction_enable` 目前默认关闭，只保留为
显式应急开关。下面的 50 条结果是这一历史方案的诊断证据，不代表当前默认协议。

## 第一阶段实现

- 共享纯函数：`src/shopping_grpo/context_window.py`
- 离线采集和 benchmark：`src/shopping_grpo/teacher_rollout.py`
- veRL ShopSimulator AgentLoop：`src/shopping_grpo/verl_adapter/agent_loop.py`
- benchmark 默认窗口：24,576 token
- benchmark 单轮生成预留：512 token
- benchmark 安全余量：512 token
- GRPO 在线输入预算：16,384 token
- GRPO 单轮生成预留：512 token
- 极端单页 observation：最多 4,096 字符，确定性保留首尾

GRPO 的输入预算低于模型最大窗口，是为了同时给下一轮环境返回和 20K PPO response
序列上限留出空间。被滑窗移除的旧动作不参与该条轨迹的 PPO loss；reward 和 Vanilla
GRPO/PPO clip 公式均未改变。

新增运行指标：

- `shopping_context/compactions`
- `shopping_context/tokens_removed`

离线 raw trajectory 新增 `context_compactions`，记录触发 step、压缩前后 token 数、
删除的交互组数与 token 数。

## CPU 与定点验证

- Python 编译检查：通过
- 配置解析：通过
- 仓库单元测试：158/158 通过
- veRL 0.8 `ShoppingToolAgentLoop` 实际导入：通过

使用原先必定产生 HTTP 400 的 task `11075` 完整历史回放：

- 压缩前：25,461 token
- 压缩后：22,149 token
- 删除最旧 3 组完整交互，共 3,312 token
- 最新页面保留
- vLLM 返回 HTTP 200，并继续生成 `back_to_search`

## 固定 50 条复评

为保持单变量比较，复评继续使用：

- benchmark：`data/benchmarks/shop_benchmark_v2_50.jsonl`
- 模型：GRPO A1 `global_step=50`
- temperature：0
- top_p：1
- max_steps：35
- max_tokens：512

执行命令：

```bash
PYTHONPATH=src .venv-grpo-v080/bin/python scripts/evaluate_shop_benchmark.py \
  --benchmark data/benchmarks/shop_benchmark_v2_50.jsonl \
  --output outputs/eval/qwen35_2b_grpo_a1_step50_v2_50_context_window/raw.jsonl \
  --summary outputs/eval/qwen35_2b_grpo_a1_step50_v2_50_context_window/summary.json \
  --base-url http://127.0.0.1:5700 \
  --model shopping-grpo-a1-step50 \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --max-steps 35 \
  --temperature 0 \
  --top-p 1 \
  --timeout 180 \
  --max-tokens 512 \
  --context-window 24576 \
  --context-safety-margin 512 \
  --context-compaction \
  --observation-token-budget 0
```

### 结果

| 指标 | SFT v3 历史结果 | GRPO A1 旧运行时 | GRPO A1 新运行时 |
|---|---:|---:|---:|
| 完成任务 | 50 | 50 | 50 |
| HTTP/error | 5 | 29 | **0** |
| 正常终局 | 14 | 15 | 14 |
| max_steps | 31 | 6 | 36 |
| 严格成功 | 5 | 5 | 5 |
| 严格成功率 | 10% | 10% | 10% |
| mean final reward | 0.154040 | 0.149595 | 0.148484 |
| 平均工具步数 | 27.58 | 22.28 | 27.76 |

新运行时的 5 个严格成功 task 仍为：
`2716, 3049, 4918, 17971, 20047`。

上下文窗口行为：

- 30/50 个任务至少触发一次压缩；
- 共触发 284 次压缩；
- 首次触发位置分布在第 23--34 步；
- 每次实际发送给模型的最大输入为 23,545 token，低于 23,552 token 输入预算；
- 累计移除 1,448 个旧工具交互组；
- 完整 raw 日志仍保存所有历史消息；
- 总耗时 582 秒（9 分 42 秒）；
- 50 条全部写入 summary，评测进程正常结束；
- 退出后 8 个 ShopSimulator slot 均可重新 reset，并成功释放 8/8。

### 观察与结论

运行时修复明确有效：旧 GRPO 评测中的 29 条上下文超限 HTTP 400 降为 0，长轨迹
能够真正执行到 35 步，并且没有环境泄漏。

模型效果没有好转：严格成功仍为相同的 5/50，平均 reward 从 SFT v3 的 0.154040
变为 0.148484。旧 GRPO 的 done rate 和平均步数受到 29 条提前中断轨迹污染，不能
用于判断提升；新结果说明 A1 50-step 至少没有在固定 benchmark 上产生可见的严格
成功率提升。

SFT v3 历史结果仍有 5 条 error。若要做完全对称的最终 SFT/GRPO 对比，应让 SFT v3
也使用本运行时重跑同一 50 条；这不影响“GRPO 新运行时已消除自身 HTTP 400”的结论。

## 第二阶段：工具感知的确定性状态投影

### 为什么修改

第一阶段证明根因是 35 步内 observation 的累计，而不是单次模型回答过长。对历史文件
进一步审计：

- GRPO A1 新 50 条中共有 1,388 个 tool observation，最长 2,391 字符，没有一个超过
  4,096 字符；
- SFT v3 中共有 7,228 个 tool observation，仅 7 个超过 4,096 字符，最长 4,335 字符。

因此仅给单次工具结果设置统一字符上限不能保护总上下文。纯右侧硬截断还可能保留冗长
描述、却删除 `Buy Now`、返回按钮、规格选项和当前可点击 ASIN。若模型看到截断页面，
动作守卫却继续使用完整页面，还会产生模型不可见的动作边界。

### 投影契约

新增 `src/shopping_grpo/observation_projection.py`，由 benchmark、Teacher/SFT 重渲染和
veRL AgentLoop 共用同一个纯确定性函数：

1. 搜索结果页保留 query、page、前 10 个商品的 ASIN/标题/价格、搜索状态和完整导航
   按钮，预算为 768 token；
2. 商品详情页预算为 4,096 token，以完整保留真实规格列表和购买入口；
3. 其他长页面预算为 768 token，保留正文头部、少量尾部、截断标记和完整操作区；
4. 使用实际 Qwen tokenizer 计数，不以字符数近似 token；
5. 投影后再次验证 token 上限和关键 footer；无法安全投影时将轨迹标为
   `infrastructure_invalid`，不把它当作模型失败；
6. `state["latest_observation"]` 与返回给模型的 `ToolResponse.text` 都写入同一份
   visible observation；完整 raw observation 只用于诊断；
7. 动作守卫只从 visible observation 提取可用 ASIN 和按钮，因此模型可见集合与守卫
   允许集合一致。

veRL 自带的 `max_tool_response_length=16384` 只保留为不会正常触发的字符级故障保护，
不再承担语义投影。reward、优势、Reference KL、rollout-log-prob bypass 和 Vanilla
GRPO/PPO clip 均未修改。

### 新增观测指标

在线 GRPO 输出：

- `shopping_projection/count`
- `shopping_projection/truncated_count`
- `shopping_projection/raw_tokens`
- `shopping_projection/visible_tokens`
- `shopping_projection/truncation_ratio`
- `shopping_projection/visible_asin_count`
- `shopping_projection/visible_button_count`
- `shopping_projection/footer_failures`
- `shopping_projection/guard_rejection_rate`
- `shopping_context/max_input_tokens`
- `shopping_context/overflow`

每条 trajectory 的 `extra_fields.shopping` 同时保留对应计数。离线 benchmark summary
额外按“本条轨迹是否发生投影截断”分桶统计 task 数和 strict success，并记录投影后的
守卫拒绝率与最大上下文。

### 固定 10 条低成本对比

使用同一个 GRPO A1 step-50 模型、同一批固定 task、temperature=0、
`max_steps=35`、`max_tokens=512`，且关闭旧历史删除：

| 运行时 | 正常终局 | max_steps | strict | mean reward | 平均步数 | 最大上下文 |
|---|---:|---:|---:|---:|---:|---:|
| 投影前历史结果 | 5 | 5 | 1 | 0.17 | 21.7 | 未记录 |
| 搜索 448 / Top-8 | 1 | 9 | 0 | 0.05 | 32.3 | 11,404 |
| 搜索 768 / Top-10 | 2 | 8 | 1 | 0.15 | 29.6 | 16,965 |

两组投影评测均为 10/10 完成、HTTP/error=0、release error=0、旧历史
compaction=0、footer failure=0。448/Top-8 明显过度压缩，因此最终默认值固定为
768/Top-10。10 条样本只用于运行时选择，不能作为模型效果的显著性结论。

输出目录（不进入 Git）：

- `outputs/eval/qwen35_2b_grpo_a1_projection_smoke10/`
- `outputs/eval/qwen35_2b_grpo_a1_projection768_smoke10/`

### SFT/GRPO 一致性门禁

现有 SFT v3 使用完整 observation，不能直接把投影格式只引入 GRPO。项目新增
`scripts/project_sft_observations.py`，从已有 Action-only JSONL 生成独立副本，不覆盖
原数据：

```bash
PYTHONPATH=src .venv-grpo-v080/bin/python scripts/project_sft_observations.py \
  --model /root/autodl-tmp/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc \
  --input outputs/sft_v3_combined/train.jsonl \
  --output outputs/sft_v3_projected_v1/train.jsonl \
  --metadata outputs/sft_v3_projected_v1/train.metadata.json
```

最终离线结果：

- 480 行、7,228 个 tool observation；
- 1,438 个搜索结果页被投影，其他页面均未超过对应预算；
- raw 3,654,932 token，visible 2,603,205 token，比例 0.7122；
- footer failure=0；
- 20,480 长度门限下可训练 471/480；其中 train 448/456、validation 23/24；
- train/validation task 交集=0，benchmark task 交集=0；
- `reasoning_content`、`<think>`、`runtime_action_guard` 和非法 tool schema 均为 0。

文件 SHA-256：

- `sft_all.jsonl`：`acab7f9a49590f6314197cbbe73f8e6b061291d69a4b2d7ec3c1d7b351826e96`
- `train.jsonl`：`56c7610a737871a690b740261d16d6a6f13db27da3d637e55d280c1be5fab90a`
- `validation.jsonl`：`d21961c34f2e4e7c931c8f89fa0789d0cc72949f92e58e9a94f3b832ac8df311`

由于未适配投影格式的 A1 模型在 10 条 smoke 中正常终局由 5 降到 2，下一步应先用这
份投影版 SFT 数据做短 SFT refresh，再让 refresh 后的模型执行 GRPO rollout、
validation 和 benchmark。当前证据不支持在旧 SFT/GRPO 模型上直接启动正式投影版
GRPO。
