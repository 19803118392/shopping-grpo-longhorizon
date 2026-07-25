# 实验 10：Shopping Agent 确定性上下文窗口

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

## 方案

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

## 实现

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
  --context-safety-margin 512
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
