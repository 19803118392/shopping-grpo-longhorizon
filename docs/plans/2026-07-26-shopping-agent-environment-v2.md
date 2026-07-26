# Shopping Agent Environment v2 技术方案

**状态：** 设计共识已冻结，尚未实现

**目标仓库：** `ShopSimulator` 当前 `main` 分支 + `shopping-grpo-longhorizon`

**分析起点：** ShopSimulator `51bb26012cee31aea7ac26177c5ffe807026ac07`；本仓库 `61126275fb41cf5c75c3ce32665ee8ad8fa4fe8b`

**最终目标：** 在完全相同的环境、数据隔离和评测条件下，验证 GRPO 是否稳定超过 SFT Baseline

**完整链路：**

```text
Teacher Rollout → SFT → GRPO → Offline Evaluation
```

本文定义的是一个稳定、可信、可训练的最小购物 Agent 环境，不是工业级电商平台，也不以构建最先进的搜索系统为目标。第一版只解决已经实质影响训练闭环的问题：商品召回不可靠、分页/Observation 信息缺失、Reward 方向错误、无效循环、基础设施错误混入训练，以及训练和评测契约不一致。

---

## 1. 已达成的核心共识

### 1.1 科学问题

我们最终要回答的不是“某个模型能否在某个临时配置下偶尔买对”，而是：

> 从同一个 SFT checkpoint 出发，在冻结的环境和评测协议下，GRPO 是否能带来可复现的增益？

因此必须保证：

- SFT Baseline 与 GRPO 使用同一搜索系统；
- Teacher Rollout、SFT、GRPO rollout、validation 和最终评测使用同一套工具、Observation、Reward、最大步数和终止规则；
- benchmark 不进入 Teacher、SFT 或 GRPO 训练；
- 基础设施失败不作为模型失败参与优化；
- 环境版本在正式生成数据前冻结，训练中途不得静默修改。

### 1.2 第一版要做什么

Environment v2 只包含：

1. 可复现的多字段 BM25；
2. 轻量 Dense Retrieval；
3. BM25 与 Dense 的 RRF 融合；
4. 正确、无盲区的分页；
5. 紧凑且信息充分的结构化 Observation；
6. 正确购买、有效替代、主动放弃、错误购买和无效循环之间方向清楚的 Reward；
7. 主动结束但不购买的能力；
8. 精确重复与“无新信息”循环检测；
9. 低成本代码级任务可达性门禁；
10. 完整的版本、指标和可复现清单。

### 1.3 第一版明确不做什么

- 不做 LLM Query Rewrite；
- 不做 LLM Judge 或学习型 Reward Model；
- 不做 Cross-Encoder Reranker；
- 不做 HNSW、IVF、PQ 等近似向量索引；
- 不做复杂的任务难度分层；
- 不让强模型逐题执行昂贵的可达性审计；
- 不做多模型上下文摘要；
- 不引入 DAPO、Clip-Higher、长度奖励或新的 GRPO 算法变量；
- 不为了单个 benchmark 任务手工编写搜索同义词或特殊规则；
- 不把目标 ASIN、Reward 细节、隐藏 goal 或标准答案暴露给模型；
- 不静默覆盖原始 ShopSimulator 协议和历史实验产物。

这些能力只有在 v2 已稳定、并有明确数据证明它们是下一项瓶颈时才重新评估。

---

## 2. 为什么需要 Environment v2

现有审计确认，抽查的 10 个任务中目标商品都真实存在于商品数据库和 BM25 索引，但 8 个失败任务中有 7 个目标商品未进入当前查询的可见召回范围。主要现象包括：

- 用户表达与商品标题的词面不同，例如“洗牙设备”与“洁牙器”；
- 查询包含过多硬约束，BM25 命中数反而骤降；
- 品牌、型号、类目、规格和属性没有被稳定地作为独立字段利用；
- Agent 不翻页或反复换近义词，但返回的结果集合几乎不变；
- 旧 projector 曾把环境每页 20 个商品裁成前 10 个，形成永久分页盲区；
- 原 Reward 会让违反预算或关键约束的错误购买仍得到稳定正收益；
- 统一小额步数惩罚不足以区分有效探索和无进展循环；
- HTTP、环境租约、上下文溢出等基础设施失败没有形成统一的训练剔除口径。

当前 `shopping-observation-v2` 已修复“每页 20 个商品只显示 10 个”的直接盲区，并保证模型可见 ASIN 与动作守卫允许 ASIN 一致。Environment v2 将保留这个原则，但把它上升为搜索、分页、Observation 和动作校验共享的正式契约。

---

## 3. 总体架构和职责边界

```text
Agent
  │
  │ search_products(query)
  ▼
ShopSimulator Environment v2
  ├── QueryNormalizer
  ├── MultiFieldBM25Searcher
  ├── DenseSearcher
  ├── RRFFusion
  ├── SearchSession / Pagination
  ├── Product / Option / Purchase State
  ├── Termination State Machine
  └── Deterministic Reward
  │
  │ structured environment state
  ▼
shopping-grpo-longhorizon
  ├── canonical Observation renderer
  ├── tool schema and action guard
  ├── Teacher/SFT data construction
  ├── veRL AgentLoop adapter
  ├── GRPO infrastructure-invalid filtering
  └── offline evaluation and manifests
```

### 3.1 ShopSimulator 负责

- 商品数据库和字段规范；
- 索引构建与加载；
- BM25、Dense 和 Hybrid 检索；
- 搜索 session、排序和分页；
- 商品详情、规格选择和实际购买状态；
- 终止原因和 Reward 的事实源；
- reset、step、release 生命周期；
- 环境版本与索引 manifest。

搜索排名和当前页面商品必须由环境产生。AgentLoop 不应自行重排、补商品或维护另一份分页状态。

### 3.2 `shopping-grpo-longhorizon` 负责

- 对 ShopSimulator 结构化状态做唯一、确定性的模型可见渲染；
- 让模型可见状态与动作守卫使用同一份状态；
- 管理统一工具 schema；
- 生成 Teacher、SFT 和 veRL parquet；
- 验证训练/评测契约 hash；
- 过滤基础设施无效轨迹；
- 记录训练、评测和搜索行为指标。

### 3.3 兼容原则

Agent 仍只提交自然语言查询：

```text
search_products(query="飞利浦 助听器 白色")
```

不向 Agent 暴露 `bm25_search`、`dense_search`、RRF 分数或目标商品排名。搜索模式是环境配置，不是模型动作空间的一部分。

---

## 4. Search v1：可复现的轻量 Hybrid Search

### 4.1 商品文档

每个商品构建一份统一的规范化文档，字段至少包含：

```text
title
brand
category
model
top_attributes
options
short_bullets
```

要求：

- 字段缺失必须显式处理，不能因空值改变文档顺序；
- 不把完整长 description 直接送入 Embedding；
- Dense 文档控制在约 128～256 tokens；
- 不包含 task goal、target ASIN 标记、Reward、用户目标或答案字段；
- 商品唯一键固定为 ASIN；
- 商品遍历顺序固定，所有并列排序最终按 ASIN 确定性打破。

### 4.2 Query 归一化

第一版只做规则明确、可回放的基础处理：

- Unicode 规范化；
- Latin 字母大小写统一；
- 全角/半角、空白和常见标点归一；
- 中文数量、单位和常见写法做有限规范化；
- 去除工具层引入的无意义模板词；
- 保留品牌、型号、数值、颜色和关键规格；
- 不做 LLM 改写；
- 不维护针对 benchmark 的人工词典。

原始 Query 和归一化 Query 都进入日志，但只有归一化 Query 进入检索。归一化规则必须有独立版本和单元测试。

### 4.3 多字段 BM25

BM25 不再只依赖一段混合文本，而是至少区分：

- 标题；
- 品牌；
- 类目；
- 型号；
- 属性和卖点；
- 规格选项。

第一版使用固定字段权重，并把分词器、停用词表、字段权重、BM25 参数和商品库 SHA-256 写入 manifest。具体权重通过离线 held-out 搜索审计确定一次，正式生成 Teacher 数据前冻结，不在训练期间调参。

默认召回：

```text
BM25 Top 150
```

### 4.4 Dense Retrieval

第一版使用轻量中文 Embedding：

```text
BAAI/bge-small-zh-v1.5
```

采用它的原因是模型规模较小、中文检索定位明确、输出维度为 512，并且可以在约 2.3 万商品规模下使用精确搜索。官方模型卡建议检索 Query 使用指令，Corpus 不加指令；该行为必须作为配置固定，而不能由调用方自由变化。

必须冻结：

- Hugging Face model revision；
- tokenizer revision；
- Query instruction；
- pooling 方法；
- 最大长度；
- embedding normalization；
- dtype；
- 构建脚本版本；
- 商品文档 SHA-256；
- embedding 文件 SHA-256。

第一版要求 CPU float32 构建和查询路径可用。约 23,421 个商品的 `512 × float32` 原始向量约占 46 MiB，因此优先采用 NumPy 矩阵乘法做精确内积检索；如果后续使用 FAISS，也只使用等价的 `IndexFlatIP`，不引入近似索引。

默认召回：

```text
Dense Top 150
```

`bge-small-zh-v1.5` 是中文侧方案，第一版不宣称它能可靠解决所有跨语言召回。

### 4.5 RRF 融合

BM25 与 Dense 使用 Reciprocal Rank Fusion：

```text
score(d) = Σ 1 / (k + rank_i(d))
```

第一版固定：

```text
k = 60
BM25 Top 150
Dense Top 150
最终去重 Top 150
```

融合只使用排名，不直接混合两个不可比的原始分数。相同 RRF 分数时按固定规则排序，最终必须能在相同输入和索引下得到完全一致的 ASIN 序列。

环境保留三个离线消融模式：

```text
bm25
dense
hybrid_rrf
```

模式只用于 P2 的离线对比。选择正式模式后，Teacher、SFT、GRPO 和最终评测统一冻结为同一个模式，禁止按任务动态切换。

### 4.6 搜索服务生命周期

- 商品向量在服务启动时加载一次；
- Embedding 模型在进程内共享，不按 env slot 重复加载；
- Query embedding 支持小批量和缓存；
- cache key 包含归一化 Query 与 Search 版本；
- 记录 Search p50/p95 latency；
- cache 命中不得改变排序；
- 索引或模型加载失败属于环境启动失败，不能降级成空搜索结果。

---

## 5. 正确分页与搜索结果页

### 5.1 唯一分页事实源

Search Engine 返回排序后的 Top 150，环境按固定页容量切片：

```text
page_size = 20
page 1 = rank 1–20
page 2 = rank 21–40
...
```

Projection 不能再次改变页面边界，也不能只展示当前页的前 K 个商品。

### 5.2 搜索页必须提供

```text
query
normalized_query
page
total_pages
total_results
rank_start
rank_end
products
navigation
```

每个商品至少提供：

```text
rank
asin
title
brand
category
price_or_price_range
compact_key_attributes
```

每页 20 个商品必须全部：

- 对模型可见；
- 对动作守卫可见；
- 可以被 `open_product` 引用；
- 在日志中保持相同 rank 和 ASIN。

### 5.3 无盲区验收

对任意返回 150 条结果的 Query，连续翻页后：

- rank 1–150 每个恰好出现一次；
- 无缺失、无重复、无跨页重排；
- `Prev`/`Next` 的边界正确；
- 当前页所有 ASIN 均可打开；
- 模型可见 ASIN 集合等于动作守卫允许 ASIN 集合。

---

## 6. Observation v2 正式契约

### 6.1 结构化状态优先

ShopSimulator 先返回结构化环境状态，本仓库再使用一个 canonical renderer 生成模型文本。不要先生成冗长页面字符串，再依靠通用右侧截断补救。

搜索页、详情页、长文本页分别渲染：

- 搜索页：完整当前页商品、排名、价格和导航；
- 详情页：ASIN、标题、价格、品牌、类目、关键属性、已选/可选规格和全部操作入口；
- Description/Features/Reviews：结构化头部、有限正文和完整尾部导航，并明确标记截断。

### 6.2 一致性

同一结构化状态同时供给：

1. 模型可见 Observation；
2. 动作守卫；
3. 轨迹日志；
4. Reward/终止诊断。

动作守卫不得访问模型看不到的隐藏页面内容。原始状态可以用于服务器诊断，但不能扩大 Agent 的可执行动作集合。

### 6.3 上下文控制

第一版继续使用确定性、规则式上下文工程：

- 当前页面保留完整投影；
- 最近少量工具交互保留完整；
- 更早的交互替换为短的确定性状态记录；
- 保留原始用户需求；
- 不调用另一模型做摘要；
- 不把一条轨迹拆成多个训练 segment。

上下文压缩必须在 Teacher、SFT、GRPO 和评测中一致。所有裁剪都要记录 token 数、比例和是否保留关键操作区。

---

## 7. 低成本任务可达性门禁

可达性门禁由代码执行，不要求当前 2B SFT Agent 成功，也不要求额外大模型跑通任务。它只验证环境本身没有损坏：

1. target ASIN 存在于冻结商品库；
2. target ASIN 被写入 BM25 和 Dense 索引；
3. 使用目标商品标题或规范化商品文档进行 oracle self-test 时能召回目标；
4. 目标详情页能打开；
5. goal 要求的规格真实存在且可以选择；
6. 选定规格后的实际价格有效；
7. `Buy Now` 可以完成；
8. gold path 能得到最高严格 Reward；
9. reset、step、finish 和 release 的协议完整；
10. 所有检查在有限时间内退出。

Oracle self-test 只用于离线环境审计，不作为 Agent Query，不写入训练 Observation，也不证明自然语言任务一定容易检索。

另行记录但不作为硬排除条件：

- 原始用户 Query 下目标商品的 BM25/Dense/Hybrid rank；
- Recall@20/50/150；
- 目标是否需要翻页；
- 查询结果数。

这样可以区分“环境路径损坏”和“自然语言检索困难”，又不会根据当前小模型能力筛掉所有困难任务。

---

## 8. Reward v2

### 8.1 基本原则

Reward 必须形成稳定方向：

```text
正确目标及正确规格
  >
满足全部硬约束的有效替代商品
  >
充分探索后的主动放弃
  >
过早放弃或错误购买
  >
重复循环
  >
撞最大步数
```

基础设施异常不在这个序列中，因为它们必须被排除而不是赋普通 Reward。

### 8.2 硬约束和软约束

硬约束至少包括任务中明确声明的：

- 预算；
- 商品类别；
- 品牌；
- 型号；
- 关键规格、颜色、尺寸或功能。

违反任一可验证硬约束的购买不能得到稳定正收益。目标 ASIN 如果选择了错误的关键规格，也不算正确购买。

软属性只有在全部硬约束通过后才参与区分。无法由商品元数据确定性验证的约束不得由 LLM Judge 临时猜测。

### 8.3 初始离散 Reward 表

第一版建议使用简单终局值，避免复杂塑形：

| 终止类型 | 建议初始 Reward | 含义 |
|---|---:|---|
| `gold_purchase` | `1.0` | 目标商品、关键规格和全部硬约束正确 |
| `valid_alternative_purchase` | `0.4` | 非目标 ASIN，但全部可验证硬约束满足 |
| `valid_abstain` | `0.0` | 完成有效探索后主动结束，未找到合适商品 |
| `early_abstain` | `-0.25` | 未满足最低探索要求便退出 |
| `wrong_purchase` | `-0.4` | 购买违反硬约束或关键规格错误 |
| `repeat_loop` | `-0.6` | 明确重复或持续无信息增益 |
| `max_steps` | `-0.7` | 未完成且耗尽最大步数 |

这些值是 v2 的首个候选配置，不写死在散落代码中。实现时放入单一版本化 Reward 配置，并在 Teacher 数据生成前完成一次离线分布审计。允许调整绝对数值，但必须保持上述严格顺序；一旦开始正式数据生成便冻结。

不使用统一的大额逐步惩罚。正常搜索、翻页、比较候选和查看规格是必要探索，不能与无进展重复同等处罚。

### 8.4 有效替代商品

非目标 ASIN 只有在下列条件全部成立时才可获得正收益：

- 类目正确；
- 不违反预算；
- 明确品牌要求满足；
- 所有可验证关键规格满足；
- 商品和选项元数据完整到足以确定性判断。

如果元数据不足，不能把“无法确认违反”当成“已确认满足”。有效替代判定必须输出逐项证据，便于审计。

---

## 9. 主动结束但不购买

增加单一、明确的工具，例如：

```text
finish_without_purchase(reason="no_suitable_product")
```

`reason` 只用于可读日志，不参与自由文本 Reward 判断。该动作：

- 终止当前轨迹；
- 不计为成功或 strict success；
- 根据最低探索门禁区分 `valid_abstain` 与 `early_abstain`；
- 始终释放环境租约。

### 9.1 最低有效探索

不能简单要求“调用过两次 search”，因为两个不同字符串可能返回完全相同的结果。

第一版要求至少发生两个有效探索事件，并且当前没有已知候选满足全部硬约束。有效事件可以是：

- 新 Query 得到实质不同的结果集合；
- 翻到新页面并看到足够多的新 ASIN；
- 打开新的候选商品；
- 查看新的规格或关键属性。

阈值必须简单、确定性且可单测。第一版不加入“必须查看 N 个候选、必须搜索 M 次、必须覆盖若干类目”等复杂规则。

---

## 10. 无进展与死循环检测

### 10.1 精确重复

检测：

- 相同规范化 Query 连续重复；
- 相同页面上的同一无效动作；
- 反复打开同一商品但状态没有变化；
- 相同参数和相同状态指纹下重复调用工具。

### 10.2 语义上的无进展

不使用 Embedding 或 LLM 判断两个 Query 是否“语义相似”。循环判定看环境结果是否带来新信息：

- 两次搜索结果 ASIN 集合的 Jaccard overlap；
- 新 ASIN 数；
- 是否出现新页面；
- 是否查看新商品；
- 是否获得新规格或属性；
- 当前候选集合是否改变。

例如：

```text
search[洁牙器]
search[洗牙设备]
search[家用洗牙工具]
search[牙齿清洁机器]
```

即使字符串不同，如果连续返回高度重叠的商品集合、没有新 ASIN、没有翻页且没有检查新候选，就属于无进展。

### 10.3 状态机

每步记录：

```text
normalized_query
result_asins
page
opened_asin
selected_options
new_asin_count
result_set_overlap
progress_event
state_fingerprint
```

阈值和窗口写入版本化配置。触发后终止为 `repeat_loop`，而不是继续消耗到 35 步。

如果同一 prompt 的 4 条 rollout 全部以相同 `repeat_loop` Reward 结束，仍然不会凭空产生 GRPO 梯度。环境的职责是停止浪费；veRL bounded dynamic sampling 负责过滤并有限补采；搜索策略本身需要由更好的 Teacher/SFT 数据改善。不能为了制造 reward variance 人工篡改同组 Reward。

---

## 11. 终止与基础设施错误

统一终止枚举：

```text
gold_purchase
valid_alternative_purchase
wrong_purchase
valid_abstain
early_abstain
repeat_loop
max_steps
context_overflow
environment_error
model_api_error
release_error
```

下列情况设置：

```text
infrastructure_invalid = true
```

- HTTP 非预期状态；
- reset/release 失败；
- 环境槽耗尽；
- Search 模型或索引加载失败；
- 环境返回结构非法；
- Reward 必需字段缺失或非有限值；
- 上下文溢出；
- vLLM、Ray 或 CUDA 运行时错误。

GRPO 中基础设施无效轨迹不能当作 Reward 0。至少整组剔除并按 bounded budget 补采。无论模型、环境还是上层框架在哪一步失败，都必须在 `finally` 路径尝试释放环境，并记录释放结果。

---

## 12. 版本冻结和可复现清单

正式版本建议使用：

```text
shopping-env-v2
search-v1
reward-v2
observation-v2
tools-v2
```

每次运行保存单一 manifest，至少包含：

- ShopSimulator Git commit；
- `shopping-grpo-longhorizon` Git commit；
- 商品数据库路径、数量和 SHA-256；
- 任务/benchmark 路径、数量和 SHA-256；
- BM25 索引 SHA-256；
- Dense embedding 与 ASIN 映射 SHA-256；
- Embedding 模型及 revision；
- Query normalization 版本；
- Search mode 和全部参数；
- Reward 配置 SHA-256；
- Tool schema SHA-256；
- Observation 配置 SHA-256；
- page size、max steps、上下文预算；
- Python 和关键依赖版本；
- 随机种子。

Teacher、SFT、GRPO 和评测启动时都检查 contract hash。不同 hash 的结果不能直接作为同环境增益对比。

原始 ShopSimulator 协议和历史实验目录保留。Environment v2 使用新分支、配置和输出目录，不覆盖历史 baseline。

---

## 13. 指标体系

### 13.1 最终任务指标

- strict success rate；
- mean final reward；
- gold purchase rate；
- valid alternative purchase rate；
- wrong purchase rate；
- valid abstain rate；
- early abstain rate；
- repeat-loop rate；
- max-step rate；
- infrastructure-invalid rate。

`done_rate` 不能单独代表成功。

### 13.2 搜索指标

离线 oracle audit：

- BM25/Dense/Hybrid Recall@20/50/150；
- target rank；
- zero-hit rate；
- 各模式互补召回；
- RRF 后排名变化。

在线 Agent 行为：

- Query 数与去重后 Query 数；
- 每次 Query 的结果数量；
- result-set novelty；
- 新 ASIN 数；
- Next/Prev 次数；
- 打开的不同 ASIN 数；
- 搜索 p50/p95 latency；
- Query embedding cache hit rate。

Target rank 只用于离线审计和最终分析，不能进入模型 Observation。

### 13.3 上下文和运行时指标

- raw/visible observation tokens；
- truncation ratio；
- context tokens before each turn；
- visible ASIN/button count；
- guard rejection rate；
- max-context rate；
- trajectory steps、tokens 和耗时；
- reset/release 成功率；
- GRPO infrastructure-invalid group 数；
- Dynamic Sampling 生成批数和剔除原因。

---

## 14. 分阶段实施计划

### P0：契约和历史基线冻结

工作：

- 冻结当前 ShopSimulator main commit、商品库和 benchmark；
- 写 Environment v2 schema、版本常量和 manifest 格式；
- 保存当前原始环境、当前 projector v2、SFT v3 和已有 GRPO 的历史结果；
- 建立跨仓库 contract hash 检查。

Go 条件：

- 同一配置两次生成相同 manifest；
- 历史 checkpoint、日志和 benchmark 不被覆盖；
- v1/v2 可以明确区分。

### P1：重建多字段 BM25

工作：

- 实现确定性的商品文档构建；
- 实现基础 Query 归一化；
- 重建多字段 BM25；
- 增加分页和 index manifest；
- 在固定非训练审计集上记录 Recall@20/50/150。

Go 条件：

- 两次重建结果和排序一致；
- 目标 ASIN 全部存在于索引；
- oracle title self-test 通过；
- 150 条分页无缺失、无重复；
- 相比旧 BM25 不出现大规模召回倒退。

### P2：Dense + RRF

工作：

- 固定 `bge-small-zh-v1.5` revision 和编码协议；
- CPU float32 构建商品 embedding；
- 实现精确内积检索；
- 实现确定性 RRF；
- 比较 `bm25`、`dense`、`hybrid_rrf`。

Go 条件：

- Dense 和 RRF 单元测试通过；
- 相同 Query 结果确定性一致；
- Hybrid 在固定审计集提供有意义的互补召回；
- CPU p95 latency 可接受；
- 内存占用和服务并发不会随 env slot 线性复制。

如果 Dense 没有带来可测召回改善，或延迟不可接受，则保留可复现 BM25 作为 v2 baseline，不为了“必须 Hybrid”继续扩大系统复杂度。

### P3：Reward、主动放弃和循环状态机

工作：

- 实现硬约束判定证据；
- 实现终局 Reward 表；
- 增加 `finish_without_purchase`；
- 实现有效探索；
- 实现精确重复和结果集无进展检测；
- 统一 termination reason 和 `infrastructure_invalid`。

Go 条件：

- Reward 顺序矩阵全部通过；
- 错误购买不获得正收益；
- 第一动作放弃被判为 `early_abstain`；
- 有效探索后放弃可判为 `valid_abstain`；
- 同 Query 重复和换词但结果不变均能有限终止；
- 正常探索不会被误判为循环。

### P4：结构化 Observation 和工具一致性

工作：

- ShopSimulator 输出结构化页面状态；
- 本仓库 canonical renderer 接入；
- 搜索页展示当前页全部 20 个商品；
- 详情/规格/导航保持完整；
- 训练和评测复用同一 renderer。

Go 条件：

- 模型可见 ASIN/button 集合等于动作守卫允许集合；
- 20 个当前页商品全部可见和可打开；
- 不存在 target/reward/goal 泄漏；
- 固定轨迹重放结果确定；
- 上下文预算内无通用右侧截断删除操作入口。

### P5：低成本环境验证

工作：

- 对全任务运行代码级 reachability gate；
- 连续 reset/release；
- gold path 购买和 Reward smoke；
- 搜索并发、分页、上下文和错误注入测试；
- 输出冻结 manifest。

Go 条件：

- benchmark 与训练数据无 task_id 重叠；
- gold path 和 Reward 全部可执行；
- 环境错误被标为 infrastructure invalid；
- 所有租约释放；
- 不需要 GPU。

### P6：重新建立可比较训练链路

只有 P0–P5 冻结后才启动：

1. 在 Environment v2 上重新生成 Teacher Rollout；
2. 构建与 v2 Observation 一致的 Action-only SFT；
3. 从原始 Base 训练新的 SFT Baseline；
4. 在冻结 benchmark 上评测 SFT；
5. 从该 exact SFT checkpoint 启动最小 GRPO smoke；
6. 通过非零 loss/grad、无环境泄漏和稳定性门禁后再做正式 GRPO；
7. 用同一 benchmark、同一环境 contract 比较 SFT 与 GRPO。

旧 SFT/GRPO 不是“没有用”：它们仍是故障定位、显存配置、veRL 适配和历史行为证据。但由于搜索、Observation、Reward 和终止规则发生变化，它们不能作为 Environment v2 下最终公平对比的训练起点或正式 baseline。

---

## 15. 测试与验收矩阵

### 15.1 Search

- 商品文档两次构建 SHA-256 一致；
- BM25 两次重建对固定 Query 排名一致；
- Dense 归一化和相似度计算正确；
- 精确检索并列排序确定；
- RRF 已知小例子结果正确；
- BM25/Dense 重复 ASIN 只保留一条；
- rank 1–150 分页无断档、无重复；
- 空 Query、无结果、特殊字符和超长 Query 有界退出。

### 15.2 Observation / Guard

- 每页 20 个商品全部显示；
- visible ASIN = actionable ASIN；
- visible button = actionable button；
- 详情页价格、规格和操作入口完整；
- 截断长文本时头部与尾部导航保留；
- 不包含 target ASIN 标记、Reward 或隐藏 goal；
- 同一结构化状态渲染文本完全一致。

### 15.3 Reward / Termination

- gold ASIN + gold option + 硬约束通过；
- gold ASIN + 错误关键 option 不算 gold；
- 非 gold 但全部硬约束满足为 valid alternative；
- 超预算、错品牌、错类目、错关键规格为 wrong purchase；
- 第一动作放弃为 early abstain；
- 有效探索后放弃为 valid abstain；
- 完全重复触发 repeat loop；
- 不同 Query、相同结果且无新信息触发 no-progress loop；
- 正常翻页和查看新候选不触发 loop；
- max steps 终止明确；
- 环境/API/release 异常标记 infrastructure invalid。

### 15.4 端到端

- gold path 从 reset 到 buy/reward/release 完整通过；
- 8 个并发 env slot 可租用并全部释放；
- Teacher/SFT/GRPO/eval contract hash 完全一致；
- benchmark task 不进入任何训练数据；
- GRPO infrastructure-invalid group 不参与 advantage/update；
- bounded dynamic sampling 仍有最大生成批数；
- 无 GPU 条件下可以完成 P0–P5 的全部必要验证。

---

## 16. 风险和止损边界

### 16.1 Hybrid 召回不是天然更好

Dense 可能召回“语义相关但硬约束错误”的商品。必须通过 BM25/Dense/Hybrid 离线消融和最终行为指标判断，而不是因为方案更复杂就默认采用。

### 16.2 Embedding 的能力边界

`bge-small-zh-v1.5` 适合轻量中文检索，但不保证可靠处理跨语言标题，也不能替代结构化过滤。第一版不因此追加多语言模型或 Reranker。

### 16.3 CPU 延迟与并发

Query Embedding 可能成为 8～24 条并发 rollout 的瓶颈。优先共享模型、缓存和微批处理；若 p95 不可接受，先回退到已验证的 BM25 v2，而不是立即增加 GPU 检索服务。

### 16.4 元数据质量影响 Reward

有效替代商品判定依赖品牌、类目、规格和价格字段。如果字段缺失或冲突，必须标为“不可验证”，不能宽松给正奖励。需要记录不可验证率。

### 16.5 主动放弃可能成为捷径

如果 `valid_abstain` 太容易或 Reward 太高，模型会学会少探索。最低探索使用信息增益而不是调用次数，并持续监控 early/valid abstain 比例。

### 16.6 新 Observation 带来分布变化

结构化 Observation 会改变模型输入分布，因此必须重新生成 Teacher 数据并训练 v2 SFT Baseline。不能只在 GRPO rollout 阶段切换新格式。

### 16.7 调参会污染比较

RRF、字段权重、Reward 数值、循环阈值都可能改变 benchmark 难度。它们只能在训练开始前通过独立审计冻结；正式训练后不得根据 benchmark 结果反向调环境。

---

## 17. 数据、存储和运行纪律

- 大型商品索引、embedding、模型、checkpoint、轨迹和日志放在 `/root/autodl-tmp` 数据盘；
- 不把大文件写入系统盘；
- 长时间索引构建、Teacher Rollout、SFT、GRPO 和评测统一使用 `screen`；
- 每个长任务保存启动命令、Git commit、manifest、stdout/stderr 和 PID；
- 训练数据和 benchmark 使用不同目录及 task_id 去重门禁；
- 不覆盖历史 v1/v2/v3 checkpoint、日志和原始 raw 轨迹；
- 密钥只通过环境变量或本机认证存储，不进入命令日志、文档和 Git。

---

## 18. 决策顺序

按以下顺序推进，任何阶段未通过 Go 条件都不进入下一阶段：

```text
冻结契约
  → 可复现 BM25
  → Dense + RRF 离线消融
  → 冻结正式 Search
  → Reward / Abstain / Loop 状态机
  → 结构化 Observation
  → 代码级 Reachability 与环境 smoke
  → 重新生成 Teacher 数据
  → 新 SFT Baseline
  → 最小 GRPO smoke
  → 正式 GRPO
  → 同条件离线评测
```

第一项实施工作应是 P0，不应直接开始训练，也不应先改 veRL。veRL 当前只负责在固定环境上完成策略优化；搜索、Reward、终止和 Observation 的事实问题应先在环境侧解决。

---

## 19. 外部技术依据

- BGE Small 中文模型卡与推荐检索用法：[`BAAI/bge-small-zh-v1.5`](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- BGE 模型结构配置：[`config.json`](https://huggingface.co/BAAI/bge-small-zh-v1.5/blob/main/config.json)
- FlagEmbedding 官方实现：[`FlagOpen/FlagEmbedding`](https://github.com/FlagOpen/FlagEmbedding)
- FAISS 索引说明：[`Faiss indexes`](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- FAISS 索引选择建议：[`Guidelines to choose an index`](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
- RRF 原始论文：[`Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods`](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)

这些资料用于确定轻量 Dense、精确内积检索和 RRF 的合理边界，不意味着第一版要复制工业检索系统。
