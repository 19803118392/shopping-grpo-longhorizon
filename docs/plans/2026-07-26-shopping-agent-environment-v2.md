# Shopping Agent Environment v2 技术方案

**状态：** 第一轮 CPU 实现已完成；待冻结任务约束元数据和执行 GPU 模型门禁

**目标仓库：** `ShopSimulator` 当前 `main` 分支 + `shopping-grpo-longhorizon`

**分析起点：** ShopSimulator `51bb26012cee31aea7ac26177c5ffe807026ac07`；本仓库 `61126275fb41cf5c75c3ce32665ee8ad8fa4fe8b`

**最终目标：** 在完全相同的环境、数据隔离和评测条件下，验证 GRPO 是否稳定超过 SFT Baseline

**完整链路：**

```text
Teacher Rollout → SFT → GRPO → Offline Evaluation
```

本文定义的是一个稳定、可信、可训练的最小购物 Agent 环境，不是工业级电商平台，也不以构建最先进的搜索系统为目标。第一版只解决已经实质影响训练闭环的问题：商品召回不可靠、分页/Observation 信息缺失、Reward 方向错误、无效循环、基础设施错误混入训练，以及训练和评测契约不一致。

本文已合并 Environment v2 方案评审后的最终收缩决策：首版搜索只做多字段 BM25，不把 Dense Retrieval 和 RRF 作为必经阶段；主动放弃与循环检测采用简单、可解释的计数规则；初期只维护轻量 Manifest。Dense/RRF 仅在实际轨迹证明新 BM25 仍是主要瓶颈时重新立项。

### 0.1 2026-07-26 第一轮实现状态

已在 ShopSimulator 和训练仓库完成不使用 GPU 的第一轮实现：

- 23,421 商品的可复现 SQLite FTS5 多字段 BM25 索引；
- 商品数据 SHA、字段权重、Tokenizer 和排序规则索引 manifest；
- 同一次查询缓存、20 条完整分页和 ASIN 确定性同分排序；
- ShopSimulator 结构化公开状态与本仓库 canonical renderer；
- v1/v2 工具配置分离及 `finish_without_purchase`；
- Reward v2、主动放弃、循环/最大步数终止；
- Environment v2 JSON 配置由运行时实际加载并严格校验，搜索、Reward、分页
  和终止阈值发生漂移时拒绝启动；
- `reward_unverifiable` 与 `infrastructure_invalid` 的独立诊断；
- 8 个槽共享只读商品库/索引、独立 session；
- Python 3.10 轻量 v2 环境、代码审计和 CPU 集成测试。

实际构建索引为 55MB，商品数 23,421。抽查的 10 个历史困难任务均通过
商品存在、索引存在、标题 oracle 召回、详情、规格、价格、购买路径和 Gold
Reward 检查。真实 HTTP 集成测试验证了 rank 1–20/21–40 无重叠、Gold
购买为 1.0、合格/过早主动结束分别为 -0.1/-0.25，以及不可核验购买不会
伪装成基础设施异常。

尚未进入 Teacher Rollout、SFT 或 GRPO。原始任务当前没有完整的
`hard_constraints`/`weighted_preferences` 分类；实现选择保守失败：
目标 ASIN 正常评分，非目标商品在缺少完整约束证据时标记
`reward_unverifiable`。正式生成 v2 Teacher 数据前必须先冻结任务约束元数据，
不能在 Reward 内根据模型轨迹或目标商品临时猜测。

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
2. 基础 Query 归一化；
3. 正确、无盲区的分页；
4. 紧凑且信息充分的结构化 Observation；
5. 正确购买、有效替代、主动放弃、错误购买和无效循环之间方向清楚的 Reward；
6. 主动结束但不购买的能力；
7. 精确重复与连续无新 ASIN 循环检测；
8. 基础设施无效轨迹隔离；
9. 低成本代码级任务可达性门禁；
10. 轻量版本、指标和可复现清单。

### 1.3 第一版明确不做什么

- 不做 LLM Query Rewrite；
- 不做 LLM Judge 或学习型 Reward Model；
- 首版不做 Dense Retrieval 和 RRF；
- 不做 Cross-Encoder Reranker；
- 不做 HNSW、IVF、PQ 等近似向量索引；
- 不做复杂的任务难度分层；
- 不让强模型逐题执行昂贵的可达性审计；
- 不做多模型上下文摘要；
- 不引入 DAPO、Clip-Higher、长度奖励或新的 GRPO 算法变量；
- 不为了单个 benchmark 任务手工编写搜索同义词或特殊规则；
- 不把目标 ASIN、Reward 细节、隐藏 goal 或标准答案暴露给模型；
- 不静默覆盖原始 ShopSimulator 协议和历史实验产物。

这些能力只有在 v2 已稳定、并有明确数据证明它们是下一项瓶颈时才重新评估。尤其是 Dense/RRF：只有新的 BM25 投入实际 Teacher Rollout、SFT 或 GRPO 后，仍有大量目标商品因同义、口语或类目表达差异无法召回，才进入单独的设计与消融阶段。

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
  ├── GRPO invalid/unverifiable filtering
  └── offline evaluation and manifests
```

### 3.1 ShopSimulator 负责

- 商品数据库和字段规范；
- 索引构建与加载；
- 多字段 BM25 检索；
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
- 验证训练与评测引用同一冻结 manifest；
- 分开过滤基础设施无效与 Reward 不可核验轨迹；
- 记录训练、评测和搜索行为指标。

### 3.3 兼容原则

Agent 仍只提交自然语言查询：

```text
search_products(query="飞利浦 助听器 白色")
```

不向 Agent 暴露 BM25 分数或目标商品排名。未来即使增加 Dense/RRF，搜索模式也只是环境配置，不是模型动作空间的一部分。

---

## 4. Search v1：可复现的多字段 BM25

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
- 不把完整长 description 直接拼入主检索字段；
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

### 4.4 Dense/RRF 的条件触发门槛

Dense Retrieval 和 RRF 不进入 Environment v2 首版主路径。多字段 BM25、Query 归一化、分页和结构化 Observation 投入实际轨迹后，只有同时满足以下条件才重新讨论：

- 固定审计任务中仍有大量目标商品不在 BM25 Top 150；
- 失败主要来自同义表达、生活化表达或类目名称差异，而不是 Agent 不翻页、硬约束冲突或商品数据缺失；
- Teacher Rollout 或新 SFT 的真实轨迹也复现了该问题；
- 通过少量离线 Dense probe 能证明存在显著互补召回。

届时可优先评估 `BAAI/bge-small-zh-v1.5`、精确内积检索和 RRF，但需要另行冻结模型 revision、编码协议、索引文件和融合参数。该后续方案不预先绑定 Environment v2。

### 4.5 搜索服务生命周期

- 第一阶段沿用本地同步 BM25 搜索；
- 不额外增加 Search 超时、重试、Embedding cache 或并发微批处理；
- 索引加载失败属于环境启动失败，不能降级成空搜索结果；
- 只有实际出现搜索阻塞，或后续引入 Dense Retrieval 时，才增加超时、重试和并发压力测试。

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
2. target ASIN 被写入 BM25 索引；
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

- 原始用户 Query 下目标商品的 BM25 rank；
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
达到资格后的主动放弃
  >
过早放弃或错误购买
  >
重复循环
  >
撞最大步数
```

基础设施异常不在这个序列中，因为它们必须被排除而不是赋普通 Reward。

Reward 的判定顺序固定为：

```text
先判断商品是否有资格
→ 再判断商品满足需求的程度
→ 最后判断是否为目标 ASIN
```

目标 ASIN 不是唯一可获得正奖励的商品，但目标 ASIN、正确规格且全部硬约束满足时获得最高奖励。

### 8.2 硬门禁和加权评分项

硬门禁决定一个商品是否有资格被购买，至少包括任务中明确声明的：

- 预算；
- 商品类别；
- 用户明确限定的品牌；
- 关键型号；
- 必须具备的核心功能；
- 明确要求的关键规格；

违反任一可验证硬门禁的购买不能得到稳定正收益。目标 ASIN 如果选择了错误的关键规格，也不算正确购买。

硬门禁通过后，再对下列一般偏好进行加权评分：

- 非强制品牌偏好；
- 一般配置；
- 非核心规格；
- 颜色；
- 外观；
- 次要使用偏好。

核心学习顺序是：

```text
找到正确品类
→ 避免超预算
→ 满足关键型号和核心功能
→ 再优化品牌与一般配置
→ 最后考虑颜色和外观
```

各子属性的具体权重在 Reward 实现阶段放入一个配置文件统一确定并冻结，不再使用所有属性简单平均或“四个属性完全同级”的逻辑。无法由商品元数据确定性验证的约束不得由 LLM Judge 临时猜测。

金额上限优先从任务文本的明确预算表达中确定性解析，支持“1万2以内”等
常见中文写法。对于“1万元左右”这类近似预算，首版固定使用 10% 上浮容差；
它仍然是硬门禁，只是先把自然语言中的近似范围确定化。无明确预算时才使用
按 ASIN 和 instruction 固定种子的回退值。该解析规则随 Reward 配置冻结。

### 8.3 初始离散 Reward 表

第一版建议使用简单终局值，避免复杂塑形：

| 终止类型 | 建议初始 Reward | 含义 |
|---|---:|---|
| `gold_purchase` | `1.0` | 目标商品、关键规格和全部硬约束正确 |
| `valid_alternative_purchase` | `0.4` | 非目标 ASIN，但全部可验证硬约束满足 |
| `graceful_stop` | `-0.1` | 达到最低探索资格后主动结束，未找到合适商品 |
| `early_abstain` | `-0.25` | 未满足最低探索要求便退出 |
| `wrong_purchase` | `-0.4` | 购买违反硬约束或关键规格错误 |
| `repeat_loop` | `-0.6` | 完全重复或连续 4 步没有新 ASIN |
| `max_steps` | `-0.7` | 未完成且耗尽最大步数 |

这些终局值作为 v2 首版固定候选，不写死在散落代码中。实现时放入单一版本化 Reward 配置，并在 Teacher 数据生成前完成一次离线分布审计；一旦开始正式数据生成便冻结。

不使用统一的大额逐步惩罚。正常搜索、翻页、比较候选和查看规格是必要探索，不能与无进展重复同等处罚。

### 8.4 有效替代商品与属性不可核验

非目标 ASIN 只有在下列条件全部成立时才可获得正收益：

- 类目正确；
- 不违反预算；
- 明确品牌要求满足；
- 所有可验证关键规格满足；
- 商品和选项元数据完整到足以确定性判断。

如果元数据不足，不能把“无法确认违反”当成“已确认满足”。有效替代判定必须输出逐项证据，便于审计。

对于非目标商品，如果品牌、规格或其他关键字段缺失，导致硬门禁无法确定性核验：

#### GRPO

- 标记为 `reward_unverifiable`；
- 作为无效奖励轨迹从当前 group 中剔除；
- 不参与 Advantage 或梯度计算；
- 在 bounded 采样预算内补采；
- 不把“不知道是否满足”默认计成 `wrong_purchase` 或 Reward 0。

#### SFT

轨迹可以保留，但必须经过动作过程质量检查，确认它仍是合理的工具操作示范。SFT 是否保留不代表该购买在 Reward 意义上是成功。

#### 评测

单独统计：

```text
unverifiable_purchase_rate
```

它不计为严格成功，也不计入普通错误购买率。目标 ASIN 的轨迹继续使用已有 Gold 信息评分，不受此规则影响。

---

## 9. 主动结束但不购买

增加单一、明确的工具，例如：

```text
finish_without_purchase(reason="no_suitable_product")
```

`reason` 只用于可读日志，不参与自由文本 Reward 判断。该动作：

- 终止当前轨迹；
- 不计为成功或 strict success；
- 根据最低探索门禁区分 `graceful_stop` 与 `early_abstain`；
- 始终释放环境租约。

### 9.1 最低有效探索

主动放弃资格使用简单计数：

```text
eligible_for_abstain =
    distinct_normalized_queries >= 2
    OR
    (
        distinct_normalized_queries >= 1
        AND opened_asins >= 1
    )
```

规则解释：

- 至少执行过两个不同的有效规范化 Query，可以主动结束；
- 或至少搜索过一次并打开过一个候选商品，可以主动结束；
- 重复相同规范化 Query 不增加计数；
- 翻页次数不单独构成有效探索；
- 达标后结束为 `graceful_stop = -0.1`；
- 未达标便结束为 `early_abstain = -0.25`。

第一版不使用 Jaccard、信息增益或 Query 语义相似度作为放弃资格。该规则容易被模型理解、容易测试，也明确防止第一步直接退出。

---

## 10. 无进展与死循环检测

### 10.1 精确重复

将工具名称和规范化后的完整参数作为动作签名：

```text
action_signature = (tool_name, canonical_parameters)
```

当前动作与前一个动作签名完全一致时，记一次连续重复。同一动作首次出现不计重复；之后连续重复计数达到 2 时终止，即连续出现 3 次完全相同动作时：

```text
termination_reason = repeat_loop
reward = -0.6
```

任一不同动作都会重置该连续重复计数。第一版不依赖通用状态指纹。

### 10.2 连续四步没有新 ASIN

环境维护本条轨迹已经对模型展示过的 ASIN 集合。每次环境动作后计算：

```text
new_asin_count = 当前 Observation 中首次出现的 ASIN 数
```

如果 `new_asin_count = 0`，连续无新 ASIN 步数加一；出现至少一个新 ASIN 时清零。连续 4 步没有任何新 ASIN，立即终止：

```text
termination_reason = repeat_loop
reward = -0.6
```

该规则覆盖不断换相似搜索词、结果始终相同、在搜索与返回页面之间反复切换、长期没有产生新候选等主要浪费模式。

第一版明确不实现：

- 结果集合 Jaccard 阈值；
- Query Embedding 相似度；
- 复杂滑动窗口；
- 通用状态指纹分析；
- LLM 循环判断。

需要记录每次触发时的最近动作和 `new_asin_count`，用于检查“详情页内正常查看属性/规格却连续没有新 ASIN”这类潜在误伤。只有真实轨迹证明误伤显著，才扩展 progress 定义或采用更复杂规则。

如果同一 prompt 的 4 条 rollout 全部以相同 `repeat_loop` Reward 结束，仍然不会凭空产生 GRPO 梯度。环境的职责是停止浪费；veRL bounded dynamic sampling 负责过滤并有限补采；搜索策略本身需要由更好的 Teacher/SFT 数据改善。不能为了制造 reward variance 人工篡改同组 Reward。

---

## 11. 终止与基础设施错误

统一终止枚举：

```text
gold_purchase
valid_alternative_purchase
wrong_purchase
graceful_stop
early_abstain
repeat_loop
max_steps
reward_unverifiable
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

这里的“Reward 必需字段缺失”指环境响应协议本应提供却异常缺失；非目标商品自身的品牌、规格等元数据不足则使用 `reward_unverifiable`，不设置 `infrastructure_invalid`。

GRPO 中基础设施无效轨迹和 `reward_unverifiable` 轨迹都不能当作 Reward 0，也不能进入 Advantage 或梯度计算。按现有 group 对齐约束剔除受影响 group，并在 bounded budget 内补采。两者必须在日志中分别统计：前者是运行问题，后者是 Reward 数据不足。无论模型、环境还是上层框架在哪一步失败，都必须在 `finally` 路径尝试释放环境，并记录释放结果。

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

第一阶段保存轻量 manifest，只要求：

- ShopSimulator Git commit；
- `shopping-grpo-longhorizon` Git commit；
- 商品数据版本或 SHA-256；
- 任务数据版本或 SHA-256；
- Search 配置；
- Reward 配置；
- Observation 版本；
- Tool 版本；
- 最大步数；
- 随机种子。

环境快速迭代期不建设复杂的跨仓库 Contract Hash 系统。在正式 Teacher Rollout、SFT 和 GRPO 启动前，再根据已经稳定的实际产物补充索引 SHA、Schema Hash、依赖版本等完整信息。

Teacher、SFT、GRPO 和评测仍必须引用同一个冻结 manifest。Manifest 不一致的结果不能直接作为同环境增益对比。

原始 ShopSimulator 协议和历史实验目录保留。Environment v2 使用新分支、配置和输出目录，不覆盖历史 baseline。

---

## 13. 指标体系

### 13.1 最终任务指标

- strict success rate；
- mean final reward；
- gold purchase rate；
- valid alternative purchase rate；
- wrong purchase rate；
- graceful-stop rate；
- early abstain rate；
- unverifiable purchase rate；
- repeat-loop rate；
- max-step rate；
- infrastructure-invalid rate。

`done_rate` 不能单独代表成功。

### 13.2 搜索指标

离线 oracle audit：

- BM25 Recall@20/50/150；
- target rank；
- zero-hit rate。

在线 Agent 行为：

- Query 数与去重后 Query 数；
- 每次 Query 的结果数量；
- 新 ASIN 数；
- Next/Prev 次数；
- 打开的不同 ASIN 数；
- 同动作连续重复数；
- 连续无新 ASIN 步数；
- 搜索耗时。

Target rank 只用于离线审计和最终分析，不能进入模型 Observation。

如果后续触发 Dense/RRF 评估，再补充 Dense/Hybrid Recall、互补召回、RRF 排名变化、Query Embedding 延迟和 cache 指标。

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

### P0：轻量快照，不扩建基础设施

工作：

- 冻结当前 ShopSimulator main commit、商品库和 benchmark；
- 写轻量 Environment v2 manifest；
- 保存当前原始环境、当前 projector v2、SFT v3 和已有 GRPO 的历史结果；
- 不建设复杂跨仓库 Contract 系统。

Go 条件：

- 同一配置两次生成相同 manifest；
- 历史 checkpoint、日志和 benchmark 不被覆盖；
- v1/v2 可以明确区分。

### P1：多字段 BM25

工作：

- 实现确定性的商品文档构建；
- 重建多字段 BM25；
- 在固定非训练审计集上记录 Recall@20/50/150。

Go 条件：

- 两次重建结果和排序一致；
- 目标 ASIN 全部存在于索引；
- oracle title self-test 通过；
- 150 条分页无缺失、无重复；
- 相比旧 BM25 不出现大规模召回倒退。

### P2：Query 归一化、分页和结构化 Observation

工作：

- 实现基础 Query 归一化；
- 修复并冻结 20 条一页的分页；
- ShopSimulator 输出结构化搜索和商品状态；
- 本仓库 canonical renderer 接入；
- 搜索页展示当前页全部 20 个商品；
- 详情、规格和导航保持完整。

Go 条件：

- 150 条分页无缺失、无重复；
- 模型可见 ASIN/button 集合等于动作守卫允许集合；
- 20 个当前页商品全部可见和可打开；
- 不存在 target/reward/goal 泄漏；
- 固定状态渲染结果确定；
- 上下文预算内不删除关键操作入口。

### P3：Reward v2

工作：

- 实现硬门禁与加权评分项；
- 实现终局 Reward 表；
- 实现有效替代判定证据；
- 实现 `reward_unverifiable`。

Go 条件：

- Reward 顺序矩阵全部通过；
- 错误购买不获得正收益；
- 非目标有效替代可得到较低正奖励；
- Gold 商品错误关键规格不算 Gold；
- 缺少关键字段的非目标购买标为 `reward_unverifiable`；
- 不使用逐步惩罚。

### P4：主动放弃和简化循环检测

工作：

- 增加 `finish_without_purchase`；
- 实现两 Query 或一 Query 加一商品的资格门槛；
- 实现连续完全相同动作检测；
- 实现连续 4 步没有新 ASIN 检测；
- 统一正常行为的 termination reason。

Go 条件：

- 第一动作放弃为 `early_abstain`；
- 达到资格后放弃为 `graceful_stop`；
- 连续重复两次可有限终止；
- 连续 4 步无新 ASIN 可有限终止；
- 正常购买路径不会被规则误杀。

### P5：基础设施无效轨迹隔离与环境验证

工作：

- 统一 `infrastructure_invalid`；
- 在 GRPO 中分开剔除 infrastructure invalid 与 reward unverifiable；
- 对全任务运行代码级 reachability gate；
- 连续 reset/release；
- gold path 购买和 Reward smoke；
- 分页、上下文和错误注入测试；
- 输出冻结 manifest。

Go 条件：

- benchmark 与训练数据无 task_id 重叠；
- gold path 和 Reward 全部可执行；
- 环境错误被标为 infrastructure invalid；
- 所有租约释放；
- 不需要 GPU。

### P6：重新生成 Teacher 和新 SFT Baseline

只有 P0–P5 冻结后才启动：

1. 在 Environment v2 上重新生成 Teacher Rollout；
2. 构建与 v2 Observation 一致的 Action-only SFT；
3. 从原始 Base 训练新的 SFT Baseline；
4. 在冻结 benchmark 上评测 SFT。

`finish_without_purchase` 由强模型 Teacher 在新环境中自然生成，不使用规则策略额外制造 Silver Data。不采用旧 checkpoint 做 SFT Refresh 作为正式方案。

### P7：GRPO 与同条件评测

1. 从新 SFT Baseline 的 exact checkpoint 启动最小 GRPO smoke；
2. 验证非零 loss/grad、Reward 差异、补采上限和环境释放；
3. 通过稳定性门禁后再做正式 GRPO；
4. 用同一 benchmark 和冻结 manifest 比较新 SFT Baseline 与 GRPO。

旧 SFT/GRPO 不是“没有用”：它们仍是故障定位、显存配置、veRL 适配和历史行为证据。但由于搜索、Observation、Reward 和终止规则发生变化，它们不能作为 Environment v2 下最终公平对比的训练起点或正式 baseline。

---

## 15. 测试与验收矩阵

### 15.1 Search

- 商品文档两次构建 SHA-256 一致；
- BM25 两次重建对固定 Query 排名一致；
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
- 非目标商品关键字段缺失为 reward unverifiable；
- 第一动作放弃为 early abstain；
- 两个不同规范化 Query 后放弃为 graceful stop；
- 一次搜索并打开候选后放弃为 graceful stop；
- 连续出现三次完全相同动作触发 repeat loop；
- 连续 4 步无新 ASIN 触发 repeat loop；
- 完整正常购买路径不触发 loop；
- max steps 终止明确；
- 环境/API/release 异常标记 infrastructure invalid。

### 15.4 端到端

- gold path 从 reset 到 buy/reward/release 完整通过；
- 8 个并发 env slot 可租用并全部释放；
- Teacher/SFT/GRPO/eval 引用同一冻结 manifest；
- benchmark task 不进入任何训练数据；
- GRPO infrastructure-invalid group 不参与 advantage/update；
- GRPO reward-unverifiable group 不参与 advantage/update；
- bounded dynamic sampling 仍有最大生成批数；
- 无 GPU 条件下可以完成 P0–P5 的全部必要验证。

---

## 16. 风险和止损边界

### 16.1 BM25 仍可能存在语义召回缺口

多字段 BM25 仍可能无法处理同义、口语和类目表达差异。第一版先用真实 Teacher/SFT/GRPO 轨迹确认问题规模；只有它仍是主要瓶颈时才评估 Dense/RRF。

### 16.2 后续 Dense 也不是天然更好

Dense 可能召回“语义相关但硬约束错误”的商品。未来如果启用，必须做 BM25/Dense/Hybrid 离线消融，不能因为方案更复杂就默认替换 BM25。`bge-small-zh-v1.5` 也不保证可靠处理所有跨语言标题。

### 16.3 简化循环规则可能误伤

在商品详情页连续查看描述、属性、评论和规格时，可能连续 4 步没有新 ASIN。首版接受这个简单规则的 trade-off，但必须记录触发前动作；如果真实正常购买路径被显著误杀，再扩展 progress 定义，不能直接加入复杂语义检测。

### 16.4 元数据质量影响 Reward

有效替代商品判定依赖品牌、类目、规格和价格字段。如果字段缺失或冲突，必须标为“不可验证”，不能宽松给正奖励。需要记录不可验证率。

### 16.5 主动放弃可能成为捷径

如果 `graceful_stop` 太容易，模型会学会只做两个表面不同的 Query 后结束。持续监控 early/graceful stop 比例和放弃前实际行为；只有出现明显投机时才收紧门槛。

### 16.6 新 Observation 带来分布变化

结构化 Observation 会改变模型输入分布，因此必须重新生成 Teacher 数据并训练 v2 SFT Baseline。不能只在 GRPO rollout 阶段切换新格式。

### 16.7 调参会污染比较

BM25 字段权重、Reward 权重和循环阈值都可能改变 benchmark 难度。它们只能在训练开始前通过独立审计冻结；正式训练后不得根据 benchmark 结果反向调环境。

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
多字段 BM25
→ Query 归一化和正确分页
→ 结构化 Observation
→ Reward v2
→ finish_without_purchase
→ 简化循环检测
→ 基础设施无效轨迹隔离
→ 代码级环境验证
→ 重新生成 Teacher Rollout
→ 从 Base 训练新 SFT
→ 最小 GRPO Smoke
→ 正式 GRPO
→ 同条件离线评测
```

实施前只做 P0 的轻量快照，第一项功能工作是多字段 BM25。不应直接开始训练，也不应先改 veRL。veRL 当前只负责在固定环境上完成策略优化；搜索、Reward、终止和 Observation 的事实问题应先在环境侧解决。

Dense Retrieval 和 RRF 不在主路径中。只有实际轨迹证明 BM25 召回仍是主要瓶颈，才重新讨论引入。

---

## 19. 条件触发 Dense 时的备选技术依据

- BGE Small 中文模型卡与推荐检索用法：[`BAAI/bge-small-zh-v1.5`](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- BGE 模型结构配置：[`config.json`](https://huggingface.co/BAAI/bge-small-zh-v1.5/blob/main/config.json)
- FlagEmbedding 官方实现：[`FlagOpen/FlagEmbedding`](https://github.com/FlagOpen/FlagEmbedding)
- FAISS 索引说明：[`Faiss indexes`](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- FAISS 索引选择建议：[`Guidelines to choose an index`](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
- RRF 原始论文：[`Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods`](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)

这些资料只保留为未来条件触发 Dense/RRF 时的备选依据，不属于 Environment v2 首版实施内容。
