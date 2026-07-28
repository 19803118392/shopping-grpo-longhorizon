# Shopping Agent 轨迹评测流水线计划

**状态：** 方案口径已确认；GRPO 训练期间允许开发隔离的纯离线评测模块
**适用范围：** 当前仓库、ShopSimulator Environment v2.1、Reward v3，以及本项目训练得到的 Base、fresh SFT、fresh GRPO 模型
**非目标：** 不建设通用 Agent 评测平台，不支持其他 Benchmark，不修改训练 Reward，不重新划分测试集

---

## 1. 目标

在现有 ShopSimulator rollout、Reward v3 和 benchmark 代码基础上，补齐一套强绑定本项目的离线轨迹评测流水线，用于：

1. 公平比较 Base、SFT、GRPO 的任务结果和行为差异；
2. 判断用户 Query 中各项需求是否得到满足；
3. 评价完整购物轨迹的搜索、核验、决策和终止质量；
4. 统计确定性的效率、合法性、上下文和异常指标；
5. 支持按错误类型定位 Badcase，并在后续训练数据、Prompt、Reward 或 GRPO 配置变化后进行回归。

评测结果用于诊断，不替代当前 Reward，也不反向参与本代 GRPO 训练。

---

## 2. 已确认的评测原则

### 2.1 四类结果分开报告

评测结果固定分成四部分：

1. 当前仓库 Reward v3 与环境终局结果；
2. Query 需求级 Rubric 的满足情况；
3. 完整 Rollout 的五维轨迹质量；
4. 确定性效率、合法性、上下文和异常指标。

四类结果不混合，不计算综合总分。

允许各部分出现不同结论。聚合报告分别展示每部分的分布和 Base/SFT/GRPO 配对差异。

### 2.2 Reward 与 Rubric 不互相覆盖

Reward 以当前代码实际结果为准，Rubric 以冻结后的 Query 需求约束为准。两者发生冲突时，不选择一方覆盖另一方，而是保留：

```text
reward_type: gold_purchase
rubric_status:
  budget: violated
reward_rubric_disagreement: true
```

当前已确认存在预算自然语言解析覆盖不足的样本，例如 Query 中的“价格别超100”未生成 `price_upper`。本代暂不修改预算解析器和 Reward；该差异由评测流水线显式记录。

### 2.3 五个轨迹质量维度固定

第一版只使用以下五个维度：

1. 搜索策略 Search Strategy；
2. 候选利用 Candidate Utilization；
3. 证据核验 Evidence Verification；
4. 决策与购买 Decision Quality；
5. 终止与效率 Termination and Efficiency。

每个维度使用 `0 / 1 / 2` 分档，输出分数和简短理由，但不设置维度权重，也不计算五维总分。

### 2.4 hard/soft 初始策略

当前任务数据没有原生 hard/soft 标签。第一版按以下原则生成候选标签：

- 明确品类、明确预算上限、否定要求、指定规格或选项：`hard`；
- “优先、最好、倾向、左右”等偏好表达：`soft`；
- 无法可靠判断的约束不得强行猜测，需要保留判断来源，必要时标记为待复核。

Flash 模型只能在代码提取的候选约束中选择、对齐和自然语言化，不能新增候选集合中不存在的需求，也不能修改底层字段、操作符和期望值。

### 2.5 Judge 的可见信息边界

评价 Actor 行为时，Judge 只能使用：

- 用户原始 Query；
- 冻结后的需求 Rubric；
- Actor 当时实际看到的投影后 Observation；
- Actor 实际输出的文本和工具调用；
- 白名单化的终局状态：`done`、`over`、`termination_reason` 和 Actor 可见的购买信息；
- 代码统计出的行为指标：工具效率、重复、合法性和上下文。

`raw_observation` 仅用于评测系统审计和排查投影问题，不得作为 Actor 行为评分证据。Judge 不能使用 Actor 当时看不到的 Gold 商品私有字段来断言它忽略了某个候选。

Judge 输入在数据层禁止携带 Environment Reward、Reward 分项、成功结论、
`reward_detail.evidence`、hard gates、weighted score、target ASIN match 和
infrastructure validity。Reward/终局面板与 Judge 面板只在 Judge 完成后由代码拼装。

### 2.6 正式模型比较协议

最终模型完成并由 validation 独立选定后，Base、fresh SFT、fresh GRPO 使用：

- 同一份任务清单；
- 同一 Environment v2.1 / Reward v3；
- 同一 Collector；
- 同一 Actor system prompt 和工具 schema；
- 同一上下文与 Observation 投影配置；
- 同一推理参数；
- 每个任务相同尝试次数。

三组结果按 `task_id` 做配对比较。

现有仓库契约和 final test metadata 目前只明确写了 SFT/GRPO 配对评测。根据用户最新决定，计划目标调整为 Base/SFT/GRPO 三者评测；进入实现前需要同步更新正式实验契约和 final test 协议说明，但不得改写 final 200 的 task 内容和 SHA256。

---

## 3. 当前正式盲测集

当前 final test 已制作并冻结：

```text
data/benchmarks/shop_benchmark_reward_v3_final_200.jsonl
```

冻结信息：

- 任务数：200；
- 完整任务全集：23,421；
- 排除历史/当前已见任务：6,720；
- 剩余候选任务：16,701；
- 选择方式：固定种子、确定性、无放回随机抽取；
- 与 fresh SFT、完整 2,000 probe、GRPO train/validation、旧 benchmark 50/200 零重叠；
- final test SHA256：
  `2c4ff070e13ddc30796d38e85170210e7d3c211992425a62090f2419fe8e0208`；
- 当前 metadata 必须保持 `evaluated=false`，直到 GRPO checkpoint 已由 validation 独立选定。

final 200 不得用于：

- GRPO checkpoint 选择；
- Prompt 或 Judge prompt 调优；
- Rubric/Judge 人工校准；
- 超参数选择；
- 训练数据或 Badcase 预筛选。

Rubric 和 Judge 的开发、校准使用现有非盲测 SFT、probe 和历史 benchmark 轨迹。final 200 的 Rubric 只在评测协议和模型全部冻结后批量生成并缓存，不进行基于任务内容的人工调参。

---

## 4. 当前数据基础

### 4.1 任务与商品事实

Rubric 候选约束可以从以下现有字段提取：

- 当前 instruction 的完整自然语言 Query；
- 目标商品 category；
- `instruction.attributes`；
- `instruction.instruction_options`；
- 目标商品 customization option axis；
- Query 中的显式品牌和型号；
- Query 中的价格上限、区间和偏好表达；
- 目标商品标题、描述、属性和价格，仅作为候选约束的结构化对齐证据。

目标商品的全部属性不能自动转成用户要求。标准单轮任务中 Actor 没有看到的 persona、reason key 和其他隐藏字段不进入 Rubric。

### 4.2 当前 Rollout

现有离线 Rollout 已保存：

- 完整 messages；
- 已执行工具步骤；
- 工具名称、参数和环境 action；
- Actor 实际可见 Observation；
- raw Observation 和投影信息；
- 每步 reward、done 和 terminal result；
- Action Guard 拒绝记录；
- context token 和 compaction 记录；
- error、release error。

这些数据足以支持第一版五维 Judge 和 step 级证据引用。需要先离线生成稳定的 `event_id`、`action_attempt_id` 和标准化步骤表示。

### 4.3 当前 GRPO 在线轨迹限制

当前 GRPO adapter 主要输出终局 Reward、步数、重复动作、上下文和 Guard 聚合指标，没有在本仓库中确认完整逐步 Observation 的持久化输出。

正式评测不依赖训练在线轨迹。最终 checkpoint 统一通过离线 Collector 重新运行。

---

## 5. 需求 Rubric 流程

### 5.1 生成流程

```text
任务 Query 与目标商品结构化字段
→ 代码提取候选约束
→ 受限 Flash 模型筛选、对齐、自然语言化
→ 代码校验模型输出
→ 按 task_id 和版本冻结缓存
```

代码校验至少保证：

- 输出引用的 candidate ID 确实存在；
- 不允许生成新的底层字段、值或操作符；
- 需要 Query span 的约束确实能回指原文；
- hard/soft 判断保留来源；
- 同一 task 的 Base/SFT/GRPO 共用完全相同的 Rubric；
- 输入任务数据或生成配置变化时，不会误用旧缓存。

### 5.2 Rubric 内容

每条 Rubric 计划保留：

- `rubric_id`；
- `candidate_id`；
- 约束类型；
- 人类可读描述；
- `hard / soft`；
- hard/soft 判断来源；
- Query 原文片段及位置；
- 底层字段；
- 操作符；
- 期望值；
- 数据来源；
- 生成与校验状态；
- Rubric/schema/prompt/model 版本。

每个任务的 Rubric bundle 还应记录 task data hash、Query hash、候选抽取器版本和缓存时间。

### 5.3 Flash 模型

第一版计划复用 Teacher rollout 使用的 OpenCode Go OpenAI-compatible 接口：

- Rubric 生成：`deepseek-v4-flash`；
- API endpoint 和 key：复用现有 Teacher 配置；
- key 只从环境变量读取，不写入 manifest、日志或缓存；
- 原始请求和响应需要脱敏后保存，便于审计；
- 使用确定性设置并缓存结果，避免 Base/SFT/GRPO 分别调用造成 Rubric 漂移。

具体请求参数在实现阶段通过小规模非盲测样本验证后冻结。

---

## 6. 轨迹 Judge

### 6.1 输入

Judge 输入包括：

- 用户 Query；
- 冻结的需求 Rubric；
- 固定的五维评分规范；
- 规范化后的 Actor 可见 Rollout；
- 白名单化的终局状态；
- 工具效率、重复、合法性和上下文统计。

输入不包括：

- Actor 未看到的 raw Observation 内容；
- 未经 Rubric 筛选的 Gold 商品完整属性；
- persona 或其他未提供给 Actor 的私有信息；
- Reward v3 分数、分项、evidence 和代码判定的成功结论；
- `strict_gold_success`、`purchase_success`、`reward_type`、
  `final_reward`、`terminal_utility`、`weighted_score`；
- 其他模型在相同任务上的结果。

每个模型的轨迹独立评分，避免 Judge 因先看到其他模型结果而产生比较偏差。模型间比较在 Judge 之后由代码完成。

### 6.2 输出

Judge 使用结构化输出，至少包含：

- 每条 Rubric 的：
  - `satisfied`
  - `violated`
  - `unknown`
  - `not_applicable`
- 每个状态的简短理由和证据 event/step IDs；
- 五个轨迹维度各自的 `0 / 1 / 2` 分和理由；
- 主要错误类型；
- 次要错误类型；
- 支持错误判断的 event/step IDs；
- 简短整体诊断；
- Judge 结果是否有效及解析错误信息。

Judge 不生成综合总分，也不能修改 Reward、终局状态或确定性指标。

### 6.3 五维初版评分锚点

#### 搜索策略

- `0`：核心品类或关键条件缺失，搜索明显无效或机械重复；
- `1`：初始搜索合理，但改写、去重或收敛一般；
- `2`：覆盖核心品类和关键条件，并根据结果有效改写、缩短或翻页。

#### 候选利用

- `0`：忽略明显高匹配的可见候选，持续探索低相关候选；
- `1`：使用了合理候选，但比较不足或略有冗余；
- `2`：识别高匹配候选，完成必要比较后及时收敛。

#### 证据核验

- `0`：未核验关键属性、规格或最终价格便购买或放弃；
- `1`：核验部分关键项，但仍有重要需求缺乏证据；
- `2`：通过可靠商品页面核验所有决策关键要求和变体价格。

#### 决策与购买

- `0`：违反硬约束、选错规格、明显仓促购买或错误放弃；
- `1`：选择基本合理，但存在部分需求未满足或证据不足；
- `2`：最终选择满足硬约束、规格正确，并得到明确轨迹证据支持。

#### 终止与效率

- `0`：过早终止、重复循环，或已有合格候选仍耗尽步骤；
- `1`：存在轻度多余探索或稍早终止；
- `2`：证据充分后及时购买，或充分探索后合理放弃。

### 6.4 Pro 模型

第一版计划复用 Teacher rollout 使用的 OpenCode Go OpenAI-compatible 接口：

- Judge：`deepseek-v4-pro`；
- API endpoint 和 key：复用现有 Teacher 配置；
- 默认单轨迹单次有效 Judge；
- 使用确定性设置；
- 保存脱敏后的原始响应、结构化结果、模型名和 prompt/schema 版本；
- API 或 JSON 解析失败只做有限重试，不能静默生成默认分数。

Judge prompt 第一版由本项目生成，后续由用户参与迭代。正式 final 200 开始前必须冻结 prompt 版本。

---

## 7. 确定性指标

以下指标优先通过代码生成，不交给 Judge 猜测。

### 7.1 Reward 与任务结果

- Reward version、type、valid；
- final reward、terminal utility；
- hard gates；
- weighted score 和 dimension scores；
- purchase success；
- strict gold success；
- done、over；
- termination reason；
- reward/Rubric disagreement。

### 7.2 工具与效率

- 执行工具步数；
- action attempt 数；
- 搜索次数；
- 打开商品次数；
- 信息页访问次数；
- 规格选择次数；
- 购买和主动放弃次数；
- 可见候选数和打开候选数；
- 购买前已核验的需求数量。

### 7.3 重复行为

分别统计：

- 重复搜索 Query；
- 连续重复搜索；
- 重复 canonical action；
- 连续重复 action；
- 环境 `repeat_loop`；
- GRPO runtime 定义的 repeat action。

这些指标定义不同，不能合并为同一个“重复次数”。

### 7.4 合法性与异常

- Action Guard 拒绝次数和原因；
- malformed tool call；
- schema 或参数错误；
- 环境拒绝或无效 action；
- invalid action limit；
- context hard limit；
- Collector/model/environment/release error；
- infrastructure invalid；
- 缺失、重复、损坏的 task 记录。

### 7.5 上下文、Token 与耗时

现有数据可直接统计：

- Observation 投影次数；
- 投影截断次数；
- raw/visible token；
- context compaction；
- 每轮已记录的输入 token；
-最大输入上下文。

正式评测前计划补充：

- prompt、completion、total、cached token；
- 整条轨迹总耗时；
- 每次模型请求耗时；
- 每次环境工具调用耗时；
-重试次数；
-日志文件大小、输出 hash 和磁盘占用。

上述新增埋点只在 GRPO 训练结束、评测协议确认后修改离线 Collector，不修改当前训练 runtime。

---

## 8. 汇总与报告

### 8.1 单任务结果

每个 task 的最终记录由以下部分组成：

```text
Run/Actor metadata
+ Frozen Rubric
+ Raw/Normalized trajectory reference
+ Reward and terminal result
+ Deterministic metrics
+ Judge result
+ Disagreement flags
```

### 8.2 模型汇总

每个模型分别报告：

- fixed-denominator gold purchase rate；
- purchase success rate；
- Reward type 分布；
- Reward/终局统计；
- 各 Rubric 的 satisfied/violated/unknown 分布；
- hard 和 soft 需求满足率；
- 五维分数分别的分布；
- 主要/次要错误类型分布；
- 工具、重复、Guard、context、token、time 和异常指标；
- Judge coverage 和 Judge invalid 数量。

不输出综合总分。

### 8.3 配对比较

Base、SFT、GRPO 按相同 task 配对，重点展示：

- 成功状态转移；
- Reward type 转移；
- hard Rubric 违反减少或增加；
- 五维行为变化；
- 步数、重复和 Guard 差异；
- Reward/Rubric 冲突；
- 代表性正向案例和 Badcase。

---

## 9. Badcase 回归集

正式评测后，按照主要错误类型选择典型 task，形成固定 Badcase manifest。

初始错误类型计划覆盖：

- 搜索条件遗漏；
- 无效/重复搜索；
- 忽略高匹配可见候选；
- 核验不足；
- 价格未核验；
- 规格选择错误；
- 违反品类或预算硬约束；
- 仓促购买；
- 过早放弃；
- 找到合格商品后继续无效探索；
- repeat loop；
- max steps；
-非法工具调用；
-上下文信息丢失；
- Reward/Rubric disagreement；
- infrastructure invalid。

Badcase 集只在 final 评测完成后构建，不能用于选择本代 GRPO checkpoint。

---

## 10. GRPO 训练期间的隔离边界

GRPO 训练期间允许开发不被训练代码 import、也不依赖训练运行状态的纯离线评测
模块，包括：

- 编写和审阅本计划；
- 定义数据 schema、Rubric 和 Judge prompt；
- 新增独立的轨迹规范化、确定性指标和报告模块；
- 使用已经完成保存的非盲测轨迹进行小规模、低资源验证；
- 编写只覆盖新离线模块的单元测试；
-设计报告格式。

训练期间禁止：

- 修改 ShopSimulator、Reward、goal/parser、终止逻辑；
- 修改 Action Guard、Observation 投影、工具 schema 或 Actor prompt；
- 修改 veRL adapter、GRPO runtime、训练配置或 train/validation manifest；
- 启动正式 benchmark、Rubric 或 Judge 批处理；
- 使用 final 200 做调试或人工校准；
- 与训练共用 ShopSimulator slot、GPU 或高强度磁盘/网络资源；
- 安装或升级当前训练环境中的依赖；
- 修改现有训练进程会 import 的公共模块或包初始化文件；
-重启训练依赖的环境服务或修改其环境变量。

即使训练进程通常已加载 Python 代码，也不能依赖该行为；worker 可能延迟启动或
重启，因此共享 runtime 文件在训练期间一律冻结。新增离线模块不得注册到训练入口，
不得由包初始化代码自动 import，也不得读取正在追加的训练日志或 checkpoint 文件。

---

## 11. 分阶段实施

### 阶段 A：GRPO 训练期间

1. 确认本计划中的评测口径；
2. 冻结 Rubric 和 Judge 的数据 schema；
3. 冻结五维评分规范和初版错误 taxonomy；
4. 设计 Flash/Pro prompt 第一版；
5. 使用已有非盲测轨迹制定人工校准样例；
6. 明确正式 run manifest 和报告结构；
7. 实现隔离的轨迹规范化、确定性指标、Rubric/Judge schema 和报告骨架；
8. 只对已保存的非盲测轨迹做小规模、低资源测试；
9. 不修改或运行训练共享链路。

### 阶段 B：GRPO 训练结束、final checkpoint 选择前

1. 在独立离线模块中实现 TaskFacts 和候选约束抽取；
2. 实现 Flash 受限 Rubric 生成、校验和缓存；
3. 实现轨迹事件规范化和确定性指标；
4. 实现 Pro Judge 和结构化结果校验；
5. 在非盲测轨迹上进行小规模人工校准；
6. 冻结 Rubric/Judge prompt、schema、模型和参数版本；
7. 补充 Collector 所需的 token/time/run manifest 埋点；
8. 用非 final task 做端到端 dry run。

### 阶段 C：GRPO checkpoint 由 validation 独立选定后

1. 锁定 Base、SFT、GRPO actor artifact；
2. 校验 final 200 和 exclusions SHA256；
3. 将 final metadata 从未评测状态进入正式评测状态；
4. 按统一协议运行 Base、SFT、GRPO；
5. 生成 final 200 的冻结 Rubric；
6. 计算确定性指标；
7. 使用冻结 Judge prompt 评分；
8. 生成四部分分栏报告和三模型配对报告；
9. 保存完整 run manifest、缓存、原始响应和审计信息。

### 阶段 D：正式评测完成后

1. 人工复核代表性冲突和 Judge 异常；
2. 按主要错误类型构建 Badcase 回归集；
3. 记录 Judge prompt 的后续改进点；
4. 后续训练代际使用 Badcase 做优先回归，但不回写本代 final 结果。

---

## 12. 第一版验收标准

第一版流水线至少满足：

1. Base/SFT/GRPO 对同一 task 使用完全相同的 Rubric；
2. 四类结果独立保存且没有综合总分；
3. Reward v3 结果原样保留；
4. Reward/Rubric 分歧可检索；
5. Judge 证据只能引用 Actor 可见步骤；
6. 每项 Rubric 和五维评分均能回指 event/step；
7. 确定性指标不由 LLM 生成；
8. final test 不参与 prompt、checkpoint 或超参数选择；
9. 所有模型、prompt、schema、数据和代码版本可追溯；
10. API key 和隐藏 Gold 信息不进入 Actor 输入或公开报告；
11. 中断、缺失和基础设施异常不会被静默过滤；
12. 正式成功率以固定 200 task 为分母。

---

## 13. 已确认的补充口径

以下口径已于 GRPO 正式训练开始后由用户确认。训练期间允许实现隔离的纯离线评测
代码，但不修改训练共享代码，也不运行 Rubric、Judge 或正式评测批处理任务。

1. **Base 的正式范围：**
   Base/SFT/GRPO 都跑 final 200。Base 使用仓库契约中的固定本地
   Qwen3.5-2B snapshot；后续实现时同步修改现有“只配对 SFT/GRPO”的协议说明。

2. **基础设施无效轨迹的 Judge 口径：**
   基础设施无效轨迹仍计入固定 200 的任务成功率分母，但不强制填五维
   `0` 分；记录 `judge_status=not_judged` 并单独报告 Judge coverage，避免把
   服务器故障解释为模型行为。

3. **final 200 Rubric 的生成时点：**
   只在 checkpoint、prompt 和 Judge 版本全部冻结后自动批量生成；不基于
   final task 内容继续调 prompt。

4. **Judge 校准规模：**
   第一轮从已有非盲测轨迹中选择约 20–30 条，覆盖 gold、partial、wrong、
   repeat、max steps、unverifiable 和 Reward/Rubric 冲突。

5. **Judge 调用次数：**
   第一版每条轨迹只保留一次有效 Pro 判断，temperature 设为 0，并缓存原始
   响应；不做多 Judge 投票。只在非盲测校准集上进行少量重复调用以检查一致性。

6. **Rubric 的待复核状态：**
   对无法可靠判定 hard/soft 的候选允许 `needs_review`，不要求 Flash 强制
   二选一。正式 Rubric 冻结前再决定保留、人工修正或丢弃。

7. **API 配置：**
   复用 Teacher 的 OpenCode Go endpoint 和 key。实现时从现有环境配置确认
   环境变量名和精确 model ID；代码、文档、manifest 和日志均不得写入 key 本身。

---

## 14. 当前实施进度

### 14.1 GRPO 训练期间已完成的隔离模块

当前已在独立的 `shopping_grpo.evaluation` 子包中完成第一批纯离线能力：

- Rubric、Judge、规范化轨迹、确定性指标和单任务结果的版本化契约；
- raw Rollout 到 Actor 可见事件流的规范化；
- executed step、Guard rejection 和 action attempt 的稳定 ID；
- 默认排除 raw Observation、Gold goal 和 persona 的 Judge 可见性边界；
- 终局 purchase 白名单和仅含效率、重复、合法性、上下文的 Judge 指标白名单；
- Reward v3 evidence、成功结论与轨迹 Judge 的数据级隔离；
- Reward/终局、工具效率、重复、合法性、上下文和异常指标；
- 代码约束的 Rubric 候选及受限 Flash 输出物化；
- “价格别超100”等 Query 预算候选提取，但不修改 Reward parser；
- 五维 0/1/2 Judge 提示词草案和结构化结果校验；
- 禁止综合总分、禁止未知 Rubric/candidate/event ID；
- Reward/Rubric disagreement 检测；
- 四部分单任务结果和固定任务全集分母汇总；
- Base/SFT/GRPO 按 task_id 的分栏配对比较；
- 固定的主要/次要错误 taxonomy；
- 原子离线 artifact、重复键校验和盲测文件名保护；
- 纯离线预处理、Judge 请求生成、结果拼装和比较 CLI；
- TaskFacts 环境映射与独立导出入口；
- Flash/Pro OpenAI-compatible JSON 客户端及严格结构校验；
- Flash/Pro 结果逐条持久化、完整 Judge 请求哈希校验和断点续跑；
- 不允许 API key 等凭据进入 run manifest。

这些模块没有被 GRPO 训练入口、ShopSimulator 或顶层包初始化导入。

### 14.2 已完成验证

- 使用合成轨迹验证事件顺序、Guard、重复动作、Rubric、Judge、四部分汇总和
  模型配对比较；
- 使用已保存的非盲测真实轨迹验证 Actor 可见 Observation 边界；
- 对已有 SFT 50-task raw 做只读一致性检查，新指标得到：
  - 29 个 strict gold；
  - 611 个执行工具步；
  - 平均 12.22 步；
  - Reward type 分布与现有 Reward v3 summary 完全一致。

当前共完成 22 项隔离测试。模型客户端测试使用 mock transport，并在显式移除
`OPENAI_API_KEY`、`OPENAI_BASE_URL` 的环境下通过。验证没有调用 ShopSimulator、
Ray、GPU、OpenCode Go API 或 final 200。

### 14.3 尚未实施

以下工作继续遵循本计划的阶段边界：

- 实际运行 ShopSimulator 私有任务数据的批量 TaskFacts 导出；
- 使用 OpenCode Go 做小规模 Flash/Pro 联通性和 JSON 契约验证；
- 正式 artifact 目录与 run manifest 实例；
- Judge 人工校准及 prompt 冻结；
- completion/cached token 和耗时 Collector 埋点；
- Base/SFT/GRPO 正式 final 200 rollout；
- final 报告和 Badcase 回归集。
