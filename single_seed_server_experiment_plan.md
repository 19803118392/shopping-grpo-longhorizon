# Shopping GRPO 单种子低成本机制实验计划（修订版）

> 状态：待实施，不代表已经取得新增结果。  
> 实施基线：当前本地代码快照 `c2e02a5e479f664605b2292a9c9a89cacb4688da`；正式执行前必须提交本计划及实现，并重新记录唯一冻结 commit。  
> 训练约束：所有新增训练统一使用 `seed=42`；不把任务级统计检验解释为训练种子稳定性。  
> 执行约束：本计划本身不授权启动训练、合并模型或运行任何 200 题评测。

## 1. 研究问题与不变契约

项目主线保持为：

```text
Baseline → Action-only LoRA SFT → Online GRPO → Evaluation
```

本轮只回答四个问题：

1. 固定监督优化步数时，SFT 收益有多少来自成功轨迹多样性；
2. GRPO 的收益能否由更多 SFT 计算量替代；
3. 在相同代码、初始化、训练 seed 和名义 Rollout 预算下，去除目标 ASIN 奖励加成是否改善约束满足；
4. 在核心奖励对照成立后，`G=2/4/8` 对有效组比例、Advantage 和真实 token 成本有什么影响。

以下契约不能被 Reward v4 改写：

- 运行环境始终为 ShopSimulator Environment v2.1；
- 环境权威奖励始终为 Reward v3；
- observation v2、tool schema v2 和最大 35 环境步保持不变；
- strict success 始终要求完整 `gold_purchase` 终局且 `reward_valid=true`；
- 训练数据与 `data/evaluation/tasks.jsonl` 以及任何新增留出集必须零重叠；
- Reward v4 只作为训练适配层的 `optimization_reward_v4`，不覆盖环境 Reward v3 结果。

本轮结果只能表述为“固定 `seed=42` 下的机制消融”。单训练 seed 不能证明训练稳定性，任务级 McNemar 和 paired bootstrap 也不能代替训练种子方差。

## 2. 最小实验矩阵与执行顺序

实验严格按以下顺序运行，前一阶段未通过验收时不进入后一阶段：

1. 版本、数据、Reward v3 重放和统计协议验收；
2. SFT 多样性与 More-SFT control；
3. 同条件 `Reward-v3-G4` 对 `Reward-v4-G4` 核心因果实验；
4. 只有核心 v4 通过 Dev 门槛，才运行 G2/G8；
5. 只有出现明显策略漂移或 SFT 成功保持率下降，才运行 KL 对照；
6. 冻结设置后，才允许申请执行新的未见留出集评测。

核心实验不能省略同条件 Reward v3 对照。历史 GRPO-v3 只作为项目背景，不作为 Reward v4 的因果基线。

## 3. 资源、时间与存储预算

### 3.1 运行资源

- GPU：单张 NVIDIA RTX 6000 96GB；
- 训练环境：Linux、CUDA、项目锁定的 Python 和依赖版本；
- 开发评测：冻结 `data/grpo/validation.jsonl` 的 Dev-50；
- Judge：仅对入围模型和代表性失败样本调用现有远程 Judge API；
- 输出：全部写入 Git 忽略目录，并保存配置、哈希、日志、紧凑轨迹和计时。

### 3.2 存储策略

当前一个完整 GRPO checkpoint 约 11GB，因此禁止为四个配置长期保留 step 25/50/75/100 的全部完整 checkpoint。

默认策略：

- 核心训练只长期保留最后一个可恢复 checkpoint 和 step 100；
- 中间 step 只保存标量指标和轻量审计产物；
- 如果为容错临时保存中间完整 checkpoint，后继 checkpoint 验证可恢复并导出哈希后再删除旧副本；
- 未晋级的模型不合并，只保留 LoRA、resolved config、日志和摘要；
- 精简方案建议预留 100–150GB 增量空间；若保留所有完整 checkpoint，应准备 250GB 以上空间。

### 3.3 时间预算

| 阶段 | 预计 GPU 时间 |
|---|---:|
| SFT 多样性与 More-SFT | 12–15 小时 |
| Reward v3/v4 核心 G4 对照 | 6–8 小时 |
| Dev-50×3 配对评测 | 3–5 小时 |
| 条件式 G2/G8 pilot | 2–4 小时 |
| 条件式延长或 KL 对照 | 0–8 小时 |
| 新留出集一次评测 | 仅获授权后估算 3–5 小时 |

精简方案按 25–35 GPU 小时准备；若所有条件实验均晋级，按 45–60 GPU 小时准备。另预留 10–16 人工小时用于实现、审计和文档。

## 4. 启动前检查

在任何 GPU 训练前完成并保存：

- 记录新的冻结 commit；不得继续使用当前对象库中无法核验的 `c3c178...`；
- `git status` 中不得存在来源不明的代码修改；
- 记录 OS、GPU、驱动、CUDA、Python、PyTorch、veRL 和 vLLM 版本；
- 校验 Base、SFT、GRPO、Dev-50 与留出集的行数和 SHA-256；
- 校验所有训练 task ID 与全部评测 task ID 零重叠；
- 离线重放 Reward v3，确认原始 Reward、strict success 和动态采样有效性未变化；
- 执行 CPU 测试、环境 smoke、GRPO dry-run 和一次 checkpoint 恢复测试；
- 确认日志不打印 API Key，输出目录剩余空间满足本阶段预算；
- 每次运行保存 resolved config、完整启动命令、stdout/stderr、硬件信息和耗时。

建议产物结构：

```text
outputs/single-seed-42/
  environment/
  sft/
    sft_n95_s144/
    sft_n190_s144/
    sft_n379_s288/       # step 144 是 GRPO 共同初始化，step 288 是 More-SFT
  grpo/
    reward_v3_g4_s100/
    reward_v4_g4_s100/
    reward_v4_g2_pilot/  # 条件执行
    reward_v4_g8_pilot/  # 条件执行
    reward_v4_g4_kl001/  # 条件执行
  evaluation/
    dev50x3/
    new_holdout/         # 仅在新留出集冻结且获授权后生成
  audit/
  logs/
```

## 5. SFT 多样性与 More-SFT control

### 5.1 数据构造

- 95 条必须是 190 条的子集，190 条必须是全部 379 条的子集；
- 不只按随机 seed 截断，必须按轨迹长度、任务约束数量、是否包含规格选择进行分层嵌套抽样；
- 保存每个子集的 task ID、行数、token 数和 SHA-256；
- 三个集合与所有评测任务零重叠。

### 5.2 训练设置

所有 SFT 使用相同基础模型、LoRA、Chat Template、Assistant-only Mask、batch、梯度累积、学习率和 `seed=42`。

为使 step 144 的前缀可直接比较，新增并冻结统一的 scheduler 设置：固定 warmup steps，warmup 后使用常数学习率；不得让 scheduler 总 horizon 随实验长度改变。

| 产物 | 数据 | 目标 global step | 用途 |
|---|---:|---:|---|
| `sft_n95_s144` | 95 | 144 | 低多样性、更多重复 |
| `sft_n190_s144` | 190 | 144 | 中等多样性 |
| `sft_n379_s288@144` | 379 | 144 | 高多样性、GRPO 共同初始化 |
| `sft_n379_s288@288` | 379 | 288 | 从同一次训练连续得到的 More-SFT control |

`sft_n379_s288` 必须在训练开始前就以 step 288 为目标，并在 step 144 保存可恢复 checkpoint。step 288 必须连续加载模型、LoRA、optimizer、scheduler 和 global step，不能从 1 重新开始。

除 optimizer steps 外，同时报告：

- 实际见过的训练样本次数；
- assistant loss token 总数；
- 截断率与轨迹长度分布；
- 峰值显存、训练时间和估算计算量；
- step 144 到 288 的训练/验证 loss 变化。

如果 scheduler 连续性或 step 144 checkpoint 恢复不能通过测试，More-SFT 不得进入结果表。

## 6. Adapter-only Optimization Reward v4

### 6.1 定义

Environment Reward v3 保持原样。训练适配层从经过校验的 Reward v3 公共明细计算 `optimization_reward_v4`。

设：

- `V`：`reward_valid`；
- `P`：环境确实发生购买终局；
- `H`：类别与预算 hard gates 均通过；
- `S∈[0,1]`：Reward v3 `weighted_score`；
- `C∈[0,1]`：Reward v3 `evidence_coverage`；
- `T`：未购买时的 Reward v3 终止类型。

定义：

\[
R_{opt-v4}=
\begin{cases}
0, & \neg V \\
-0.85, & P\land\neg H \\
1.0, & P\land H\land \operatorname{close}(S,1)\land\operatorname{close}(C,1) \\
\min(0.25,-0.30+0.55SC), & P\land H \\
r(T), & \neg P
\end{cases}
\]

其中 `close` 使用与环境一致的 `abs_tol=1e-8`，不能直接用浮点精确相等；`r(T)` 沿用 Reward v3：graceful stop `-0.15`、early abstain `-0.35`、max steps `-0.50`、repeat loop `-0.65`。

### 6.2 双指标契约

每条轨迹同时保存且不得混名：

- `environment_reward_v3`：环境原始终局 Reward；
- `strict_gold_success_v3`：现有 `gold_purchase + reward_valid=true`；
- `constraint_complete_purchase_v4`：购买通过 hard gates，且 `S`、`C` 均为 1；
- `optimization_reward_v4`：只供 v4 GRPO 优化和辅助分析；
- `target_asin_match`：仅作为诊断字段。

Reward v4 不产生新的“strict success”定义。简历和报告将其称为“Target-ASIN-bonus-free terminal objective”，不得称为完全无 oracle 的 Reward。

### 6.3 强制测试与离线审计

- 同一购买、hard gates、`S`、`C` 和终止结果，仅在 `gold_purchase` 与 `valid_alternative_purchase` 间切换时，v4 必须相同；
- 仅修改 `target_asin_match` 或目标 ASIN、同时冻结已编译约束时，v4 必须不变；
- Gold ASIN 但类别错误或超预算仍为 `-0.85`；
- 缺少关键规格或证据不能得到 `1.0`；
- 重复搜索和增加无效步骤不能提高终局 v4；
- `reward_valid=false` 返回 0，并从动态采样有效组中排除；
- 检查所有输入为有限数且位于合法范围，异常字段使样本无效而不是静默裁剪；
- 对历史轨迹离线双重打分，并按有效软约束数量 `0/1/2/3+` 报告满分替代商品比例；
- 人工抽查无软约束任务和非 Gold 满分任务，识别类别比较器或稀疏约束导致的假阳性。

通过离线不变性、边界测试和人工审计后，才允许启动 v4 训练。

## 7. GRPO 核心因果实验

### 7.1 共同设置

所有 GRPO 从同一个 `sft_n379_s288@144` checkpoint 开始，统一使用：

- seed `42`；
- 100 updates；
- temperature/top-p `0.7/0.9`；
- 最大环境步数 35；
- G=4、prompt batch=2、名义每次更新 8 条 Rollout；
- 动态采样开启，最多补采 3 批；
- 相同 LoRA、优化器、Advantage 归一化、KL 和硬件设置；
- 固定 step 100 作为主比较点，不从多个 checkpoint 中择优。

必须运行：

| 实验 | 环境 Reward | 优化 Reward | G | Updates | 目的 |
|---|---|---|---:|---:|---|
| `reward_v3_g4_s100` | v3 | 原始 v3 | 4 | 100 | 同条件因果基线 |
| `reward_v4_g4_s100` | v3 | adapter v4 | 4 | 100 | 检验 ASIN bonus 去除 |

先各运行 5-update smoke，确认 reward 路由、动态采样和 checkpoint 可恢复，再从全新输出目录开始正式100 updates；smoke checkpoint 不得继续用于正式结果。

### 7.2 成本与训练诊断

两组同时记录：

- 全同奖励组、全成功组、全失败组比例；
- Reward 方差、平均绝对 Advantage；
- 原始生成组数、有效保留组数和补采批次数；
- 实际 Rollout 数、有效 Rollout 数和输出 token 数；
- 每个有效 prompt group 的平均成本；
- repeat loop、max steps、Guard rejection/action attempts 和非法动作率；
- v3 strict success 保持率、v4 constraint-complete rate；
- 单步耗时、总耗时和峰值显存。

在 dry-run 中验证 veRL 的 prompt batch、rollout `n` 和 PPO mini batch 语义。不能只根据 `G × prompt_batch=8` 假定各组真实训练预算相同。

## 8. 条件式 G 与 KL 消融

只有 `reward_v4_g4_s100` 通过第 9 节的核心晋级门槛，才运行：

| 实验 | G | Prompt batch | PPO mini batch | 第一阶段 |
|---|---:|---:|---:|---:|
| `reward_v4_g2_pilot` | 2 | 4 | dry-run 后冻结 | 30 updates |
| `reward_v4_g8_pilot` | 8 | 1 | dry-run 后冻结 | 30 updates |

G2/G8 pilot 只能用于讨论早期训练效率，不能直接宣称100-update最终最优 G。如果 pilot 与 G4 差异明确且成本允许，应在查看新留出集之前预注册是否延长到100 updates；未延长则只报告30-update曲线和实际成本。

只有出现以下任一现象时，才运行 `reward_v4_g4_kl001`：

- 相对 SFT 的策略 KL 持续异常升高；
- SFT 已成功任务保持率明显下降；
- loop、Guard rejection 或平均步数明显恶化。

KL 对照使用 G4、相同初始化和100 updates；否则明确记录“未触发预注册条件”，不运行该实验。

## 9. Dev-50×3 配对评测与晋级门槛

### 9.1 统一协议

- 50 tasks × 3 attempts，固定分母150；
- actor seed 由 `(42, task_id, attempt_index)` 稳定派生；
- temperature/top-p `0.7/0.9`，max steps 35；
- 对照模型共享 task、attempt 和 actor seed；
- 基础设施错误、Reward 无效、footer failure、Guard rejection 和模型失败分开统计；
- Guard rejection 同时报告动作尝试分母和任务级分母。

输出：

- v3 strict gold success 与 Wilson 95% CI；
- v4 constraint-complete rate 与 Wilson 95% CI；
- 平均 v3/v4 Reward、平均步骤、loop 和 Guard；
- `pass@3` 与 `pass^3`；
- 任务级 win/tie/loss、10,000次 paired bootstrap 和精确 McNemar；
- 按约束数量、规格选择、预算、轨迹长度和有效软约束数分桶；
- 实际 Rollout、有效组和 token 成本。

paired bootstrap 以 task 为 cluster，不能把150次 attempt 当成150个独立任务。统计报告必须再次注明只有一个训练 seed。

### 9.2 核心 v4 晋级门槛

`reward_v4_g4_s100` 相对同条件 `reward_v3_g4_s100` 必须同时满足：

- attempt coverage 为100%；
- infrastructure invalid、`reward_valid=false` 和 footer failure 均为0；
- v4 constraint-complete rate 至少提高3pp；
- 任务级 paired bootstrap 95% 下界不低于0；
- v3 strict gold success 下降不超过2pp；
- 平均步骤增幅不超过10%，Guard rejection rate 不上升；
- loop、错误购买和无关动作没有异常增加；
- 实际生成 token 增幅不超过20%。

未通过时停止 v4 后续训练，保留负结果，不运行 G2/G8、KL 或新留出集。

## 10. Tool-choice 协议诊断

评测入口可以新增：

```text
--tool-choice auto|required
```

但 `required` 只作为协议压力测试，不能与 `auto` 混合构成主要训练收益结论。先在5个任务上确认当前 vLLM 版本支持该设置且终止流程正常。

Dev-50 至少比较 Base-auto 与 SFT-auto；Base-required 与 SFT-required 只用于判断强制工具调用能解释多少协议差异。分别报告合法工具调用、有效终局、Guard/action attempts、给定合法调用后的购买成功率、v3 strict success 和 v4 constraint-complete rate。

若 Base-required 改善，只能说明强制调用降低了协议门槛；它不能完全分离 JSON/工具协议能力与购物策略能力。

## 11. Final 评测边界

现有 `data/evaluation/tasks.jsonl` 的 Final-200 已经被查看并完成冻结确认。它不得因本计划产生的新模型而再次被称为未见、盲测或唯一最终确认集，也不得用来调参。

要升级为新的算法结论，必须在任何新结果产生前：

1. 获得一份来源清楚、与所有训练和 Dev 数据零重叠的新留出集；
2. 冻结 task ID、数据 SHA-256、评测脚本、模型清单和运行协议；
3. 将其记为新 holdout，而不是覆盖已有 Final-200；
4. 获得用户对该 200 题运行的明确执行授权。

新留出集最多运行：

- `sft_n379_s288@144`；
- `sft_n379_s288@288`；
- `reward_v3_g4_s100`；
- `reward_v4_g4_s100`，或在预注册扩展成立时唯一选定的 v4 配置。

每个模型只运行一次确定性 Rollout：temperature `0`、top-p `1`、max steps35、固定完整分母。看到结果后不得再调整超参数、prompt、reward、checkpoint 或 inference setting 并重跑。

如果无法建立新的未见留出集，则最终交付只能称为“Dev-50×3 固定单 seed 消融 + 已公开 Final-200 的回顾性项目结果”，不得宣称新的最终泛化提升。

## 12. 历史结果来源审计

`121→124` 与 `114→117` 来自不同 checkpoint、代码快照和运行批次，不预设它们是同一次运行中的“七题差异”。审计只做以下事情：

- 分别保留 run ID、模型/checkpoint/轨迹哈希和统计协议；
- 如果原始逐题产物存在，分别生成固定200题的四格迁移矩阵；
- 每个 run 内验证 `失败→成功 - 成功→失败 = 总成功数差`；
- 缺少同源逐题产物时直接标为不可配对，不跨 run 拼接；
- 现有 `9/185/6` 只属于 `114→117` 冻结确认，不能预填为未来 v4 结果。

该审计不需要 GPU，不阻塞核心新实验，但必须在最终文档中保持不同 run 的边界。

## 13. 分阶段运行安排

| 阶段 | 工作 | 预计 GPU | 停止条件 |
|---|---|---:|---|
| 0 | 版本/数据冻结、v4实现、测试、离线回放、dry-run | 约1小时 | 任一契约或恢复测试失败 |
| 1 | SFT n95/n190/n379@144 与 n379@288 | 12–15小时 | scheduler/global step 不连续 |
| 2 | v3-G4 与 v4-G4 smoke + 100 updates | 6–8小时 | Reward 路由、采样或恢复异常 |
| 3 | Dev-50×3 与统计 | 3–5小时 | v4 未达到晋级门槛 |
| 4 | 条件式 G2/G8，必要时 KL | 2–12小时 | 预注册触发条件不成立 |
| 5 | 冻结新留出集与一次评测 | 3–5小时 | 无新留出集或无明确授权 |

任何阶段出现 OOM、Ray/vLLM 启动失败、动态采样长尾或磁盘不足时，先保存 resolved config、stdout/stderr、最后可恢复 checkpoint 和硬件信息，再修复；不得把所有后续实验继续排队。

## 14. 验收标准

- 新冻结 commit、数据、配置、模型和轨迹 SHA-256 齐全；
- 训练/Dev/新留出集 task ID 零重叠；
- Environment Reward v3 和 strict success 契约未变化；
- v4 只通过独立字段进入训练，v3/v4 指标可以从同一轨迹同时重算；
- v4 通过 ASIN 不变性、边界、无效样本和稀疏约束审计；
- SFT 三组在 step144 使用相同优化设置，并报告实际 assistant tokens；
- More-SFT 的 optimizer、scheduler 和 global step 连续到288；
- v3-G4 与 v4-G4 除 optimization reward 外无其他变量变化；
- 所有 G 实验同时报告名义和真实 Rollout/token 成本；
- 模型选择只使用 Dev-50×3，固定 step100，不在多个 checkpoint 中择优；
- 缺失、错误、invalid 和 not-judged attempt 保留在固定分母中；
- 没有真实运行产物的能力只写“已实现/待验证”，不填写性能数字。

## 15. 简历与面试表述

真实完成后可以陈述：

- 在固定 `seed=42` 下，通过嵌套数据和等优化步数实验分析 SFT 轨迹多样性；
- 使用连续 More-SFT control 区分额外监督计算和在线 GRPO 收益；
- 在相同 SFT 初始化、G、训练步数和硬件下，对比 Reward v3 与 Target-ASIN-bonus-free terminal objective；
- 使用有效组比例、Advantage 方差、真实 token 成本和任务级配对统计解释 rollout 数选择；
- 同时报告约束满足收益、strict gold 保持率、策略回退和失败画像。

不得陈述：

- 多种子稳定提升；
- 已排除训练随机性；
- Reward 完全不使用 oracle；
- v4 constraint-complete 等同于现有 strict success；
- 在已查看的 Final-200 上得到新的盲测结论；
- 未达到预注册门槛或尚未运行完成的消融结果。

## 16. 明确排除项

本轮不引入 DPO、PPO 对照、Progress Reward、Evidence Memory、长度课程、Difficulty Curriculum、多训练 seed、历史兼容启动器或使用已查看 Final-200 调参。Judge API 不参与主要 Reward 计算，只用于入围模型的辅助失败审计。
