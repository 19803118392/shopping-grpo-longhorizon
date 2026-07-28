# Shopping Agent Post-Training

面向 ShopSimulator 的端到端购物 Agent 后训练项目，覆盖 Teacher Rollout、LoRA
SFT、veRL GRPO 和离线轨迹评测。

当前正式实验世代固定为：

```text
ShopSimulator Environment v2.1
→ Reward v3
→ fresh-v1 Teacher / SFT
→ reward_v3_fresh_v1 GRPO
→ trajectory-evaluation-v1
```

仓库中的旧 Reward、旧数据切分、旧 benchmark 和旧实验记录仅用于复现与数据排除
溯源，不是当前训练或评测入口。

## 当前能力

- 内嵌、版本化的 ShopSimulator Environment v2.1；
- DeepSeek Teacher Rollout、断点续跑和规则验收；
- Qwen3.5 工具调用格式的 Action-only LoRA SFT；
- Reward v3、自定义终局和约束感知奖励；
- veRL GRPO、A0/A1 配置和 Dynamic Sampling；
- Base、SFT、GRPO 的统一离线 Rollout；
- 需求 Rubric、五维轨迹 Judge、确定性指标和配对比较；
- 冻结的 Reward v3 final 200 盲测任务。

## 仓库结构

- `environments/ShopSimulator/`：Environment v2.1、Reward v3、商品数据和测试；
- `src/shopping_grpo/`：Collector、工具协议、veRL 适配和离线评测模块；
- `configs/verl/vanilla_grpo_reward_v3_fresh_v1.yaml`：当前 GRPO 配置；
- `scripts/run_grpo_reward_v3_fresh_v1.sh`：当前正式 GRPO 启动入口；
- `scripts/build_trajectory_evaluation_artifacts.py`：纯离线评测 artifact 构建；
- `scripts/run_trajectory_evaluation_models.py`：Rubric Flash 与轨迹 Pro Judge；
- `data/splits/grpo_reward_v3_fresh_v1_*`：当前冻结 train/validation；
- `data/benchmarks/shop_benchmark_reward_v3_final_200.jsonl`：冻结 final test；
- `docs/grpo-reward-v3-fresh-v1.md`：当前训练实验契约；
- `docs/plans/2026-07-28-shopping-trajectory-evaluation-pipeline.md`：评测契约与阶段计划。

## 安装与环境

基础数据处理环境：

```bash
python3 -m pip install -r requirements.txt
cp .env.example .env
set -a
. ./.env
set +a
```

`.env` 不会被 Python 自动加载，也不得提交到 Git。Teacher 与 Judge 的 endpoint 和
API key 均从环境变量读取。

首次准备内嵌 ShopSimulator：

```bash
bash scripts/setup_embedded_shopsimulator_v2.sh
```

启动 Environment v2.1 服务：

```bash
source environments/ShopSimulator/.venv-shopsim-v2/bin/activate
cd environments/ShopSimulator/shop_env
SHOPSIM_ENV_SLOTS=8 SHOPSIM_PORT=5700 ./run_environment_v2.sh
```

环境源码快照、依赖和运行边界见
[`environments/ShopSimulator/README.md`](environments/ShopSimulator/README.md)。

## Teacher Rollout 与 fresh-v1 SFT

导出无隐藏目标信息的任务 ID 清单：

```bash
PYTHONPATH=src environments/ShopSimulator/.venv-shopsim-v2/bin/python \
  scripts/export_shop_task_ids.py \
  --shopsim-root environments/ShopSimulator/shop_env \
  --output data/shop_tasks.jsonl
```

采集示例：

```bash
PYTHONPATH=src python3 scripts/collect_teacher_rollouts.py \
  --tasks data/shop_tasks.jsonl \
  --output outputs/teacher/raw.jsonl \
  --base-url http://127.0.0.1:5700 \
  --model deepseek-v4-flash \
  --thinking --reasoning-effort high \
  --limit 10 --max-steps 35
```

构建验收后的 SFT 数据：

```bash
PYTHONPATH=src python3 scripts/build_sft_data.py \
  --raw outputs/teacher/raw.jsonl \
  --accepted outputs/teacher/accepted.jsonl \
  --rejected outputs/teacher/rejected.jsonl \
  --stats outputs/teacher/stats.json \
  --sft outputs/teacher/sft.jsonl
```

LoRA SFT 使用 `scripts/split_sft_data.py`、`scripts/inspect_sft_data.py` 和
`scripts/train_lora_sft.py`。当前 GRPO 的初始策略是 fresh-v1 SFT
`checkpoint-141` 合并模型；精确数据 hash 和模型边界记录在
[`docs/grpo-reward-v3-fresh-v1.md`](docs/grpo-reward-v3-fresh-v1.md)。

## Reward v3 / fresh-v1 GRPO

当前正式入口只有：

```bash
bash scripts/run_grpo_reward_v3_fresh_v1.sh a0 --dry-run
bash scripts/run_grpo_reward_v3_fresh_v1.sh a1 --dry-run
```

正式启动前设置新的 checkpoint 目录和 SwanLab key：

```bash
export GRPO_OUTPUT_DIR=/absolute/path/to/new-reward-v3-run
export SWANLAB_API_KEY=...
bash scripts/run_grpo_reward_v3_fresh_v1.sh a0
```

- `a0`：native Reward v3、标准 GRPO advantage normalization；
- `a1`：constraint-aware Reward v3、Dr.GRPO、Dynamic Sampling。

launcher 固定 Environment v2.1 manifest、Reward v3 数据、fresh-v1 SFT、
Qwen3.5 tool schema 和 `.venv-grpo-v080`。完整准备、资产 SHA256 和运行边界见
[`docs/grpo-reward-v3-fresh-v1.md`](docs/grpo-reward-v3-fresh-v1.md)。

## trajectory-evaluation-v1

评测强绑定当前 ShopSimulator Rollout，不计算综合总分，分别报告：

1. Environment Reward 与终局；
2. Query 需求 Rubric；
3. 五维轨迹质量；
4. 效率、重复、合法性、上下文和异常指标。

Judge 只接收 Actor 实际可见的轨迹、白名单终局字段和行为指标。Reward v3
分数、成功结论、隐藏 Gold 和 `reward_detail.evidence` 均在数据层隔离，最终仅由
代码把四个独立面板拼装到报告中。

离线预处理示例：

```bash
PYTHONPATH=src python3 scripts/build_trajectory_evaluation_artifacts.py \
  preprocess \
  --raw outputs/eval/raw.jsonl \
  --output outputs/eval/preprocessed.jsonl
```

Rubric 和 Judge 的完整 artifact 顺序、schema、缓存规则和正式三模型协议见
[`docs/plans/2026-07-28-shopping-trajectory-evaluation-pipeline.md`](docs/plans/2026-07-28-shopping-trajectory-evaluation-pipeline.md)。

冻结 final test 位于：

```text
data/benchmarks/shop_benchmark_reward_v3_final_200.jsonl
```

它在 checkpoint、Prompt 和 Judge 版本冻结前不得用于调试、校准或模型选择。

## 测试

离线评测模块不启动模型、环境或 GPU：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m unittest discover -s tests \
  -p 'test_trajectory_evaluation.py' -v
```

其余环境、Reward、Collector 和 GRPO 测试按对应文档的固定 Python 环境运行。

## 历史资产

旧 Reward v1/v2、旧 `grpo_*_v1` 数据、旧 benchmark 和历史 A0/A1 记录不属于当前
正式链路。部分文件仍保留原路径，是因为 Environment 兼容导入、历史实验复现以及
final-test exclusions metadata 仍引用它们；不要把这些资产改名后混入 Reward v3
结果。历史实验说明集中在 [`docs/experiments/`](docs/experiments/)。
