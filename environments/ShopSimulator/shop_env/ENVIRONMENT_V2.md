# ShopSimulator Environment v2

Environment v2 是一个显式启用、可回退的购物文本环境。旧实验默认仍使用
`lucene_v1` 和旧 Reward；只有设置 v2 环境变量时才切换。

## 已实现

- SQLite FTS5 多字段 BM25，字段包括标题、品牌/店铺、类目、型号、属性、规格和短卖点；
- 索引 Manifest 记录 Python 与 SQLite 精确版本；字节级复现要求数据、代码和
  构建工具链一致；
- NFKC、大小写、空白、标点和基础金额归一化；
- 固定 Top 150、每页 20 条、ASIN 同分排序；
- 同一次查询的分页结果缓存，不在翻页时重新检索；
- 结构化公开 Observation，不接受 goal、target ASIN 或 Reward 字段；
- `gold_purchase`、`valid_alternative_purchase`、主动结束、错误购买、循环和最大步数 Reward；
- `reward_unverifiable` 与基础设施异常分离；
- 三次连续完全相同动作和连续四次无新候选的探索动作终止规则；
- 启动时加载并严格校验 `configs/environment_v2.json`，拒绝搜索、Reward、
  分页或终止阈值与代码/索引漂移；
- 8 个环境槽共享商品库和只读索引，但每个槽使用独立 session；
- 代码级任务可达性审计。

## 构建

所有大文件位于 `/root/autodl-tmp`。索引不提交 Git。

```bash
cd /root/autodl-tmp/shopping-grpo-longhorizon

screen -dmS shopsim-v2-setup bash -lc '
  set -euo pipefail
  cd /root/autodl-tmp/shopping-grpo-longhorizon &&
  bash scripts/setup_embedded_shopsimulator_v2.sh \
    > outputs/shopsim_v2_setup.log 2>&1
'
```

索引构建产物：

```text
shop_env/search_engine/environment_v2.sqlite3
shop_env/search_engine/environment_v2.manifest.json
```

## 启动

```bash
screen -dmS shopsim-v2 bash -lc '
  set -euo pipefail
  source /root/autodl-tmp/shopping-grpo-longhorizon/environments/ShopSimulator/.venv-shopsim-v2/bin/activate &&
  cd /root/autodl-tmp/shopping-grpo-longhorizon/environments/ShopSimulator/shop_env &&
  mkdir -p shop_env/logs &&
  SHOPSIM_ENV_SLOTS=8 SHOPSIM_PORT=5700 ./run_environment_v2.sh \
    > shop_env/logs/shopsim_v2_5700.log 2>&1
'
```

启动脚本固定：

```text
SHOP_ENVIRONMENT_VERSION=shopsimulator-environment-v2
SHOP_SEARCH_BACKEND=multifield_bm25_v2
SHOP_MAX_STEPS=35
```

## CPU 验证

```bash
cd /root/autodl-tmp/shopping-grpo-longhorizon/environments/ShopSimulator
PYTHONPATH=shop_env python3 -m unittest discover -s shop_env/tests -v

PYTHONPATH=shop_env python3 shop_env/scripts/audit_environment_v2.py \
  --index shop_env/search_engine/environment_v2.sqlite3 \
  --task-id 4918 \
  --output shop_env/search_engine/environment_v2.audit.json
```

审计只证明目标商品、索引、详情、规格、价格、购买路径和 Gold Reward
没有损坏，不代表当前 2B 模型一定能通过自然语言检索完成任务。

## Reward 元数据边界

Environment v2 会从**当前 instruction 自己的结构化标注**生成约束契约：

```json
{
  "hard_constraints": {
    "complete": true,
    "contract_version": "shopping-task-constraints-v1",
    "annotation_source": "instruction.attributes",
    "brand": [],
    "model": [],
    "core_functions": ["任务 attributes 中已有的要求"],
    "key_specs": [],
    "annotated_option_count": 1
  },
  "weighted_preferences": []
}
```

类目和预算由既有独立硬门禁检查，`instruction_options` 由
`key_options` 门禁检查，当前 instruction 的 `attributes` 作为
`core_functions` 检查。编译器不会读取目标商品的额外隐藏字段，也不会根据
关键词猜测品牌、型号或软偏好；因此 `weighted_preferences` 暂为空。

当 instruction 缺少 `attributes`/`instruction_options` schema，或候选商品
元数据不足以核验已声明约束时，非目标购买仍保守标记为
`reward_unverifiable`。只有约束契约完整且所有门禁通过时，非目标商品才得到
`valid_alternative_purchase`。CPU 全量检查确认当前 23,421 条有效任务均能
生成完整契约。

金额解析支持“1万2以内”等常见写法。“预算 1 万元左右”不是严格上限，
v2 固定使用 10% 容差；该规则必须在正式数据生成前随 Reward 配置一起冻结。

## 循环检测边界

“连续四步没有新 ASIN”实际按**有候选发现意义的动作**统计：

- 搜索和翻页：只有出现未见过的 ASIN 才算新进展；
- 打开商品：第一次打开该 ASIN 算进展，重复打开旧候选算无进展；
- Description、Features、Attributes、规格选择和 Back to Search：
  不增加无进展计数。

完全相同动作的连续重复检测保持独立。这样仍能终止重复搜索和反复打开旧候选，
但正常查看一个商品的详情、卖点和属性不会被误判为循环。
