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
- 三次连续完全相同动作和连续四步无新 ASIN 的终止规则；
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

目标 ASIN 可以使用已有 Gold 信息确定性评分。非目标商品只有在任务提供：

```json
{
  "hard_constraints": {
    "complete": true,
    "brand": [],
    "model": [],
    "core_functions": [],
    "key_specs": []
  },
  "weighted_preferences": []
}
```

并且商品元数据足够核验时，才可能得到 `valid_alternative_purchase`。
当前原始任务没有完整的硬约束/软偏好分类，因此非目标购买会保守标记为
`reward_unverifiable`，不会被误当成成功或普通失败进入 GRPO。正式生成
Environment v2 Teacher 数据前，必须冻结这份任务约束元数据；实现不会根据
目标商品或模型行为临时猜测分类。

金额解析支持“1万2以内”等常见写法。“预算 1 万元左右”不是严格上限，
v2 固定使用 10% 容差；该规则必须在正式数据生成前随 Reward 配置一起冻结。
