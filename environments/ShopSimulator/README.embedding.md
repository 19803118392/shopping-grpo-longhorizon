# Embedded ShopSimulator

该目录是本训练仓库使用的购物环境源码快照，不是 Git submodule。来源和精确
来源 commit 记录在 `EMBEDDED_SOURCE.json`；内嵌后的精确文件内容由当前训练
仓库 commit 冻结。

保留内容包括 Environment v2 服务、搜索、Observation、Reward、终止规则、
测试、页面模板和压缩商品数据。虚拟环境、解压后的 140MB JSON、生成的搜索
索引、日志和缓存不会提交 Git。

从训练仓库根目录执行：

```bash
bash scripts/setup_embedded_shopsimulator_v2.sh
```

启动服务：

```bash
screen -dmS shopsim-v2 bash -lc '
  set -euo pipefail
  source /root/autodl-tmp/shopping-grpo-longhorizon/environments/ShopSimulator/.venv-shopsim-v2/bin/activate
  cd /root/autodl-tmp/shopping-grpo-longhorizon/environments/ShopSimulator/shop_env
  mkdir -p shop_env/logs
  SHOPSIM_ENV_SLOTS=8 SHOPSIM_PORT=5700 ./run_environment_v2.sh \
    > shop_env/logs/shopsim_v2_5700.log 2>&1
'
```

详细设计和验证说明见 `shop_env/ENVIRONMENT_V2.md`。
