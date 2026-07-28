# Reward v3 / fresh-v1 GRPO 执行契约

这条实验线把 Environment v2.1 / Reward v3 / fresh-v1 视为新的后训练起点。
旧 SFT v3 probe、`grpo_*_v1` split/parquet、旧 GRPO checkpoint 和旧 launcher
只保留为归档。

## 冻结输入

- SFT 初始策略：
  `/root/autodl-tmp/checkpoints/qwen35-2b-sft-v1-fresh-merged`
- merge 来源：fresh-v1 SFT `checkpoint-141`
- Environment：`shopsimulator-environment-v2.1`
- Reward：`shopsimulator-reward-v3`
- Observation：`shopping-observation-v2`
- Tools：`shopping-tools-v2`
- max environment steps：35
- context：24,576；单回合生成上限：512

fresh-v1 SFT train 的正确 SHA256 是
`8cd1f72130b3c781d5ffe08fe3e399b2a9e45d204e3f3bd0d8e677d1b51c8ec5`。
旧契约值只有 62 位，不能作为校验值。validation SHA256 是
`f8ae506d0fa9d1526342a9f717da24922c8a55776d076a296698abac4fde05b3`。

## 数据生成

先生成 2,000 个全新候选 task：

```bash
cd /root/autodl-tmp/shopping-grpo-longhorizon
PYTHONPATH=src .venv-grpo-v080/bin/python \
  scripts/prepare_grpo_reward_v3_fresh_v1.py candidate
```

候选池会排除：

1. fresh-v1 全部 604 个 raw task；
2. benchmark v1 和 benchmark v2_50；
3. 历史 Teacher raw task；
4. 历史 `grpo_probe_pool_v1`。

启动 v2.1 ShopSimulator 和 fresh merged vLLM 后，执行可断点续跑的 8-worker
probe：

```bash
PYTHONPATH=src .venv-grpo-v080/bin/python \
  scripts/probe_grpo_reward_v3_fresh_v1.py
```

probe 固定 temperature=0、top_p=1、max_steps=35、max_tokens=512，并校验每条
轨迹的 `environment_version`；所有正常终局必须携带 Reward v3。完成后按实际
short/medium/long 分布比例冻结 1,000 train 和 50 validation：

```bash
PYTHONPATH=src .venv-grpo-v080/bin/python \
  scripts/prepare_grpo_reward_v3_fresh_v1.py select
PYTHONPATH=src .venv-grpo-v080/bin/python \
  scripts/prepare_grpo_reward_v3_fresh_v1.py validation
PYTHONPATH=src .venv-grpo-v080/bin/python \
  scripts/prepare_grpo_reward_v3_fresh_v1.py audit
```

ShopSimulator 仍在线时生成 parquet：

```bash
bash scripts/build_grpo_reward_v3_fresh_v1_parquet.sh
```

parquet 只包含 system prompt、用户可见 instruction 和 task_id，不包含 goal、
标准答案、reward_detail 或 Teacher trajectory。

## 正式启动边界

正式启动只允许使用：

```bash
export GRPO_OUTPUT_DIR=/root/autodl-tmp/checkpoints/<new-reward-v3-run>
bash scripts/run_grpo_reward_v3_fresh_v1.sh a0
```

或经明确实验决策使用 `a1`。新 launcher 强制：

- Reward v3 environment manifest；
- v2.1 AgentLoop；
- v2 tool schema；
- fresh merged SFT；
- fresh Reward v3 parquet；
- 固定 `.venv-grpo-v080`；
- 新 checkpoint 目录。

旧 `scripts/run_vanilla_grpo.sh` 默认拒绝启动。未经用户明确授权，不执行上述正式
GRPO 命令。
