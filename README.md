# ConfAgent

ConfAgent 用 API-Bank 工具定义研究调用级置信度和跨会话事实记忆。记忆实验使用 Qwen3-8B，在相同请求和工具下配对比较下一步动作。

**复用已授权事实减少了重复追问，并提高了有依据的调用率。** V9 同时需要两个已保存字段，正确调用从 **0/24 升至 24/24**，追问任一已保存字段从 **19/24 降至 0/24**。

## 实验结果

| 轮次 | 实验与结果 |
| --- | --- |
| [V1](results/memory/v1/报告.md) | 候选核验决策 4/4 正确，规则实际注入 0/3，实验止于入库。 |
| [V2](results/memory/v2/报告.md) | 经验规则对照。原题编造率 0/10 → 0/10，压力题 3/10 → 3/10。 |
| [V3](results/memory/v3/报告.md) | 两次归纳规则各修复 0/3 个开发错误，实验止于开发。 |
| [V4](results/memory/v4/报告.md) | 加入过往错误与正确处理作为实例记忆。留出题编造率 15/24 → 15/24。 |
| [V5](results/memory/v5/报告.md) | 复用已保存事实。重复追问 14/24 → 0/24，正确调用 0/24 → 24/24，达到预设判据。 |
| [V6](results/memory/v6/报告.md) | 增加更新、撤回和历史重放对照。更新题正确调用 0/20 → 19/20，无效动作护栏未通过。 |
| [V7](results/memory/v7/报告.md) | 各组共用证据说明。重复追问 14/16 → 5/16，更新题正确调用 0/20 → 8/20，两项均未达预设判据。 |
| [V8](results/memory/v8/报告.md) | 记忆附上字段匹配说明和用户授权原话。重复追问 14/16 → 0/16，更新题正确调用 0/20 → 20/20，两项均达到预设判据。 |
| [V9](results/memory/v9/报告.md) | 同时复用两条独立授权记录。正确调用 0/24 → 24/24，单侧精确配对检验 p = 5.96×10⁻⁸，8 道安全控制题全部通过。 |

V8 的探索性对照使用同一个最新值。附上匹配说明和授权原话的组正确调用 36/36，仅提供字段和值的组为 27/36。实验使用模拟场景和工具返回，结论限于已知用户及字段、且授权可核验的单步任务。逐题标签、汇总和图表见 [公开结果](results/memory)。

前期置信度实验复核了 198 次调用。原始 logprob 的 AUROC 为 0.6351，采样一致性为 0.8812。校准改善了交叉验证的 Brier 和 ECE，新域 Brier 未优于常数先验。详见 [置信度接口和评测](code/README_step_confidence.md)。

## 本地复现

准备本地 Qwen3-8B 模型，并将 [API-Bank 数据](https://huggingface.co/datasets/liminghao1630/API-Bank) 的 `test-data/level-3.json`、`test-data/level-1-response.json` 和 `training-data/lv3-response-train.json` 按原目录放在 `data/api-bank/` 下。仓库随附各轮的模拟题清单。

```bash
python -m pip install -r requirements-memory.txt vllm
export MODEL_PATH="$PWD/models/Qwen3-8B"
bash code/launch_replicas.sh '0 1' 18000

python code/run_memory_multislot.py --model-path "$MODEL_PATH" \
  --replica-base-url http://127.0.0.1:18001/v1 --output runs/v9
python -m unittest discover -s tests -p 'test_memory*.py'
bash code/stop_replicas.sh 0 1
```

V2–V9 的入口见 [code/](code)。V2 使用仓库已有的历史评测证据。V3/V4 读取 V2 默认输出目录里的开发材料，V4 还核验 V3 的留出集未运行。V5–V9 可直接从模拟数据重跑。公开结果保留指标和逐题标签，运行日志和本地协议放在各自的运行目录。

## 后续计划

1. **组员一｜补充双字段边界题。** 参考 V9 补 8 道模拟题，覆盖两项均有效、只缺一项、撤回或身份不匹配、当前值覆盖旧值。每题标明可见的字段依据、期望动作及模拟候选，交付一份 JSON 给组员二。[V9 题库](data/memory_multislot_cases.json)
2. **组员二｜补上调用前参数核验。** 承接置信度接口建议的参数来源检查，复用 V9 的授权和字段匹配规则。候选参数有依据就放行，能从当前请求或有效记忆确认取值就纠正错参，仍缺值就只追问相应字段。用同一批候选离线比较核验前后的无依据调用、正确调用和重复追问次数，交付一个脚本和一张结果表。[置信度接口](code/README_step_confidence.md)

用手写候选即可在 CPU 上开始回放。标准答案只用于评分。复测发现的新错误补回题库，再跑一遍回归。
