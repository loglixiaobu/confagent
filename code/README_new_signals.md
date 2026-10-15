# 固定旧步骤的新信号初筛

本轮判断自陈置信度和多采样一致性能否区分已清洗的正确／错误步骤。复用 `outputs/legacy_binary_v2/calls_binary.jsonl` 的 198 个调用及原 logprob，不执行新工具，不重做标签，不训练校准器。完整结果位于 [`../outputs/new_signals_fast_v1/新信号评估.md`](../outputs/new_signals_fast_v1/新信号评估.md)。

198 步均取得两个新分数，标签为 158 正确、40 错误。原 logprob、自陈和一致性的 AUROC 分别为 0.6351、0.7172、0.8812；优先检查最低分 20% 时，错误捕获率分别为 29.0%、33.4%、68.7%。同分跨过预算时按随机抽查的期望计算。一致性去重后 AUROC 为 0.8793；配对任务 bootstrap 支持其优于原 logprob，自陈的提升仍不确定。

## 两个分数的定义

- **自陈置信度**：向 Qwen3-8B 提供需求、工具文档、此前调用与实际结果，以及原候选调用；关闭思考，以温度 0 请求一个 0–1 数字。不提供当前调用的结果、隐藏参考或修订标签。
- **多采样一致性**：在重建的调用前状态独立生成 3 次后续动作，温度 0.7、top_p 1、关闭思考。采样请求不展示原候选调用。分数是与原调用等价的样本数除以 3，取值为 0、1/3、2/3、1；不是新样本的多数派比例。
- 主一致性要求相同 API 与语义参数，规范化数值、时间、安全算术和个人地址末尾句点。`EmailReminder.content` 的自由文本措辞不参与主匹配，其他字段必须一致；另报保留全部字段的严格一致性。
- 无工具输出或未解析出动作记为不一致；截断单列。响应含多个调用时只取首个，沿用原运行器口径。请求失败以同种子重试一次；仍缺样本时整步一致性保留缺失，不缩小分母。自陈格式不合法也保留缺失。

原日志缺少完整助手正文、工具调用 ID 和原始参数字符串，历史助手正文用空字符串重建，并恢复重复调用提示。重建保留原工具 schema，包括当时 `list(str)` 被错误暴露成字符串的问题。当前调用的执行结果和修订标签不进入模型请求；此前实际观察到的旧环境反馈仍保留。

因此，这些是**重建状态下的自评和一致性**。原 logprob 来自旧思考策略生成原动作时的分数，新一致性来自关闭思考后的重新采样，两者的比较是探索性诊断，不能把差异全部归因于信号机制。本轮没有独立留出验证，也没有拟合概率校准器；稳定重复错误仍可能得到高一致性，多个合理后续动作也可能降低一致性。

## 代码与测试

| 文件 | 作用 |
|---|---|
| `collect_new_signals.py` | 重建调用前状态、补采两个分数，缓存请求响应并支持相同协议续跑；不执行 API |
| `evaluate_new_signals.py` | 关联冻结标签，计算覆盖率、同样本 AUROC、配对任务 bootstrap、错误捕获率、去重与历史敏感性，并生成图和报告 |
| `../tests/test_new_signal_collection.py` | 检查当前结果和标签隔离、历史重建、分数解析、语义动作匹配 |
| `../tests/test_new_signal_evaluation.py` | 检查缺失处理、共同样本、任务聚类重采样、配对差和同分预算统计 |
| `../tests/test_published_new_signals.py` | 检查公开分数、精简动作、状态和冻结标签完整对应，并复核一致性及主要 AUROC |

## 公开产物

以下文件均位于 `outputs/new_signals_fast_v1/`：

| 文件 | 内容与用途 |
|---|---|
| `scores.jsonl` | 198 步的自陈分数、一致性、严格一致性及采样质量计数；不改原标签 |
| `sample_actions.jsonl` | 按步骤保存自陈原始文本及 3 次采样的动作、结束原因、调用数量和种子，用于核查打分 |
| `states.jsonl` | 模型可见的调用前状态、工具文档及原候选；不含修订标签或当前调用的返回结果 |
| `protocol.json` | 冻结信号定义、采样参数、数据和代码来源摘要 |
| `new_signal_metrics.json` | 完整覆盖率、AUROC、配对区间、去重与历史敏感性等统计 |
| `new_signal_metrics.csv` | 各信号在不同队列的指标，便于表格查看 |
| `joined_scores.csv` | 按任务和步骤关联冻结标签、原 logprob、新分数及分组标志 |
| `new_signal_comparison.png` | 三信号对比图 |
| `新信号评估.md` | 中文结果、方法与适用范围 |
| `collection_summary.json` | 请求数量、采样质量与 token 用量汇总 |
| `validation.json` | 公开产物完整性、测试和复算校验记录 |

公开版本保留研究所需的状态和精简采样动作。重新采集时产生的完整请求响应缓存留在本地，用于恢复运行和检查传输问题。

## CPU 复算现有指标

从仓库根目录运行。已有冻结标签与分数即可复算，不调用模型、不需要 GPU，也不依赖完整 API-Bank 数据。

```bash
pip install -r requirements-analysis.txt
python code/evaluate_new_signals.py \
  --labels outputs/legacy_binary_v2/calls_binary.jsonl \
  --scores outputs/new_signals_fast_v1/scores.jsonl \
  --out-dir outputs/new_signals_recomputed \
  --bootstrap 1000 --seed 20261015
```

新目录用于保存复算图表与报告，方便对照仓库中已发布的结果。任务 bootstrap 保留每个任务内的全部步骤；比较新旧信号时使用相同任务重采样副本。

## 重新采集

可以直接使用公开状态重新采集，无需完整 API-Bank 数据；须自行准备支持工具调用的 Qwen3-8B OpenAI 兼容服务。采集依赖：

```bash
pip install openai httpx
```

确认服务已启动后，从仓库根目录运行：

```bash
python code/collect_new_signals.py \
  --states outputs/new_signals_fast_v1/states.jsonl \
  --out-dir outputs/my_signals \
  --endpoints http://localhost:8000/v1 \
  --no-consistency-thinking --cooldown 1
```

若要从原轨迹重新构造状态，才需要准备 API-Bank 数据。`PATH_TO_API_BANK` 应包含 `test-data/level-3.json`；以下命令仅准备状态，不请求模型：

```bash
python code/collect_new_signals.py \
  --out-dir outputs/my_reconstructed_states \
  --data-dir PATH_TO_API_BANK \
  --no-consistency-thinking --prepare-only
```

这两个命令使用新目录，避免覆盖已发布的冻结结果。采集会发起模型推理，但不执行采样出的工具调用。相同协议与本地完整请求缓存可续跑；数据、模型可见状态或采集代码变化时应使用新目录。服务启动和硬件资源管理由运行者配置。

模型采样存在随机性，服务版本和运行设置也会影响重新采集的数值。精确复核本次统计应使用已发布的冻结分数与精简动作。
