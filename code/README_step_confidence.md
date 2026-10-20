# 步骤级置信度：最小可用接口与离线复算

本轮将一致性信号接成可调用接口：输入调用前上下文与候选工具调用，独立补采三个动作，在本地比较语义一致性，再用冻结 sigmoid 输出 `p_correct` 和 `needs_verification`。接口不执行工具。

| 代码 | 作用 |
| --- | --- |
| `confidence_runtime.py` | 在线评分接口、JSON 校准器加载、纯 CPU 概率映射 |
| `calibrate_consistency.py` | 在旧 198 步上做固定五折、按任务分组的 OOF 校准，再冻结部署参数 |
| `evaluate_minimal_validation.py` | 对公开的新任务记录重新连接标签、比较概率质量、统计验证预算与成本 |

结果均在 [`outputs/confidence_minimal_v1/`](../outputs/confidence_minimal_v1/)：

| 文件 | 含义 |
| --- | --- |
| `calibrator.json` | 单变量 sigmoid 参数、K=3 采样定义、验证规则及公开输入哈希 |
| `oof_predictions.jsonl` / `.csv` | 旧步骤的折号、折外概率、训练折先验和验证选择 |
| `calibration_metrics.json`、`一致性校准报告.md`、`calibration_reliability.png` | 内部开发数据上的概率指标、可靠性图和限制 |
| `validation_steps.jsonl` | 50次调用的候选、实际观测、调用前消息、工具文档、logprob摘要、三个采样动作和置信度 |
| `validation_labels.jsonl`、`validation_labels_original.jsonl`、`validation_label_review.json` | 最终二元标签、原标签及屏蔽分数后的审查修订依据 |
| `validation_tasks.jsonl`、`validation_task_review.jsonl` | 所选任务的参考资料和任务结束审查证据；供环境／评估使用，不应作为模型输入 |
| `validation_trajectories.jsonl`、`validation_task_outcomes.jsonl` | 全部30个任务的停止信息、最终输出、独立于调用标签的任务／结束标签 |
| `validation_protocol.json`、`validation_complete.json` | 采集约定、完成数量和公开文件校验信息 |
| `collection_summary.json` | 230次请求及主轨迹／补采 token 汇总，替代原始请求日志 |
| `validation_metrics.json`、`新任务验证报告.md`、`validation_diagnostic.png` | 新任务上的覆盖率、概率质量、检查分配与局限 |
| `runtime_example_input.json` / `runtime_example_output.json` | 一次实际评分的最小输入与输出 |
| `publication.json` | 原始实验哈希与实际公开文件哈希、压缩变更和离线数值一致性说明 |

## 无需模型的复算

在仓库根目录安装分析依赖后运行：

```bash
python -m pip install -r requirements-analysis.txt
python code/calibrate_consistency.py
python code/evaluate_minimal_validation.py
```

以上仅使用 CPU 和仓库内的已发布记录，不需要完整 API-Bank 数据集，不发起模型请求。校准程序会拒绝覆盖参数／输入不一致的冻结模型；若更换依赖后出现浮点拟合差异，请用 `--out-dir outputs/my_calibration` 保留单独结果。

仅演示概率映射，连分析依赖也不需要：

```bash
python -S code/confidence_runtime.py \
  --calibrator outputs/confidence_minimal_v1/calibrator.json \
  --raw-consistency 0.3333333333333333
```

输出 `p_correct≈0.66748`、`needs_verification=true`；三次全一致时约为0.94545。它们是依据旧开发数据估计的局部调用正确概率，不是任务成功保证。

## 接入智能体

在线使用需安装 `openai`、`httpx` 并自行提供 OpenAI 兼容模型端点。固定模型为 Qwen3-8B，K=3、temperature=0.7、top_p=1、max_tokens=4096，补采关闭思考。

```python
import json
import sys

sys.path.insert(0, "code")
from confidence_runtime import estimate_step_confidence

with open("outputs/confidence_minimal_v1/runtime_example_input.json", encoding="utf-8") as source:
    state = json.load(source)

result = estimate_step_confidence(
    **state,
    calibrator="outputs/confidence_minimal_v1/calibrator.json",
    base_url="http://localhost:8000/v1",  # 替换为自行配置的端点
    request_namespace="my-task:step-1",
)
print(result["p_correct"], result["needs_verification"])
```

也可注入 `client` 或 `request_fn(**payload)`，以及 `before_request()` / `after_request(attempt_record)` 资源门控回调。接口在当前进程中顺序请求，每次尝试至少冷却一秒；端点的资源管理由调用者负责。

候选仅在本地比较，不追加进采样请求；当前调用结果、参考答案、标签不可出现在输入中。每个候选额外付出三次模型请求，**20%的检查预算不代表20%的评分成本**。原始请求遇传输失败重试一次，样本仍缺失时原分数与概率均为 `null`，建议验证；无工具响应算不一致，多工具只取首个调用并保留质量标志。

验证规则固定为 `raw_consistency < 1`，缺失也触发。当前仅输出建议，不改变候选或后续轨迹。最终回答与无工具结束事件尚未训练置信度模型，应另外判断；`calibrate_consistency(None, calibrator)` 可表达缺失概率。

## 结果应如何理解

旧开发数据为198步、158正40负，47个有调用任务。按任务 OOF：Brier 0.1235→0.0917，ECE 0.1347→0.0674。信号曾在同批旧数据上筛选，OOF仍是内部开发诊断，不能视作独立外部测试；不同折的概率映射也会改变合并后的 AUROC。

冻结模型在额外30任务上接通：26个任务产生50次调用，49正1负；任务交付独立复核为22合格、8失败。仅1个错误导致 AUROC／捕错率很不稳定。新任务校准 Brier 0.0599优于原始一致性的0.1600，但差于旧训练先验的0.0527，**不能宣称跨分布校准成功**。不同任务的成功率也不能直接比较为能力提升。

本批共230次请求：主轨迹80次／89,869 token，置信度补采150次／92,383 token，总182,252 token。无实际验证纠错或经验更新实验。标签来自屏蔽分数的 Codex 模型代理审查，不是独立人工双标。

公开上下文保留原语义，用稳定任务内ID替换生成的工具调用ID；采样保留动作、种子和判定标志，删除重复长响应。部署参数使用完全相同的旧标签与分数在CPU重新生成，使溯源匹配公开协议：参数、OOF预测和全部研究指标不变，不使用新任务标签拟合，也没有重采模型。
