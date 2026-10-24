# 置信度驱动的智能体递归自我改进

**首次了解项目：[三分钟说明](mid_presentation/项目速读.md)**——任务、九张图、核心代码、指标、当前进展，以及下一步设计和验证方案。

本仓库研究一个核心问题：**如何让基于工具调用的 LLM 智能体在多步执行过程中，估计每一步的置信度、据此自主触发验证与纠错，并把纠错经验递归地沉淀回后续执行。**

研究对象是 **API-Bank level-3 多步工具调用任务**，主模型为 **Qwen3-8B**（本地 vLLM 部署）。本项目**初始测试的是token 级 logprob** 作为置信度信号源，而 vLLM 的 OpenAI 兼容接口原生支持 `logprobs`，也可以选择其他的信号

## 最新发布：步骤置信度最小模块与中期汇报（2026-10-20）

在已有二元标签清洗、logprob 重评和新信号初筛的基础上，本次上传了**一致性概率校准、冻结后新任务验证，以及对全部实验和失败诊断的汇报材料**。

先看 [中期研究报告](mid_presentation/中期研究报告.md)、[完整报告 PDF](mid_presentation/中期研究报告.pdf) 或 [11 页汇报 PDF](mid_presentation/中期汇报.pdf)。报告正文为中文，**9 张汇报图的标题、坐标轴、图例和注释全部为英文**。

| 本次上传的文件 | 内容与用途 |
|---|---|
| [`mid_presentation/`](mid_presentation/README.md) | 报告 Markdown / HTML / PDF、汇报 PDF、讲稿、图表总览，9 张 PNG 与同名 SVG，以及指标表和实验台账 |
| [`mid_presentation/build_presentation.py`](mid_presentation/build_presentation.py)、[`export_report_pdf.py`](mid_presentation/export_report_pdf.py) | 使用已发布数值快照离线重绘图表、汇报和完整报告；无需模型服务或 GPU |
| [`code/calibrate_consistency.py`](code/calibrate_consistency.py) | 按任务分组做五折 OOF 校准，并用全部旧步骤拟合、冻结最终 sigmoid |
| [`code/confidence_runtime.py`](code/confidence_runtime.py) | 将调用前状态及候选调用的多采样一致性转换为正确概率、验证建议和原因 |
| [`code/evaluate_minimal_validation.py`](code/evaluate_minimal_validation.py) | 使用已保存的新任务步骤、标签和采集汇总，离线复算验证指标与开销 |
| [`code/README_step_confidence.md`](code/README_step_confidence.md) | 最小模块的调用方式、复算命令、各文件用途和适用范围 |
| [`outputs/confidence_minimal_v1/`](outputs/confidence_minimal_v1/) | 冻结校准器、OOF 预测、精简的新任务样本与标签、任务结束判定、指标、图和分项报告 |
| [`outputs/confidence_v1/`](outputs/confidence_v1/)、[`outputs/legacy_relabel_v1/`](outputs/legacy_relabel_v1/)、[`outputs/new_signals_v1/`](outputs/new_signals_v1/) | 参考前缀缺负例、已替代的三元标签方案、长思考补采早停等历史摘要；用于解释失败经验，不作为额外独立验证集 |

主要结果与边界：

- 同一批 **198 步、158 正 / 40 负** 的探索性比较中，logprob / 自陈 / 一致性 AUROC 分别为 **0.6351 / 0.7172 / 0.8812**。一致性的配对优势区间未跨 0，自陈相对 logprob 的优势仍不确定。
- 在这批旧任务上按任务分组 OOF，一致性校准后 **Brier 从 0.1235 降至 0.0917，ECE 从 0.1347 降至 0.0674**。这是内部开发结果，不能消除先在同批数据筛选信号的偏差。
- 冻结后新增 **30 个任务、50 次调用，49 正 / 1 负，任务交付合格 22/30**。新任务校准 Brier 为 **0.0599**，未胜固定先验 **0.0527** 或 logprob **0.0209**；仅一个错误不足以验证稳定的泛化能力。新旧任务与执行设置不同，不能将成功率差异解释为能力提升。
- 已接通“调用前一致性 → 正确概率 → 验证建议”。**尚未验证实际纠错收益、结束步骤置信度或递归改进效果**。评分对每个候选调用额外进行三次采样，其成本计入报告，不能只计算选中步骤的验证预算。

本次公开包保留可复算的研究数据与离线代码，排除原始请求日志、服务启动及温控记录等服务器运行信息。来源清单区分归档原件与公开精简文件，复现方法见各目录说明。下文初始实验与环境说明保留为历史，当前结论以上述报告为准。

---

## 1. 研究内容

官方口径的三步研究内容：

1. **步骤级置信度估计与校准** —— 结合推理过程、工具调用、执行结果获取并校准置信度，用于识别薄弱环节、分配验证资源。
2. **自主验证与纠错** —— 针对低置信或异常变化步骤，触发信息补充、工具验证、局部重试、计划调整；研究适用条件，减少错误传播。
3. **递归改进与评估** —— 独立验证并筛选纠错经验，以记忆/策略更新/训练改进后续执行；评估成功率、泛化能力与开销。

三个可检验子问题：**Q1** 选择性预测（置信度能否定位错误步骤）· **Q2** 预算约束下的验证资源分配 · **Q3** 递归改进与跨任务形态泛化。

---

## 2. 项目环境

### 环境安装

```bash
# 1) 创建环境
conda create -n confagent python=3.11 -y
conda activate confagent

# 2) 安装 vLLM
pip install vllm==0.19.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3) 本项目依赖
pip install openai numpy scikit-learn matplotlib paramiko

# 4) 【必做】卸载 datasets 与 pyarrow —— 见第 7 节已知问题
pip uninstall -y datasets pyarrow
```

**conda 26.x 注意**：默认频道需接受 ToS 会报 `CondaToSNonInteractiveError`，写 `~/.condarc` 改用清华镜像即可。

---

## 3. 数据集：API-Bank

| 项目 | 说明 |
|---|---|
| 来源 | API-Bank (Li et al., 2023) |
| 使用 split | **level-3 test**（多步 API 调用链），文件 `test-data/level-3.json` |
| 训练 split | `training-data/lv3-train.json` |
| 放置路径 | `~/confagent/data/api-bank/` |

### 3.1 单条记录结构

```jsonc
{
  "requirement": "用户自然语言需求",
  "apis": [                                   // gold 调用链（逐步）
    {
      "api_name": "QueryMeeting",
      "input":    { "user_name": "Tom" },
      "output":   { "output": {...} },         // 环境真实返回
      "exception": null
    }
  ],
  "response": "期望的最终回复"
}
```

查看数据集统计：

```bash
python code/apibank_env.py --data-dir ~/confagent/data/api-bank
```

### 3.2 确定性重放环境（无需实现真实 API）

API-Bank 的 level-3 记录链已保存每步 `(api_name, input) -> output`，因此本项目把它当作**确定性重放环境**：

- 模型调用与 gold 调用**匹配** → 返回记录中的真实 output
- 模型调用**不匹配** → 返回异常，并分类失败原因（`wrong_api` / `wrong_value` / `missing_arg` / `extra_call` / `search_miss`）

### 3.3 步骤可靠性标签 `y_t` 的构造

自由文本参数（如 `EmailReminder.content`）无法逐字复现，若用精确字符串匹配会产生大量**假阴性**，直接污染校准标签。因此采用**字段感知匹配**：

| 字段类型 | 判定规则 |
|---|---|
| 结构化字段（时间/地点/收件人/ID/姓名） | 严格匹配（时间做规范化：`2021-01-01 10:00` ≡ `2021-01-01 10:00:00`） |
| 自由文本字段（长句、多词） | 只查"非空 + 主题相关"（与 gold 至少 1 个有意义词重叠），不评判措辞 |

同时输出两套标签用于**敏感性分析**：`matched_gold`（宽松口径，主标签）与 `strict_matched_gold`（严格口径，全字段精确匹配）。模型重试会导致步骤位置与 gold 错位，因此采用**按 `api_name` 贪心对齐**而非位置对齐。

---

## 4. 模型：Qwen3-8B

| 项目 | 说明 |
|---|---|
| 模型 | Qwen3-8B（指令微调版） |
| 部署 | 本地 vLLM，OpenAI 兼容接口 |
| 路径 | `~/confagent/models/Qwen3-8B`（约 15.3GB） |

下载（ModelScope 示例）：

```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-8B --local_dir ~/confagent/models/Qwen3-8B
```

服务端需开启工具调用解析：`--enable-auto-tool-choice --tool-call-parser hermes`。

---

## 5. 运行步骤

### 步骤 0 · 启动 vLLM 服务

```bash
# 单卡
GPU=1 PORT=8000 bash code/run_vllm.sh

# 多卡多副本（每个副本独占一张卡，端口递增）
bash code/launch_replicas.sh "1 2" 8000        # GPU 1,2 -> 端口 8000,8001
```

> 运行前请按自己的环境配置项目目录与 conda 环境。

### 步骤 1 · 验证 logprob 可用性（项目第一道技术关卡）

确认 vLLM 能**同时**返回 `tool_calls` 解析结果与 token 级 `logprob`：

```bash
python code/verify_logprobs.py --base-url http://localhost:8000/v1 --model Qwen3-8B
```

### 步骤 2 · 批量采集轨迹

```bash
python code/run_batch.py \
  --endpoints localhost:8000,localhost:8001 \
  --limit 50 --concurrency 2 \
  --out results/traj_main.jsonl
```

或一条命令完成"起副本 → 跑批 → 报温度"：

```bash
bash code/run_experiment.sh "1 2" 8000 --limit 50 --concurrency 2 \
  --verify-policy direct --verify-budget 0 --out results/traj_seed0.jsonl
```

常用参数：`--distractors N`（注入 N 个干扰 API 提高难度）、`--self-report`（采集自陈置信度）、`--consistency-k K`（多采样一致性）、`--no-thinking`（关闭思维链）。

### 步骤 3 · 置信度校准评估

```bash
python code/calibration.py --traj results/traj_main.jsonl --out-dir outputs
```

输出 ECE / MCE / Brier / NLL / AUROC / AURC、可靠性图与风险—覆盖率曲线，并做消融 **A1**（哪种置信度信号最可校准）与 **A2**（哪种校准方法最有效）。

### 步骤 4 · 本地复盘（零第三方依赖）

```bash
python code/analyze_results.py results/traj_main_clean.jsonl
```

### 步骤 5 · 自主验证与纠错（研究内容二）

新实验统一使用 `feedback_mode=realistic` 和 schema v4。核心问题是：在相同的 K=3
置信度评分成本、相同 Critic 与相同检查次数上限下，低置信选择是否优于标签盲随机选择。
正式测试应让三组复用同一冻结 task manifest；以下用开发运行展示命令形式。

```bash
# 直接执行：无评分、无执行前检查
python code/run_batch.py --endpoints localhost:8000 --limit 50 \
  --verify-policy direct --verify-budget 0 --feedback-mode realistic \
  --out results/q2_direct.jsonl

# 随机选择：仍支付相同 K=3 评分成本，但忽略分数
python code/run_batch.py --endpoints localhost:8000 --limit 50 \
  --verify-policy random_check --verifier critic --verify-budget 4 \
  --random-check-rate 0.5 --max-checks-per-task 1 \
  --feedback-mode realistic --out results/q2_random.jsonl

# 主要方法：冻结置信度选择可疑步骤，再由 Critic 检查和修订
python code/run_batch.py --endpoints localhost:8000 --limit 50 \
  --verify-policy confidence_check --verifier critic --verify-budget 4 \
  --max-checks-per-task 1 --feedback-mode realistic \
  --out results/q2_confidence.jsonl

# 多条件等预算分析（第一个文件作为配对比较基线）
python code/analyze_verification.py \
  results/q2_direct.jsonl results/q2_random.jsonl results/q2_confidence.jsonl \
  --require-complete-pairs --json-out outputs/q2_metrics.json \
  --plot-out outputs/q2_summary.png \
  --corrections-out outputs/q2_verified_corrections.jsonl
```

核心策略：`direct / random_check / confidence_check`。`provenance / exception / always /
threshold / fused` 与 `consistency / hybrid` 保留为辅助基线或消融。完整服务器流程、manifest
和预算说明见 `研究内容二_自主验证与纠错开发说明.md`。

---

## 6. 目录结构与文件职责

```
.
├── README.md                     # 本文件
├── 项目进展与分工说明_组内同步.md    # 组内分工与进度同步
├── code/                         # 全部源码
└── results/                      # 实验轨迹数据（trajectory.jsonl 快照）
```

### code/

| 文件 | 职责 |
|---|---|
| `trajectory.py` | **全组唯一的轨迹格式定义**。`TrajectoryStep` / `Trajectory` / `TrajectoryWriter`，受控词表校验，JSONL 追加写入，错误传播率计算 |
| `apibank_env.py` | API-Bank 加载器 + 确定性 mock 环境。字段感知匹配、双口径标签、按 api_name 贪心对齐、失败原因分类 |
| `react_agent.py` | ReAct 主循环 + 步骤级置信度提取。推理段 logprob 特征（geo_mean_prob 等）、多采样一致性、自陈置信度、重复调用熔断 |
| `run_batch.py` | 多端点批处理、任务 manifest/数据哈希、endpoint 预检、实验协议与完成摘要、`ThermalGuard` 温度保护 |
| `calibration.py` | 校准评估（研究内容一核心分析）。ECE/Brier/NLL/AUROC/AURC、温度缩放/Platt/Isotonic、可靠性图与风险—覆盖率曲线 |
| `analyze_results.py` | 纯 Python 结果分析（无第三方依赖），用于本地复盘已采集轨迹 |
| `verification.py` | **研究内容二核心**：参数来源与单位检查、结构化 critic、修订安全门、多采样验证和纠错归因 |
| `analyze_verification.py` | 验证触发 precision/recall、纠错收益/伤害、恢复成功率、成本与配对 bootstrap 分析 |
| `run_vllm.sh` | 启动单个 vLLM 服务（Qwen3-8B，单卡） |
| `launch_replicas.sh` | 在指定 GPU 组上启动多个 vLLM 副本 |
| `stop_replicas.sh` | 停止 vLLM 副本并等待显存释放 |
| `run_experiment.sh` | 一条命令跑一轮实验（起副本 → 跑批 → 报温度） |
| `thermal_watch.sh` | GPU 温度后台监控，超阈值写告警日志 |
| `vllm_serve_patched.py` | vLLM 启动包装器：绕过"某张 GPU 故障导致 vLLM 无法启动"（NVML 容错补丁） |
| `verify_logprobs.py` | 技术验证：vLLM 能否同时返回 tool_call 与 token logprob |

### tests/

`tests/test_verification.py` 使用纯本地假客户端验证状态机，不下载或加载任何模型：

```bash
python -m unittest discover -s tests -v
```

### results/

| 文件 | 说明 |
|---|---|
| `traj_pilot.jsonl` | 小规模 pilot（8 任务） |
| `traj_vanilla.jsonl` | vanilla ReAct 基线（5 任务） |
| `traj_main_seed0.jsonl` | 主实验 seed=0（67 轨迹） |
| `traj_main_clean.jsonl` | 清洗后主实验集（50 轨迹） |

轨迹 schema（`trajectory.py` 定义）：

```
task_id, step_id, action_type, action_input, observation,
confidence, confidence_source, verified, correction,
cost_tokens, cost_calls, final_success
```

schema v4 额外记录 `candidate_action / confidence_result / selection_policy /
selected_for_check / verification_result / revision_guard / correction_outcome / budget_before /
budget_after`，并把 Actor、置信评分、Critic、工具调用、token 和延迟分开计量。任务终局同时区分：

- `final_success`：最终完成全部必要步骤，允许先失败后恢复；
- `clean_success`：无任何已执行错误的成功；
- `recovered_success`：发生过执行错误、后来成功恢复。

Q2 新运行统一写为 schema v4：区分 `replay_success` 与需要最终回答的 `final_success`，记录
冻结 Q1 评分、选择决策、修订安全门、replay/grounded 双标签和 signal/verifier 分项成本；
执行前 verifier 纠错与执行后的局部重试分别归因。

---

## 7. 已知问题与重要结论

### 7.1 环境类

- **pyarrow 26 的 jemalloc 后台线程导致 vLLM 段错误**（`SIGSEGV`，报 `Model architectures ['Qwen3ForCausalLM'] failed to be inspected`，子进程退出码 139）。崩溃点飘忽不定，`gdb` 可抓到 `jemalloc_bg_thd ... libarrow.so.2600` 原生栈。
  **解决方案**：卸载 `datasets` 与 `pyarrow`（vLLM 仅在 `bench` 附加项中需要，非硬依赖）。环境变量 `MALLOC_CONF` / `ARROW_DEFAULT_MEMORY_POOL` 均无效。

### 7.2 方法类

- **早期接口配置中工具调用段的 token logprob 恒为 0**（约束解码下概率≈1），因此当时的实验主要从**推理段（`<think>`）**提取。该观察依赖接口和解码配置，不能概括为所有工具参数 logprob 都无效。
- **回灌历史时必须剥离 `<think>` 段**，否则模型认为"已经想过了"，推理段只剩 1–2 个 token，置信度假性变成 1.000。
- **旧标签下 logprob 区分度较弱**：初始报告 AUROC 约 0.51–0.54。修订二元标签后，同一 198 步的 `geo_mean_prob` AUROC 为 **0.6351**；不能继续用旧标签结果替代当前结果。
- **小样本与标签口径会影响结论**：早期 pilot 指标仅作调试记录，不作为独立验证；当前可核查结果见中期报告。

自陈与一致性的初筛、内部校准及新任务验证现已完成，结果和局限见本页顶部与中期报告。

---

## 8. 参考文献

- Shinn et al. *Reflexion: Language Agents with Verbal Reinforcement Learning.* NeurIPS 2023.
- Asai et al. *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* ICLR 2024.
- Li et al. *API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs.* EMNLP 2023.

## 最新进展：二元标签修订与 logprob 重新评估

该阶段完成的是 [二元标签修订与 logprob 重新评估](outputs/legacy_binary_v2/二元标签与logprob评估.md)：对已有执行轨迹进行标签清洗，再使用原有 logprob 重新计算指标，没有训练新模型。前文初始实验统计保留为历史，本节给出当前二元标注口径。

新增代码：

| 文件 | 作用 |
|---|---|
| `code/relabel_binary_steps.py` | 将调用和任务结束结果标为正确／错误，记录判定依据，保留原始数据 |
| `code/evaluate_binary_logprob.py` | 重新计算 AUROC 等指标，进行重试去重分析并生成图表 |

最新结果：

- **198 次工具调用**：158 正确、40 错误，全部参与主评估。
- **18 次无工具结束输出**：9 次任务完成、9 次未完成；原日志没有保存其 logprob。
- **50 个任务**：按新完成规则复核，14 完成、36 未完成。
- `geo_mean_prob` AUROC 为 **0.6351**；去重后为 **0.6980**。这是标签修订后的重评结果，不代表模型能力提升。

标签明细、指标和图表位于 [`outputs/legacy_binary_v2/`](outputs/legacy_binary_v2/)。

## 最新进展：自陈置信度与多采样一致性初筛

在清洗后的 **198 步（158 正确、40 错误）** 上补采两个新信号，固定原调用和标签，不执行新工具。完整结果见 [自陈置信度与多采样一致性评估](outputs/new_signals_fast_v1/新信号评估.md)。

| 信号 | AUROC | 优先检查最低分 20% 的错误捕获率 |
|---|---:|---:|
| 原 logprob | 0.6351 | 29.0% |
| 自陈置信度 | 0.7172 | 33.4% |
| 多采样一致性（K=3） | **0.8812** | **68.7%** |

一致性去重后 AUROC 为 **0.8793**。同分跨过抽查预算时，捕获率按随机顺序的期望计算，表中 20% 对应 39.6 步的期望预算；中期报告另给严格 39 步的可执行预算。本轮使用重建上下文、关闭思考的快速初筛，初筛当时尚未拟合校准器；后续的内部校准与新增任务验证见本页顶部和中期报告。

| 新文件 | 作用 |
|---|---|
| [`code/collect_new_signals.py`](code/collect_new_signals.py) | 重建调用前状态，补采自陈分数和 3 次独立动作；不执行工具 |
| [`code/evaluate_new_signals.py`](code/evaluate_new_signals.py) | 固定标签计算 AUROC、配对区间、错误捕获率和去重结果，生成报告与图 |
| [`code/README_new_signals.md`](code/README_new_signals.md) | 信号定义、文件说明、CPU 复算及重新采集方法 |
| `tests/test_new_signal_collection.py`、`tests/test_new_signal_evaluation.py`、`tests/test_published_new_signals.py` | 分别检查上下文隔离与动作匹配、统计口径、公开结果与样本的一致性 |

[`outputs/new_signals_fast_v1/`](outputs/new_signals_fast_v1/) 保存逐步 `scores.jsonl`、精简采样动作 `sample_actions.jsonl`、模型可见状态 `states.jsonl`、冻结协议 `protocol.json`，以及指标 JSON/CSV、逐步对应 CSV、对比图、采集汇总和校验记录。**已有分数可直接用 CPU 复算指标，无需模型服务或完整 API-Bank 数据。**
