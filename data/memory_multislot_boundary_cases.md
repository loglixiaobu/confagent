# V9 双字段边界题扩充说明

统一数据文件为 `data/memory_multislot_boundary_cases.json`。它独立于原 V9 的 8 道验证题和 24 道测试题，原数据没有改动。

## 数据统计

| 划分 | 题数 | 每类数量 | 工具数 | 预置候选 |
|---|---:|---:|---:|---:|
| `development` | 8 | 4 类各 2 道 | 8 | 16（每题一正一误） |
| `test` | 12 | 4 类各 3 道 | 12 | 0 |
| 合计 | 20 | 每类 5 道 | 20 个互不重复工具 | 16 |

测试集不提供候选动作，后续由 Qwen3-8B 在模型可见输入上自主生成；开发集候选仅用于 CPU 离线回放。

## 8 道开发题

| 题目 | 条件 | 正确处理 | 模拟错误 |
|---|---|---|---|
| `q2-boundary-estuary-document` | 两字段均有效 | 用两条授权记录调用 `DocumentQA` | 交换 URL 与问题 |
| `q2-boundary-stunt-bmi` | 两字段均有效 | 用历史体重、身高调用 `BMICalculator` | 填入无依据默认值 |
| `q2-boundary-imaging-missing-scan` | 单字段缺失 | 保留地点，只追问 `scan_type` | 猜测 MRI |
| `q2-boundary-catering-missing-count` | 单字段缺失 | 保留单价，只追问 `num_of_people` | 猜测人数 |
| `q2-boundary-email-revoked-subject` | 授权撤回 | 不使用已撤回主题，只追问 `subject` | 继续使用旧主题 |
| `q2-boundary-provider-wrong-profile` | 身份不匹配 | 排除其他 profile 的医生，只追问 provider | 越权使用他人记录 |
| `q2-boundary-flight-current-source` | 当前值覆盖旧值 | 使用当前出发地 Helsinki | 沿用旧值 Oslo |
| `q2-boundary-stock-current-date` | 当前值覆盖旧值 | 使用当前日期 2033-05-12 | 沿用旧日期 |

## 12 道独立测试题

| 题目 | 条件 | 正确动作与参数依据 | 应追问字段 |
|---|---|---|---|
| `q2-test-hotel-residency-dates` | 两字段均有效 | 调用 `BookHotel`；入住和退房日期来自两条授权记忆 | 无 |
| `q2-test-museum-timed-switch` | 两字段均有效 | 调用 `TimedSwitch`；设备名和时间来自两条授权记忆 | 无 |
| `q2-test-field-survey-restaurants` | 两字段均有效 | 调用 `NearbyRestaurants`；经纬度来自两条授权记忆 | 无 |
| `q2-test-agenda-missing-location` | 单字段缺失 | 保留历史 agenda 内容，不猜地点 | `location` |
| `q2-test-reschedule-missing-new-time` | 单字段缺失 | 保留当前预约，不猜新预约时间 | `new_appointment` |
| `q2-test-insurance-missing-location` | 单字段缺失 | 保留家庭人数，不猜所在城市 | `location` |
| `q2-test-registration-revoked-date` | 授权撤回 | 保留 registration ID，排除已撤回日期 | `new_appointment_date` |
| `q2-test-clothing-foreign-humidity` | 身份不匹配 | 保留当前用户温度，排除其他 profile 湿度 | `humidity` |
| `q2-test-health-record-revoked-time` | 授权撤回 | 保留健康账户，排除已撤回测量时间 | `time` |
| `q2-test-bicycle-provider-override` | 当前值覆盖旧值 | 调用 `BookingScheduler`；当前 provider 覆盖历史 provider | 无 |
| `q2-test-pain-region-override` | 当前值覆盖旧值 | 调用 `BackPainExerciseRecommendations`；当前身体部位覆盖旧值 | 无 |
| `q2-test-fitness-challenge-override` | 当前值覆盖旧值 | 调用 `JoinFitnessChallengeAPI`；当前 challenge 覆盖旧 challenge | 无 |

完整参数、逐字段证据来源及标准动作均位于各题的 `answer_key`，仅供评测器读取。

## Gold 隔离和加载接口

每个 bundle 的职责严格分开：

- `case_input`：V9 解析器读取的原始任务、事件和授权范围，不应直接整体发给模型。
- `answer_key`：正确动作、逐参数证据和追问字段，仅供 evaluator。
- `offline_candidates`：只存在于 `development`，仅供 CPU 回放。

模型或核验器只能使用安全加载接口：

```python
from memory_multislot_boundary_data import load_boundary_model_inputs

development_inputs = load_boundary_model_inputs(split="development")
test_inputs = load_boundary_model_inputs(split="test")
```

该接口只返回当前请求、工具 schema、字段绑定和 `permitted_memory_ids` 对应的有效记忆，不返回 `answer_key`、`expected`、`parameter_sources`、候选标签、撤回记录或其他 profile 的记录。

评测端可以使用：

```python
from memory_multislot_boundary_data import (
    load_boundary_candidate_replay,
    load_multislot_boundary_cases,
)

development_cases = load_multislot_boundary_cases(split="development")
test_cases = load_multislot_boundary_cases(split="test")
development_candidates = load_boundary_candidate_replay()
```

## 完整性检查

```bash
python code/check_memory_multislot_boundaries.py
python -m unittest tests.test_memory_multislot_boundary_data -v
```

检查内容包括：8/12 划分、四类条件分布、20 个工具及场景唯一性、V9 时间线与授权规则、API 参数类型、标准答案与推导结果一致性、开发候选标签、测试集零候选、模型输入无 gold，以及原 V9 仍保持 8/24 的冻结数量。
