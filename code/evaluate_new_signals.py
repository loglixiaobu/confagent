#!/usr/bin/env python
"""CPU-only paired evaluation of new scores on the frozen 198-call labels.

Input JSONL rows: task_id, integer step_id, self_report (0..1 or null),
consistency (0..1 or null), consistency_k (positive integer when scored).
No scores are imputed, calibrated, inverted, or selected using labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SIGNALS = ("logprob", "self_report", "consistency")
NEW_SIGNALS = SIGNALS[1:]
SEED = 20261015
DISPLAY = {"logprob": "原 logprob", "self_report": "自陈置信度", "consistency": "多采样一致性"}
DISPLAY["strict_consistency"] = "严格一致性（次要）"


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_path(path):
    """Reproduction paths are relative to the repository, with no host prefix."""
    return Path(os.path.relpath(Path(path).resolve(), ROOT)).as_posix()


def key(row):
    task, step = row.get("task_id"), row.get("step_id")
    if not isinstance(task, str) or not task or type(step) is not int or step < 1:
        raise ValueError("Every row needs nonempty task_id and integer step_id >= 1")
    return task, step


def probability(value):
    """An explicit zero is valid; missing, bools, NaN and bad ranges are not."""
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        return None
    return float(value)


def join_scores(calls, score_rows, require_history_flags=False):
    frozen = {}
    for row in calls:
        k = key(row)
        if k in frozen:
            raise ValueError(f"Duplicate frozen label key: {k}")
        if type(row["revision"]["y"]) is not int or row["revision"]["y"] not in (0, 1):
            raise ValueError(f"Nonbinary label: {k}")
        if type(row.get("first_semantic_attempt")) is not bool:
            raise ValueError(f"Missing frozen dedup flag: {k}")
        if "before_first_mock_error" in row and type(row["before_first_mock_error"]) is not bool:
            raise ValueError(f"Invalid frozen history flag: {k}")
        if require_history_flags and "before_first_mock_error" not in row:
            raise ValueError(f"Missing frozen history flag: {k}")
        frozen[k] = row
    scores = {}
    for row in score_rows:
        k = key(row)
        if k in scores:
            raise ValueError(f"Duplicate score key: {k}; consolidate retries before evaluation")
        if k not in frozen:
            raise ValueError(f"Score key has no frozen call label: {k}")
        scores[k] = row
    joined = []
    invalid = Counter()
    for k, row in frozen.items():
        scored = scores.get(k, {})
        raw = row["original_step"].get("raw_logprobs") or {}
        logprob = probability(raw.get("geo_mean_prob"))
        if not raw.get("available") or raw.get("degenerate"):
            logprob = None
        p = {"logprob": logprob}
        for name in (*NEW_SIGNALS, "strict_consistency"):
            value = scored.get(name)
            p[name] = probability(value)
            if value is not None and p[name] is None:
                invalid[name] += 1
        kval = scored.get("consistency_k")
        if p["consistency"] is not None and (type(kval) is not int or kval < 1):
            invalid["consistency_missing_or_invalid_k"] += 1
            p["consistency"] = None
        if p["strict_consistency"] is not None and (type(kval) is not int or kval < 1):
            invalid["strict_consistency_missing_or_invalid_k"] += 1
            p["strict_consistency"] = None
        joined.append({"task_id": k[0], "step_id": k[1], "y": row["revision"]["y"],
                       "first_semantic_attempt": row["first_semantic_attempt"],
                       "before_first_mock_error": row.get("before_first_mock_error", True),
                       "reason_code": row["revision"].get("reason_code"),
                       "consistency_k": kval if type(kval) is int and kval > 0 else None,
                       "has_score_row": k in scores, **p})
    return joined, {"label_rows": len(calls), "score_rows": len(scores),
                    "missing_score_rows": len(calls) - len(scores), "invalid_scores": dict(invalid),
                    "missing_history_flags": sum("before_first_mock_error" not in r for r in calls)}


def nonnegative(value):
    if type(value) in (int, float) and math.isfinite(value) and value >= 0:
        return value
    return None


def collection_diagnostics(score_rows, folder):
    """Read public research counts, never runtime logs or host metadata.

    Metadata comes from the score input directory, independently of the output
    destination. collection_summary.json is an optional, curated export. Only the count and
    token fields below are read. Absent, malformed or invalid fields remain
    unknown; none of these diagnostics changes a score or evaluation cohort.
    """
    folder = Path(folder)
    fields = ("n_samples", "n_agree", "n_strict_agree", "n_no_tool", "n_truncated")
    quality = {"score_rows": len(score_rows), "totals": {f: 0 for f in fields},
               "rows_with_counter": {f: 0 for f in fields}, "rows_with_no_tool": 0,
               "rows_with_truncation": 0, "expected_samples_with_known_count": 0,
               "missing_samples_with_known_count": 0}
    for row in score_rows:
        for field in fields:
            value = row.get(field)
            if type(value) is int and value >= 0:
                quality["totals"][field] += value
                quality["rows_with_counter"][field] += 1
        quality["rows_with_no_tool"] += int(nonnegative(row.get("n_no_tool")) not in (None, 0))
        quality["rows_with_truncation"] += int(nonnegative(row.get("n_truncated")) not in (None, 0))
        k, n = row.get("consistency_k"), row.get("n_samples")
        if type(k) is int and k > 0 and type(n) is int and n >= 0:
            quality["expected_samples_with_known_count"] += k
            quality["missing_samples_with_known_count"] += max(0, k-n)
    n = quality["totals"]["n_samples"]
    quality["no_tool_fraction_of_counted_samples"] = quality["totals"]["n_no_tool"]/n if n else None
    quality["truncation_fraction_of_counted_samples"] = quality["totals"]["n_truncated"]/n if n else None

    summary_path = folder / "collection_summary.json"
    summary = {"file_present": summary_path.exists()}
    allowed = ("successful_requests", "failed_requests", "samples", "no_tool_samples",
               "multiple_tool_samples", "truncated_samples", "prompt_tokens",
               "completion_tokens", "total_tokens", "usage_records",
               "successful_responses_without_usage")
    public_counts = {field: None for field in allowed}
    if summary_path.exists():
        summary["sha256"] = sha(summary_path)
        try:
            loaded = json.loads(summary_path.read_text())
            if not isinstance(loaded, dict):
                raise TypeError("Public collection summary must be an object")
            for field in allowed:
                value = loaded.get(field)
                if type(value) is int and value >= 0:
                    public_counts[field] = value
        except (json.JSONDecodeError, TypeError):
            summary["parse_error"] = True
    quality["reported_sample_counts"] = {field: public_counts[field] for field in
        ("samples", "no_tool_samples", "multiple_tool_samples", "truncated_samples")}
    cost = {field: public_counts[field] for field in
        ("successful_requests", "failed_requests", "prompt_tokens", "completion_tokens",
         "total_tokens", "usage_records", "successful_responses_without_usage")}

    protocol_path = folder / "protocol.json"
    protocol = {"file_present": protocol_path.exists()}
    if protocol_path.exists():
        try:
            loaded = json.loads(protocol_path.read_text())
            if not isinstance(loaded, dict):
                raise TypeError("Protocol must be an object")
            protocol["sha256"] = sha(protocol_path)
            for field in ("version", "model", "consistency_k", "consistency_temperature", "consistency_top_p",
                          "consistency_thinking", "max_tokens", "self_report_temperature", "self_report_thinking",
                          "seed", "context", "consistency_definition", "strict_consistency_definition"):
                if field in loaded:
                    protocol[field] = loaded[field]
        except (json.JSONDecodeError, TypeError):
            protocol["parse_error"] = True
    return {"quality": quality, "cost": cost, "summary": summary, "protocol": protocol}


def error_capture(p, y, budget=.2):
    """Expected capture when ties cross an exactly 20% verification budget."""
    total_errors = float((1 - y).sum())
    if not total_errors:
        return None
    remaining, found = budget * len(y), 0.
    for score in sorted(set(p)):
        mask = p == score
        fraction = min(1., max(0., remaining / int(mask.sum())))
        found += fraction * float((1 - y[mask]).sum())
        remaining -= min(remaining, int(mask.sum()))
        if remaining <= 0:
            break
    return found / total_errors


def metrics(p, y):
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=int)
    result = {"n": len(y), "correct": int(y.sum()), "errors": int((1-y).sum()),
              "AUROC": None, "error_AUPRC": None, "Brier": None, "ECE": None,
              "error_capture_at_20pct": None, "correct_mean": None, "error_mean": None,
              "random_error_AUPRC": float((1-y).mean()) if len(y) else None,
              "random_error_capture_at_20pct": .2 if (1-y).sum() else None,
              "unique_score_values": len(set(p)), "bins": []}
    if not len(y):
        return result
    if len(set(y)) == 2:
        result["AUROC"] = float(roc_auc_score(y, p))
        result["error_AUPRC"] = float(average_precision_score(1-y, 1-p))
    result["Brier"] = float(np.mean((p-y)**2))
    result["error_capture_at_20pct"] = error_capture(p, y)
    for label, name in [(1, "correct_mean"), (0, "error_mean")]:
        if (y == label).any():
            result[name] = float(p[y == label].mean())
    ece = 0.
    for i in range(5):
        mask = (p >= i/5) & ((p < (i+1)/5) if i < 4 else (p <= 1))
        n = int(mask.sum())
        if n:
            avg, acc = float(p[mask].mean()), float(y[mask].mean())
            ece += n / len(y) * abs(avg-acc)
            result["bins"].append({"lo": i/5, "hi": (i+1)/5, "n": n,
                                   "confidence": avg, "accuracy": acc})
    result["ECE"] = ece
    return result


def paired_bootstrap(predictions, y, tasks, n_boot=1000, seed=SEED):
    """Use the same sampled task copies for each score and its AUROC delta."""
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    y, tasks = np.asarray(y, dtype=int), np.asarray(tasks)
    pp = {name: np.asarray(p, dtype=float) for name, p in predictions.items()}
    if any(len(p) != len(y) for p in pp.values()) or len(tasks) != len(y):
        raise ValueError("Bootstrap inputs must have the same number of rows")
    groups = np.unique(tasks)
    indices = {g: np.flatnonzero(tasks == g) for g in groups}
    rng = np.random.default_rng(seed)
    aucs = {name: [] for name in pp}
    deltas = {name: [] for name in pp if name != "logprob" and "logprob" in pp}
    if len(groups):
        for _ in range(n_boot):
            idx = np.concatenate([indices[g] for g in rng.choice(groups, len(groups), replace=True)])
            yy = y[idx]
            if len(set(yy)) != 2:
                continue
            values = {name: float(roc_auc_score(yy, p[idx])) for name, p in pp.items()}
            for name, value in values.items():
                aucs[name].append(value)
            for name in deltas:
                deltas[name].append(values[name]-values["logprob"])

    def interval(values):
        return {"ci95": np.quantile(values, [.025, .975]).tolist() if values else None,
                "valid_resamples": len(values), "total_resamples": n_boot,
                "unit": "task", "seed": seed,
                "single_class_resamples_omitted": n_boot-len(values)}

    return {"AUROC": {name: interval(values) for name, values in aucs.items()},
            "AUROC_difference_vs_logprob": {name: interval(values) for name, values in deltas.items()}}


def summarize(rows, names, population_n, n_boot=1000, seed=SEED):
    """rows must already be the shared complete cases for the named scores."""
    if any(row.get(name) is None for row in rows for name in names):
        raise ValueError("Cohort contains missing scores")
    yy = np.array([r["y"] for r in rows], dtype=int)
    pp = {name: np.array([r[name] for r in rows], dtype=float) for name in names}
    boot = paired_bootstrap(pp, yy, [r["task_id"] for r in rows], n_boot, seed)
    result = {"n": len(rows), "population_n": population_n,
              "coverage": len(rows)/population_n if population_n else 0.,
              "tasks": len({r["task_id"] for r in rows}), "correct": int(yy.sum()),
              "errors": int((1-yy).sum()),
              "error_tasks": sorted({r["task_id"] for r in rows if r["y"] == 0}),
              "signals": {}, "paired_AUROC_difference_vs_logprob": {}}
    for name in names:
        stat = metrics(pp[name], yy)
        stat["AUROC_bootstrap"] = boot["AUROC"][name]
        result["signals"][name] = stat
    for name, ci in boot["AUROC_difference_vs_logprob"].items():
        a, b = result["signals"][name]["AUROC"], result["signals"]["logprob"]["AUROC"]
        result["paired_AUROC_difference_vs_logprob"][name] = {
            "point": a-b if a is not None and b is not None else None, **ci}
    return result


def evaluate_rows(rows, n_boot=1000, seed=SEED):
    cohorts = {}
    for name, selected in [("all_calls", rows),
                           ("first_semantic_attempt", [r for r in rows if r["first_semantic_attempt"]]),
                           ("before_first_mock_error", [r for r in rows if r.get("before_first_mock_error", True)])]:
        n = len(selected)
        own = {signal: summarize([r for r in selected if r[signal] is not None],
                                 [signal], n, n_boot, seed) for signal in SIGNALS}
        paired = {signal: summarize([r for r in selected if r[signal] is not None and r["logprob"] is not None],
                                    ["logprob", signal], n, n_boot, seed) for signal in NEW_SIGNALS}
        common = summarize([r for r in selected if all(r[s] is not None for s in SIGNALS)],
                           SIGNALS, n, n_boot, seed)
        cohorts[name] = {"population_n": n, "own_coverage": own,
                         "paired_vs_logprob": paired, "common_all": common}
        if any(r.get("strict_consistency") is not None for r in selected):
            strict_rows = [r for r in selected if r.get("strict_consistency") is not None and r["logprob"] is not None]
            cohorts[name]["secondary_strict_consistency"] = summarize(
                strict_rows, ["logprob", "strict_consistency"], n, n_boot, seed)
    return cohorts


def error_type_diagnostics(rows):
    """Fixed-label descriptive groups; missing scores stay out of their means."""
    grouped = {"correct": [r for r in rows if r["y"] == 1]}
    for row in rows:
        if row["y"] == 0:
            grouped.setdefault(row.get("reason_code") or "missing_reason_code", []).append(row)
    result = []
    for group, members in grouped.items():
        stat = {"group": group, "y": 1 if group == "correct" else 0, "n": len(members), "signals": {}}
        for signal in NEW_SIGNALS:
            values = [r[signal] for r in members if r.get(signal) is not None]
            stat["signals"][signal] = {"n_scored": len(values),
                                       "coverage": len(values)/len(members) if members else 0.,
                                       "mean": float(np.mean(values)) if values else None}
        result.append(stat)
    return result


def number(value, digits=4):
    return "不可计算" if value is None else f"{value:.{digits}f}"


def ci_text(ci):
    return "不可计算" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def write_report(out, result):
    all_calls = result["cohorts"]["all_calls"]
    own = all_calls["own_coverage"]
    diagnostics = result["collection_diagnostics"]
    protocol = diagnostics["protocol"]
    thinking = protocol.get("consistency_thinking")
    mode = ("无思考采样策略与原动作的一致程度" if thinking is False else
            "开启思考的重建策略与原动作的一致程度" if thinking is True else
            "重建状态下采样策略与原动作的一致程度（未提供 thinking 模式）")
    lines = ["# 自陈置信度与多采样一致性：固定旧步骤的新信号试验", "",
             "复用已清洗的二元标签和原工具调用，新增评分不执行工具、不推进任务。"
             "评估脚本只进行 CPU 离线统计；采集模型请求及耗时以采集协议为准。", "",
             f"冻结工具调用共 {all_calls['population_n']} 步；读入评分行 {result['input_audit']['score_rows']} 条。"
             "正确为正类，高分应更正确；缺失分数保留缺失，不填 0，也不拿前一步分数代替。", "",
             "| 信号 | 覆盖/总数 | 正确/错误 | AUROC | 任务 bootstrap 95% CI |", "| --- | ---: | ---: | ---: | --- |"]
    if result["input_audit"]["score_rows"] < all_calls["population_n"]:
        lines[4:4] = ["**当前采集尚未覆盖全部冻结步骤，以下仅为已采集部分的临时诊断，不能作为全局结果。**", ""]
    for signal in SIGNALS:
        co = own[signal]
        m = co["signals"][signal]
        lines.append(f"| {DISPLAY[signal]} | {co['n']}/{co['population_n']} | {co['correct']}/{co['errors']} | "
                     f"{number(m['AUROC'])} | {ci_text(m['AUROC_bootstrap']['ci95'])} |")
    lines += ["", "不同信号若覆盖不同，不应直接用上表的 AUROC 差作为提升证据。以下对比固定同一组步骤。", "",
              "| 新信号与原 logprob 同样本比较 | 样本数 | 新信号 AUROC | logprob AUROC | AUROC 差 | 配对差 95% CI |",
              "| --- | ---: | ---: | ---: | ---: | --- |"]
    for signal, co in all_calls["paired_vs_logprob"].items():
        d = co["paired_AUROC_difference_vs_logprob"][signal]
        lines.append(f"| {DISPLAY[signal]} | {co['n']} | {number(co['signals'][signal]['AUROC'])} | "
                     f"{number(co['signals']['logprob']['AUROC'])} | {number(d['point'])} | {ci_text(d['ci95'])} |")
    if "secondary_strict_consistency" in all_calls:
        co = all_calls["secondary_strict_consistency"]
        d = co["paired_AUROC_difference_vs_logprob"]["strict_consistency"]
        lines.append(f"| {DISPLAY['strict_consistency']} | {co['n']} | {number(co['signals']['strict_consistency']['AUROC'])} | "
                     f"{number(co['signals']['logprob']['AUROC'])} | {number(d['point'])} | {ci_text(d['ci95'])} |")
    lines += ["", "| 三信号均有值的共同样本 | 信号 | AUROC | 错误 AUPRC | 最低分 20% 错误捕获率 | Brier | ECE（5 等宽箱） |",
              "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for signal, m in all_calls["common_all"]["signals"].items():
        lines.append(f"| {all_calls['common_all']['n']} | {DISPLAY[signal]} | {number(m['AUROC'])} | "
                     f"{number(m['error_AUPRC'])} | {number(m['error_capture_at_20pct'])} | "
                     f"{number(m['Brier'])} | {number(m['ECE'])} |")
    lines += ["", "AUROC 随机基线为 0.5；随机抽查 20% 的期望错误捕获率为 0.20。"
              "同分跨过 20% 预算时按随机顺序的期望计算。Brier/ECE 只是把原始分数视作概率的诊断；"
              "一致性频率和自陈值都尚未拟合校准器，不能据此宣称已完成置信度校准。", "",
              "| 正确组／固定错误原因 | 标签数 | 自陈覆盖 | 自陈均值 | 一致性覆盖 | 一致性均值 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for group in result["error_type_diagnostics"]:
        name = "正确组" if group["y"] else group["group"]
        a, b = (group["signals"][signal] for signal in NEW_SIGNALS)
        lines.append(f"| {name} | {group['n']} | {a['n_scored']}/{group['n']} | {number(a['mean'])} | "
                     f"{b['n_scored']}/{group['n']} | {number(b['mean'])} |")
    lines += ["", "错误类型直接取冻结 revision.reason_code；均值仅使用各组已有分数，覆盖不足时不能当全组均值。"
              "这张表只描述不同错误上信号的表现；不据此改标签、挑阈值或替换主指标，"
              "各类型的 API 和任务构成不同也会影响均值。", "",
              "| 去重敏感性：只留原标注的首次语义尝试 | 信号 | 覆盖/去重总数 | AUROC | 95% CI |",
              "| --- | --- | ---: | ---: | --- |"]
    dedup = result["cohorts"]["first_semantic_attempt"]
    for signal, co in dedup["own_coverage"].items():
        m = co["signals"][signal]
        lines.append(f"| 原有 first_semantic_attempt 标志 | {DISPLAY[signal]} | {co['n']}/{co['population_n']} | "
                     f"{number(m['AUROC'])} | {ci_text(m['AUROC_bootstrap']['ci95'])} |")
    lines += ["", "去重口径沿用冻结标签中的 first_semantic_attempt，包含任务、API、语义参数和相关前置证据；"
              "不依据新分数或标签挑选重试。JSON 同时保存去重后的配对和三信号共同样本结果。", "",
              "| 历史敏感性：首次 mock 报错之前（含产生首次报错的调用） | 信号 | 覆盖/子集总数 | AUROC | 95% CI |",
              "| --- | --- | ---: | ---: | --- |"]
    history = result["cohorts"]["before_first_mock_error"]
    for signal, co in history["own_coverage"].items():
        m = co["signals"][signal]
        lines.append(f"| 原有 before_first_mock_error 标志 | {DISPLAY[signal]} | {co['n']}/{co['population_n']} | "
                     f"{number(m['AUROC'])} | {ci_text(m['AUROC_bootstrap']['ci95'])} |")
    lines += ["", "该子集减少此前模拟器错误反馈对历史状态的影响，仍复用旧环境记录；不替代全部调用的主分析。", "",
              f"按任务聚类 bootstrap {result['bootstrap']['replicates']} 次，固定种子 {result['bootstrap']['seed']}。"
              "重采样保留任务内的全部步骤和重复任务副本；单类重采样无法计算 AUROC，单独记为省略。"
              "AUROC 差与两条信号使用完全相同的任务副本，避免把独立区间相减当配对检验。", "",
              "原轨迹没有保存完整逐步输入，新增信号来自重建状态：应称为“重建状态下的自评/一致性”，"
              "不等同于原生成当时的原生自陈置信度。当前动作的执行返回、隐藏答案和清洗标签不得用于生成这两个执行前信号。"
              "历史仍可能包含旧模拟器的错误反馈。", "",
              "原 logprob 是旧动作生成时保存的 token 分数；自陈是针对既有候选动作另行询问，"
              "一致性是从重建前置状态重新提出后续动作。输入状态和探测任务均存在差异，"
              "配对 AUROC 比较仅比较这些操作性定义下的信号，不能把差异全部归因于信号机制本身。", "",
              f"本轮采样模式：**{mode}**。协议记录 consistency_thinking=`{thinking}`，"
              f"self_report_thinking=`{protocol.get('self_report_thinking')}`，K=`{protocol.get('consistency_k')}`，"
              f"temperature=`{protocol.get('consistency_temperature')}`，top_p=`{protocol.get('consistency_top_p')}`，"
              f"max_tokens=`{protocol.get('max_tokens')}`。"
              "原 logprob 来自旧 thinking 策略产生的动作；本轮结果可诊断新增打分器的价值，"
              "不能声称复现了原始生成策略，或把 AUROC 差单独归因于一种置信度机制。", "",
              "这是探索性筛选，复用的标签和任务已被审阅，没有独立留出集，也没有训练新的置信度模型。"
              "一致地重复同一个错误仍可能给高分；多个合理的后续动作可能拉低一致性。", "",
              f"一致性 K 分布（有分数步骤）：`{json.dumps(result['consistency_k_distribution'], ensure_ascii=False)}`。"
              f"非法分数计数：`{json.dumps(result['input_audit']['invalid_scores'], ensure_ascii=False)}`。", "",
              "主一致性是各独立样本与原候选调用在预定语义归一化下的匹配比例，不是样本内部的多数派占比。"
              "匹配要求相同 API 和语义参数：规范化数值、时间、安全算式、个人地址末尾句点；"
              "EmailReminder.content 是自由文本，主指标排除该字段，保留收件人、时间、地点等其余字段。"
              "严格一致性若提供则单独作为次要诊断，不据结果替换主信号。无工具调用属于有效的未匹配，截断另行报告；"
              "传输失败导致样本缺失时，整步一致性保持缺失，不能缩小分母。"
              "若响应含多个工具调用，沿用原运行器只取首个调用；严格一致性保留全部参数字段（包括自由文本），"
              "按 JSON 值比较并忽略对象键顺序。无工具调用或 length 截断且未解析出调用记为不一致。"
              "每步只有 3 次采样时，分数仅有 0、1/3、2/3、1 四档，适合初筛，精度和稳定性有限。", ""]
    quality, cost = diagnostics["quality"], diagnostics["cost"]
    total = quality["totals"]
    samples = quality["reported_sample_counts"]

    def count(value):
        return "未知" if value is None else str(value)

    lines += ["采集质量与成本单独统计，不影响上述样本、标签或信号分数：", "",
              f"- 已汇总 {quality['score_rows']} 条步骤评分，成功取得采样 {total['n_samples']} 次；"
              f"与原候选语义一致 {total['n_agree']} 次，严格一致 {total['n_strict_agree']} 次。",
              f"- 逐步计数中，无工具调用 {total['n_no_tool']} 次（{number(quality['no_tool_fraction_of_counted_samples'])}），"
              f"length 截断 {total['n_truncated']} 次（{number(quality['truncation_fraction_of_counted_samples'])}）；"
              "二者可重叠，不相加当作独立错误数。"
              f"计数已知步骤中缺少采样 {quality['missing_samples_with_known_count']} 次。"]
    if diagnostics["summary"]["file_present"]:
        lines += [f"- 公开采集摘要：成功请求 {count(cost['successful_requests'])} 次，失败请求 {count(cost['failed_requests'])} 次；"
                  f"独立采样 {count(samples['samples'])} 次，其中无工具调用 {count(samples['no_tool_samples'])} 次、"
                  f"多个工具调用 {count(samples['multiple_tool_samples'])} 次、截断 {count(samples['truncated_samples'])} 次。",
                  f"- 已记录 usage 的响应 {count(cost['usage_records'])} 条：prompt tokens {count(cost['prompt_tokens'])}，"
                  f"completion tokens {count(cost['completion_tokens'])}，total tokens {count(cost['total_tokens'])}；"
                  f"另有 {count(cost['successful_responses_without_usage'])} 条成功响应缺 usage。"]
    else:
        lines += ["- 未提供 collection_summary.json，请求次数和 token 成本未知；不以 0 次模型调用代替采集成本。"]
    lines += ["", "质量字段的逐步覆盖情况保存在 JSON。请求与 token 成本来自公开的 collection_summary.json，"
              "仅用于描述采集开销；未知计数保留未知。评估不读取运行时日志。", "",
              "![同样本新信号对比](new_signal_comparison.png)", "",
              "产物：`new_signal_metrics.json`（完整覆盖率/配对区间/去重）、`new_signal_metrics.csv`（各队列指标）、"
              "`joined_scores.csv`（逐步对应，不含新增标签）、`new_signal_comparison.png`。", "",
              "在项目根目录运行以下评估命令，仅做 CPU 离线统计，不会自动启动模型或发起推理：", "",
              "```bash", "python code/evaluate_new_signals.py "
              f"--labels {shlex.quote(result['paths']['labels'])} "
              f"--scores {shlex.quote(result['paths']['scores'])} "
              f"--out-dir {shlex.quote(result['paths']['out_dir'])} "
              f"--bootstrap {result['bootstrap']['replicates']} --seed {result['bootstrap']['seed']}", "```", "",
              "如需重新采集 fast 模式信号，须自行准备支持工具调用的模型服务。使用公开状态和新的输出目录，避免覆盖冻结结果。下面的采集命令会发起模型推理：", "",
              "```bash", "python code/collect_new_signals.py --states outputs/new_signals_fast_v1/states.jsonl --out-dir outputs/my_signals --endpoints http://localhost:8000/v1 --no-consistency-thinking --cooldown 1", "```"]
    (out / "新信号评估.md").write_text("\n".join(lines)+"\n")


def make_figure(out, rows, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    colors = {"logprob": "#64748b", "self_report": "#2563eb", "consistency": "#d97706"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    common = result["cohorts"]["all_calls"]["common_all"]
    complete = [r for r in rows if all(r[s] is not None for s in SIGNALS)]
    a = axes[0, 0]
    for i, name in enumerate(SIGNALS):
        m = common["signals"][name]
        val, ci = m["AUROC"], m["AUROC_bootstrap"]["ci95"]
        if val is not None:
            a.bar(i, val, color=colors[name], width=.5)
            a.text(i, val+.035, f"{val:.3f}", ha="center")
        if ci:
            a.plot([i, i], ci, color="black")
            for value in ci:
                a.plot([i-.06, i+.06], [value, value], color="black")
    a.axhline(.5, color="gray", linestyle="--")
    a.set(xticks=range(3), xticklabels=SIGNALS, ylim=(0, 1.07), ylabel="AUROC",
          title=f"Same {common['n']} calls; task bootstrap 95% CI")
    a = axes[0, 1]
    yy = np.array([r["y"] for r in complete])
    if len(set(yy)) == 2:
        for name in SIGNALS:
            pp = np.array([r[name] for r in complete])
            fpr, tpr, _ = roc_curve(yy, pp)
            a.plot(fpr, tpr, label=name, color=colors[name])
        a.legend(fontsize=8)
    a.plot([0, 1], [0, 1], "--", color="gray")
    a.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC on shared complete cases")
    a = axes[1, 0]
    for i, name in enumerate(SIGNALS):
        val = common["signals"][name]["error_capture_at_20pct"]
        if val is not None:
            a.bar(i, val, color=colors[name], width=.5)
            a.text(i, val+.025, f"{val:.3f}", ha="center")
    a.axhline(.2, color="gray", linestyle="--", label="Random verification")
    a.set(xticks=range(3), xticklabels=SIGNALS, ylim=(0, 1.05), ylabel="Fraction of errors captured",
          title="Verify the lowest-scored 20%; ties averaged")
    a.legend(fontsize=8)
    a = axes[1, 1]
    coverage = [result["cohorts"]["all_calls"]["own_coverage"][s]["n"] for s in SIGNALS]
    a.bar(range(3), coverage, color=[colors[s] for s in SIGNALS], width=.5)
    for i, n in enumerate(coverage):
        a.text(i, n+.03*max(1, len(rows)), str(n), ha="center")
    a.set(xticks=range(3), xticklabels=SIGNALS, ylim=(0, max(1, len(rows))*1.15), ylabel="Scored calls",
          title=f"Coverage of {len(rows)} frozen call labels")
    fig.suptitle("New confidence signals on reconstructed legacy states (uncalibrated)")
    fig.savefig(out / "new_signal_comparison.png", dpi=170)
    plt.close(fig)


def write_tables(out, rows, result):
    table = []
    for population, cohort in result["cohorts"].items():
        blocks = [(f"own_{s}", co) for s, co in cohort["own_coverage"].items()]
        blocks += [(f"paired_{s}", co) for s, co in cohort["paired_vs_logprob"].items()]
        blocks += [("common_all", cohort["common_all"])]
        if "secondary_strict_consistency" in cohort:
            blocks += [("secondary_strict_consistency", cohort["secondary_strict_consistency"])]
        for block, co in blocks:
            for signal, stat in co["signals"].items():
                ci = stat["AUROC_bootstrap"]["ci95"]
                delta = co["paired_AUROC_difference_vs_logprob"].get(signal, {})
                dci = delta.get("ci95")
                table.append({"population": population, "comparison": block, "signal": signal,
                              "coverage": co["coverage"], "tasks": co["tasks"],
                              **{k: stat[k] for k in ("n", "correct", "errors", "AUROC", "error_AUPRC", "Brier", "ECE", "error_capture_at_20pct")},
                              "AUROC_CI_low": ci[0] if ci else None, "AUROC_CI_high": ci[1] if ci else None,
                              "AUROC_delta_vs_logprob": delta.get("point"),
                              "delta_CI_low": dci[0] if dci else None, "delta_CI_high": dci[1] if dci else None})
    for filename, records in [("new_signal_metrics.csv", table), ("joined_scores.csv", rows)]:
        with (out / filename).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]) if records else ["task_id", "step_id"])
            writer.writeheader()
            writer.writerows(records)


def evaluate(labels_path, scores_path, out_dir, n_boot=1000, seed=SEED):
    labels_path, scores_path, out_dir = Path(labels_path), Path(scores_path), Path(out_dir)
    calls, scores = read_jsonl(labels_path), read_jsonl(scores_path)
    summary_path = labels_path.parent / "label_summary.json"
    expected = None
    if summary_path.exists():
        expected = json.loads(summary_path.read_text()).get("artifact_sha256", {}).get(labels_path.name)
        if expected and sha(labels_path) != expected:
            raise ValueError("Frozen call labels no longer match label_summary.json")
    rows, audit = join_scores(calls, scores, require_history_flags=bool(expected))
    result = {"version": "reconstructed-new-signals-v1", "positive_label": "correct",
              "score_direction": "higher_is_more_correct", "new_calibrator_fitted": False,
              "evaluation_gpu_used": False, "evaluation_model_requests": 0,
              "labels_sha256": sha(labels_path), "scores_sha256": sha(scores_path),
              "evaluation_code_sha256": sha(__file__), "input_audit": audit,
              "paths": {"labels": relative_path(labels_path), "scores": relative_path(scores_path),
                        "out_dir": relative_path(out_dir)},
              "bootstrap": {"replicates": n_boot, "seed": seed, "unit": "task"},
              "consistency_k_distribution": dict(Counter(str(r["consistency_k"]) for r in rows if r["consistency"] is not None)),
              "collection_diagnostics": collection_diagnostics(scores, scores_path.parent),
              "error_type_diagnostics": error_type_diagnostics(rows),
              "cohorts": evaluate_rows(rows, n_boot, seed)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "new_signal_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    write_tables(out_dir, rows, result)
    make_figure(out_dir, rows, result)
    write_report(out_dir, result)
    print(json.dumps({s: {"n": result["cohorts"]["all_calls"]["own_coverage"][s]["n"],
                           "AUROC": result["cohorts"]["all_calls"]["own_coverage"][s]["signals"][s]["AUROC"]}
                      for s in SIGNALS}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=str(ROOT / "outputs/legacy_binary_v2/calls_binary.jsonl"))
    parser.add_argument("--scores", default=str(ROOT / "outputs/new_signals_fast_v1/scores.jsonl"))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/new_signals_fast_v1"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be >= 1")
    evaluate(args.labels, args.scores, args.out_dir, args.bootstrap, args.seed)
