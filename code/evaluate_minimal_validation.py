#!/usr/bin/env python
"""Evaluate newly collected, score-blind labels with a frozen calibrator.

No fitting, model requests or tool executions occur here. Each signal reports
its actual coverage, and matched cohorts make probability metrics comparable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from confidence_runtime import calibrate_consistency, load_calibrator

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261019
SIGNALS = ("raw_consistency", "p_correct", "training_prior", "logprob")
DISPLAY = {"raw_consistency": "原始一致性", "p_correct": "冻结校准概率",
           "training_prior": "训练集常数先验", "logprob": "原 logprob"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def probability(value):
    return float(value) if type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 else None


def row_key(row):
    """Accept explicit identifiers or a task:step key; never silently disagree."""
    explicit = None
    if "task_id" in row or "step_id" in row:
        if (not isinstance(row.get("task_id"), str) or not row["task_id"]
                or type(row.get("step_id")) is not int or row["step_id"] < 1):
            raise ValueError("Each row needs a nonempty task_id and integer step_id >= 1")
        explicit = row["task_id"], row["step_id"]
    alternate = None
    if "key" in row:
        value = row["key"]
        if isinstance(value, str) and ":" in value:
            task, step = value.rsplit(":", 1)
            try:
                alternate = task, int(step)
            except ValueError as exc:
                raise ValueError("Label key must be task_id:integer_step_id") from exc
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            alternate = tuple(value)
        else:
            raise ValueError("Invalid label key")
        if (not isinstance(alternate[0], str) or not alternate[0]
                or type(alternate[1]) is not int or alternate[1] < 1):
            raise ValueError("Invalid task/step label key")
    if explicit is not None and alternate is not None and explicit != alternate:
        raise ValueError("Explicit identifiers and key disagree")
    if explicit is None and alternate is None:
        raise ValueError("Row is missing task/step identifiers")
    return explicit or alternate


def join_validation(steps, labels, calibration):
    """Join every new step exactly once and verify stored frozen probabilities."""
    prior = probability(calibration.get("training_prior"))
    if prior is None:
        raise ValueError("Frozen calibrator must supply its training_prior")
    step_map, label_map = {}, {}
    for target, rows, name in ((step_map, steps, "step"), (label_map, labels, "label")):
        for row in rows:
            key = row_key(row)
            if key in target:
                raise ValueError("Duplicate " + name + " key: " + str(key))
            target[key] = row
    if not step_map or set(step_map) != set(label_map):
        raise ValueError("New validation steps and labels must have identical, nonempty key sets")
    joined, quality = [], Counter()
    for key, step in step_map.items():
        label = label_map[key]
        if type(label.get("y")) is not int or label["y"] not in (0, 1):
            raise ValueError("Validation label must be binary: " + str(key))
        confidence = step.get("confidence")
        if not isinstance(confidence, dict):
            raise ValueError("New step is missing confidence output")
        raw_value, calibrated_value = confidence.get("raw_consistency"), confidence.get("p_correct")
        raw, calibrated = probability(raw_value), probability(calibrated_value)
        if raw_value is not None and raw is None or calibrated_value is not None and calibrated is None:
            raise ValueError("Invalid raw/calibrated probability: " + str(key))
        if raw is not None and not math.isclose(raw * 3, round(raw * 3), abs_tol=1e-9):
            raise ValueError("Raw agreement must follow the frozen K=3 grid")
        expected = calibrate_consistency(raw, calibration)
        if calibrated is None and expected["p_correct"] is not None or calibrated is not None and expected["p_correct"] is None:
            raise ValueError("Raw score and calibrated probability have different coverage")
        if calibrated is not None and not math.isclose(calibrated, expected["p_correct"], rel_tol=0, abs_tol=1e-10):
            raise ValueError("Stored probability does not match the frozen calibrator")
        if (type(confidence.get("needs_verification")) is not bool
                or confidence["needs_verification"] != expected["needs_verification"]):
            raise ValueError("Stored verification decision disagrees with the frozen policy")
        raw_lp = step.get("raw_logprobs") or {}
        logprob = None
        if isinstance(raw_lp, dict) and raw_lp.get("available") is True and not raw_lp.get("degenerate"):
            logprob = probability(raw_lp.get("geo_mean_prob"))
        quality["missing_consistency"] += raw is None
        quality["missing_logprob"] += logprob is None
        flags = confidence.get("quality") or {}
        for field in ("n_no_tool", "n_truncated", "n_multiple_tool_calls", "n_invalid_tool_arguments",
                      "n_transport_failures", "n_invalid_responses", "n_gate_blocks"):
            value = flags.get(field)
            if type(value) is int and value >= 0:
                quality[field] += value
        joined.append({"task_id": key[0], "step_id": key[1], "y": label["y"],
                       "reason": label.get("reason", ""), "raw_consistency": raw,
                       "p_correct": calibrated, "training_prior": prior, "logprob": logprob,
                       "needs_verification": expected["needs_verification"]})
    return joined, dict(quality)


def probability_metrics(rows, signal, bins=10):
    available = [row for row in rows if row.get(signal) is not None]
    y = np.array([row["y"] for row in available], dtype=int)
    p = np.array([row[signal] for row in available], dtype=float)
    result = {"n": len(available), "n_total": len(rows), "n_missing": len(rows) - len(available),
              "coverage": len(available) / len(rows) if rows else None,
              "positive": int(y.sum()), "negative": int(len(y) - y.sum()),
              "AUROC": None, "AUROC_unavailable_reason": None,
              "Brier": None, "ECE": None, "NLL": None, "reliability_bins": []}
    if not available:
        result["AUROC_unavailable_reason"] = "no_scored_steps"
        return result
    if len(set(y)) == 2:
        result["AUROC"] = float(roc_auc_score(y, p))
    else:
        result["AUROC_unavailable_reason"] = "single_class"
    result["Brier"] = float(np.mean((p - y) ** 2))
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    result["NLL"] = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))
    result["ECE"] = 0.
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = (p >= low) & ((p < high) if index < bins - 1 else (p <= high))
        if not selected.any():
            continue
        mean, accuracy = float(p[selected].mean()), float(y[selected].mean())
        count = int(selected.sum())
        result["ECE"] += count / len(y) * abs(mean - accuracy)
        result["reliability_bins"].append({"low": low, "high": high, "n": count,
                                           "mean_probability": mean, "accuracy": accuracy})
    return result


def verification_metrics(rows):
    selected = [row for row in rows if row["needs_verification"]]
    errors = sum(1 - row["y"] for row in rows)
    caught = sum(1 - row["y"] for row in selected)
    return {"rule": "raw_consistency < 1; missing score also triggers verification",
            "n": len(rows), "errors": errors, "selected": len(selected), "captured_errors": caught,
            "coverage": len(selected) / len(rows) if rows else None,
            "error_capture_rate": caught / errors if errors else None,
            "verification_precision": caught / len(selected) if selected else None,
            "selected_correct_steps": sum(row["y"] for row in selected),
            "missing_score_selected": sum(row["raw_consistency"] is None for row in selected)}


def budget_metrics(rows, signal, fraction=.2, seed=SEED, missing_first=False):
    """Strict floor budget; tie randomization uses no labels or score outcomes."""
    scored = list(rows) if missing_first else [row for row in rows if row.get(signal) is not None]
    ordered = sorted(scored, key=lambda row: (row["task_id"], row["step_id"]))
    random.Random(seed).shuffle(ordered)
    score = lambda row: float("-inf") if row.get(signal) is None else row[signal]
    ordered.sort(key=score)
    budget = math.floor(len(scored) * fraction)
    selected = ordered[:budget]
    errors = sum(1 - row["y"] for row in scored)
    caught = sum(1 - row["y"] for row in selected)
    expected_caught = 0.
    remainder, index = budget, 0
    while remainder and index < len(ordered):
        end = index + 1
        while end < len(ordered) and score(ordered[end]) == score(ordered[index]):
            end += 1
        group = ordered[index:end]
        take = min(remainder, len(group))
        expected_caught += take / len(group) * sum(1 - row["y"] for row in group)
        remainder -= take
        index = end
    return {"signal": signal, "fraction": fraction, "seed": seed, "budget_rule": "floor(fraction * n)",
            "n": len(scored), "excluded_missing": len(rows) - len(scored),
            "missing_first": missing_first, "budget": budget,
            "actual_coverage": budget / len(scored) if scored else None,
            "errors": errors, "captured_errors_fixed_seed": caught,
            "error_capture_rate_fixed_seed": caught / errors if errors else None,
            "captured_errors_random_tie_expectation": expected_caught,
            "error_capture_rate_random_tie_expectation": expected_caught / errors if errors else None,
            "random_baseline_capture_rate": budget / len(scored) if scored and errors else None,
            "selected_keys": [f'{row["task_id"]}:{row["step_id"]}' for row in selected]}


def evaluate_rows(rows, quality=None):
    cohorts = {"all_steps": rows,
               "matched_consistency": [row for row in rows if row["raw_consistency"] is not None],
               "matched_all_signals": [row for row in rows if all(row[signal] is not None for signal in SIGNALS)]}
    metrics = {"version": "minimal-validation-evaluation-v1", "seed": SEED,
               "scope": "new-task integration validation with frozen calibration; no refitting",
               "n_steps": len(rows), "n_tasks": len({row["task_id"] for row in rows}),
               "positive": sum(row["y"] for row in rows), "negative": sum(1 - row["y"] for row in rows),
               "step_accuracy": sum(row["y"] for row in rows) / len(rows) if rows else None,
               "quality": quality or {}, "cohorts": {},
               "fixed_verification_rule": verification_metrics(rows),
               "lowest_20_percent_including_missing_first": budget_metrics(rows, "raw_consistency", missing_first=True),
               "probability_metrics": {"ECE": "10 equal-width bins", "NLL_clip": 1e-12,
                                       "ranking_direction": "larger means more correct"}}
    for name, subset in cohorts.items():
        signals = SIGNALS if name != "matched_consistency" else SIGNALS[:-1]
        metrics["cohorts"][name] = {"n": len(subset),
                                    "positive": sum(row["y"] for row in subset),
                                    "negative": sum(1 - row["y"] for row in subset),
                                    "signals": {signal: probability_metrics(subset, signal) for signal in signals},
                                    "lowest_20_percent": {signal: budget_metrics(subset, signal) for signal in signals}}
    matched = metrics["cohorts"]["matched_consistency"]["signals"]
    calibrated_brier, prior_brier = matched["p_correct"]["Brier"], matched["training_prior"]["Brier"]
    metrics["limitations"] = {"negative_steps_below_10": metrics["negative"] < 10,
                              "calibrated_brier_exceeds_training_prior":
                              calibrated_brier > prior_brier if calibrated_brier is not None and prior_brier is not None else None}
    return metrics


def run_diagnostics(out, metrics, rows=None):
    """Describe executed/zero-call tasks and model cost without inferring success."""
    out = Path(out)
    trajectories_path, complete_path = out / "validation_trajectories.jsonl", out / "validation_complete.json"
    result = {"executed_tasks": None, "zero_call_tasks": None,
              "tasks_with_tool_calls": metrics["n_tasks"], "task_success_evaluated": False,
              "completion_marker_present": complete_path.is_file(), "complete": None}
    if complete_path.is_file():
        completion = json.loads(complete_path.read_text(encoding="utf-8"))
        if completion.get("complete") is not True or completion.get("calibrator_unchanged") is not True:
            raise ValueError("Validation is incomplete or the frozen calibrator changed")
        for field, expected in (("steps", metrics["n_steps"]), ("correct", metrics["positive"]), ("errors", metrics["negative"])):
            if completion.get(field) != expected:
                raise ValueError("Completion marker disagrees with joined labels: " + field)
        result["complete"] = True
    else:
        completion = None
    if trajectories_path.is_file():
        trajectories = read_jsonl(trajectories_path)
        ids = [row.get("task_id") for row in trajectories]
        if (any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids)
                or any(type(row.get("steps")) is not int or row["steps"] < 0 for row in trajectories)):
            raise ValueError("Task trajectories require unique task IDs and nonnegative integer step counts")
        if sum(row["steps"] for row in trajectories) != metrics["n_steps"]:
            raise ValueError("Task trajectory step totals disagree with the scored step file")
        if rows is not None:
            step_counts = Counter(row["task_id"] for row in rows)
            if not set(step_counts).issubset(ids) or any(step_counts[row["task_id"]] != row["steps"] for row in trajectories):
                raise ValueError("Per-task trajectory steps disagree with the scored step file")
        if completion is not None and completion.get("tasks") != len(trajectories):
            raise ValueError("Completion marker task count disagrees with trajectories")
        result.update(executed_tasks=len(trajectories), zero_call_tasks=sum(row["steps"] == 0 for row in trajectories),
                      stop_reasons=dict(Counter(row.get("stop_reason", "unknown") for row in trajectories)),
                      real_tool_executions=sum(row.get("real_tool_executions", 0) for row in trajectories),
                      recommendations_applied=any(row.get("recommendations_applied") is True for row in trajectories))
    outcomes_path = out / "validation_task_outcomes.jsonl"
    if outcomes_path.is_file():
        if not trajectories_path.is_file():
            raise ValueError("Task outcomes require task trajectories for complete task-ID alignment")
        outcomes = read_jsonl(outcomes_path)
        outcome_ids = [row.get("task_id") for row in outcomes]
        if (any(not isinstance(task_id, str) or not task_id for task_id in outcome_ids)
                or len(outcome_ids) != len(set(outcome_ids)) or set(outcome_ids) != set(ids)):
            raise ValueError("Task outcomes must align exactly with unique trajectory task IDs")
        for row in outcomes:
            if any(type(row.get(field)) is not int or row[field] not in (0, 1)
                   for field in ("task_success", "stop_event_y")):
                raise ValueError("Task success and stop-event labels must be binary integers")
        successes = sum(row["task_success"] for row in outcomes)
        correct_stops = sum(row["stop_event_y"] for row in outcomes)
        result["task_success_evaluated"] = True
        result["task_outcomes"] = {
            "n_tasks": len(outcomes), "successful_tasks": successes, "failed_tasks": len(outcomes) - successes,
            "task_success_rate": successes / len(outcomes) if outcomes else None,
            "correct_stop_events": correct_stops, "incorrect_stop_events": len(outcomes) - correct_stops,
            "stop_event_accuracy": correct_stops / len(outcomes) if outcomes else None,
            "failure_reason_codes": dict(Counter(row.get("reason_code", "unknown")
                                                  for row in outcomes if row["task_success"] == 0))}
    requests_path = out / "validation_requests.jsonl"
    if requests_path.is_file():
        attempts = read_jsonl(requests_path)
        cost = {"source": "validation_requests.jsonl", "n_model_requests": len(attempts), "by_kind": {}}
        for attempt in attempts:
            kind = str(attempt.get("kind", "unknown"))
            group = cost["by_kind"].setdefault(kind, {"requests": 0, "successful_responses": 0,
                                                       "usage_records": 0, "prompt_tokens": 0,
                                                       "completion_tokens": 0, "total_tokens": 0,
                                                       "latency_sum_s": 0.})
            group["requests"] += 1
            response = attempt.get("response")
            if isinstance(response, dict):
                group["successful_responses"] += 1
                usage = response.get("usage")
                if isinstance(usage, dict):
                    group["usage_records"] += 1
                    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        value = usage.get(field)
                        if type(value) is int and value >= 0:
                            group[field] += value
            latency = attempt.get("latency_s")
            if type(latency) in (int, float) and math.isfinite(latency) and latency >= 0:
                group["latency_sum_s"] += latency
        cost["total_tokens_with_usage"] = sum(group["total_tokens"] for group in cost["by_kind"].values())
        result["model_cost"] = cost
    elif (out / "collection_summary.json").is_file():
        # Public packages retain curated research counts rather than request
        # logs. Only the explicitly listed fields enter the output metrics.
        loaded = json.loads((out / "collection_summary.json").read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or loaded.get("version") != "minimal-confidence-public-collection-summary-v1":
            raise ValueError("Unsupported public collection summary")
        total = loaded.get("n_model_requests")
        by_kind = loaded.get("by_kind")
        if type(total) is not int or total < 0 or not isinstance(by_kind, dict):
            raise ValueError("Invalid public collection request counts")
        fields = ("requests", "successful_responses", "usage_records", "prompt_tokens", "completion_tokens", "total_tokens")
        clean = {}
        for kind, group in by_kind.items():
            if kind not in ("primary", "confidence_probe") or not isinstance(group, dict):
                raise ValueError("Unsupported public collection request kind")
            if any(type(group.get(field)) is not int or group[field] < 0 for field in fields):
                raise ValueError("Public collection counts must be nonnegative integers")
            if not group["usage_records"] <= group["successful_responses"] <= group["requests"]:
                raise ValueError("Inconsistent public collection response counts")
            if group["prompt_tokens"] + group["completion_tokens"] != group["total_tokens"]:
                raise ValueError("Inconsistent public collection token counts")
            clean[kind] = {field: group[field] for field in fields}
        token_total = loaded.get("total_tokens_with_usage")
        if (sum(group["requests"] for group in clean.values()) != total
                or type(token_total) is not int or token_total != sum(group["total_tokens"] for group in clean.values())):
            raise ValueError("Public collection summary totals disagree")
        result["model_cost"] = {"source": "collection_summary.json", "n_model_requests": total,
                                "by_kind": clean, "total_tokens_with_usage": token_total}
    return result


def figure(out, rows, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"raw_consistency": "#6366f1", "p_correct": "#0d9488", "training_prior": "#a16207", "logprob": "#dc2626"}
    names = {"raw_consistency": "Raw agreement", "p_correct": "Frozen sigmoid", "training_prior": "Training prior", "logprob": "Token logprob"}
    cohort_name = "matched_all_signals" if metrics["cohorts"]["matched_all_signals"]["n"] else "matched_consistency"
    cohort = metrics["cohorts"][cohort_name]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = axes[0, 0]
    for signal, values in cohort["signals"].items():
        bins = values["reliability_bins"]
        if bins:
            ax.plot([b["mean_probability"] for b in bins], [b["accuracy"] for b in bins],
                    "o-", color=colors[signal], label=names[signal])
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed step accuracy",
           title=f"Reliability on the same {cohort['n']} scored steps")
    if ax.lines:
        ax.legend(fontsize=8)
    ax = axes[0, 1]
    for label, shift, color in ((1, -.065, "#2563eb"), (0, .065, "#dc2626")):
        counts = [sum(row["y"] == label and row["raw_consistency"] is not None
                      and math.isclose(row["raw_consistency"], value, abs_tol=1e-9) for row in rows)
                  for value in (0, 1 / 3, 2 / 3, 1)]
        ax.bar(np.array([0, 1 / 3, 2 / 3, 1]) + shift, counts, width=.12,
               color=color, label="Correct" if label else "Incorrect")
    ax.set(xticks=[0, 1 / 3, 2 / 3, 1], xticklabels=["0/3", "1/3", "2/3", "3/3"],
           xlabel="Independent samples agreeing with candidate", ylabel="Steps", title="New step labels and raw agreement")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    scored = [(signal, value["Brier"]) for signal, value in cohort["signals"].items() if value["Brier"] is not None]
    ax.bar([names[signal] for signal, _ in scored], [value for _, value in scored], color=[colors[signal] for signal, _ in scored])
    ax.set(ylabel="Brier score (lower is better)", title=f"Frozen probability quality; matched n={cohort['n']}")
    ax.tick_params(axis="x", labelsize=8)
    ax = axes[1, 1]
    fixed = metrics["fixed_verification_rule"]
    budget = metrics["lowest_20_percent_including_missing_first"]
    capture = [fixed["error_capture_rate"], budget["error_capture_rate_random_tie_expectation"]]
    coverage = [fixed["coverage"], budget["actual_coverage"]]
    positions = np.arange(2)
    ax.bar(positions - .18, [value or 0 for value in coverage], .35, label="Steps selected", color="#64748b")
    ax.bar(positions + .18, [value or 0 for value in capture], .35, label="Errors captured", color="#0d9488")
    ax.set(xticks=positions, xticklabels=["Raw < 1 / missing", "Lowest 20% budget"], ylim=(0, 1),
           ylabel="Fraction", title=f"Verification allocation; n_errors={fixed['errors']}\nBudget uses tie expectation")
    if fixed["errors"] == 0:
        ax.text(.5, .9, "No errors: capture is undefined", transform=ax.transAxes, ha="center", fontsize=9)
    ax.legend(fontsize=8)
    actual_tasks = metrics.get("run_diagnostics", {}).get("executed_tasks")
    task_label = f"{actual_tasks} tasks ({metrics['n_tasks']} with calls)" if actual_tasks is not None else f"{metrics['n_tasks']} tasks with calls"
    fig.suptitle(f"Frozen confidence API: {task_label}, {metrics['n_steps']} steps, {metrics['negative']} errors", fontsize=12)
    fig.savefig(Path(out) / "validation_diagnostic.png", dpi=170)
    plt.close(fig)


def _number(value, digits=4):
    return "不可计算" if value is None else f"{value:.{digits}f}"


def report(out, metrics):
    fixed = metrics["fixed_verification_rule"]
    lines = ["# 新任务上的冻结置信度接口验证", "",
             f"本报告包含 {metrics['n_tasks']} 个有工具调用的任务、{metrics['n_steps']} 次工具调用："
             f"{metrics['positive']} 正确、{metrics['negative']} 错误，步骤正确率 {_number(metrics['step_accuracy'])}。", "",
             "校准器在采集新任务之前冻结；本脚本只连接分数与二元标签、计算指标，不重新拟合，也不调用模型或执行工具。"
             "这是新任务上的集成验证，不能解释为大规模独立泛化证明。", ""]
    design = metrics.get("collection_design", {})
    if design.get("source_file"):
        lines += [f"任务来源：`{design['source_file']}`。与此前 level-3 需求的完全重叠数为 "
                  f"{design.get('old_level3_requirement_overlap', '未提供')}。", ""]
    if design.get("extension_rule"):
        lines += [f"预先冻结的样本量约定：先跑 {design.get('initial_tasks')} 个任务；"
                  f"若调用错误不足 10 个，按预选顺序逐个追加任务，达到 10 个错误即停，最多 {design.get('maximum_tasks')} 个。"
                  "触发条件涉及错误数量，因此不能当作固定样本量的大规模测试；不按 AUROC 挑选或剔除任务。", ""]
    run = metrics.get("run_diagnostics", {})
    if run.get("executed_tasks") is not None:
        lines += [f"实际已执行 {run['executed_tasks']} 个任务，其中 {run['zero_call_tasks']} 个没有工具调用；"
                  f"{run['tasks_with_tool_calls']} 个任务贡献本报告的工具步骤。这些数量不是任务成功率。", ""]
    if run.get("task_success_evaluated"):
        outcomes = run["task_outcomes"]
        lines += ["## 独立的任务完成与结束标签", "",
                  f"按屏蔽分数的审查资料复核全部 {outcomes['n_tasks']} 个任务：**{outcomes['successful_tasks']} 任务成功／交付合格、"
                  f"{outcomes['failed_tasks']} 任务失败**，任务成功率 {_number(outcomes['task_success_rate'])}。"
                  f"结束结果标签另计 {outcomes['correct_stop_events']} 正确、{outcomes['incorrect_stop_events']} 错误。", "",
                  f"这组任务／结束结果与 {metrics['n_steps']} 次工具调用的 {metrics['positive']} 正、{metrics['negative']} 负分开统计，"
                  "不能混成同一准确率。失败包括目标已完成但额外执行或终答出错的情况，不能全称为核心目标未完成。"
                  "本轮使用不同任务，不能与此前旧 50 个任务的成功率直接比较或据此声称能力提升。", "",
                  "| 任务失败原因 | 次数 |", "| --- | ---: |"]
        names = {"generation_truncated": "生成截断，未完成任务",
                 "unsupported_final_claim_and_unrequested_fabricated_call": "编造额外调用参数，并误述信息来源",
                 "finite_mock_no_recorded_response": "有限 mock 无匹配响应，任务未实际执行成功",
                 "unsupported_cost_period_claim": "费用周期的断言缺少依据"}
        for reason, count in outcomes["failure_reason_codes"].items():
            lines.append(f"| {names.get(reason, reason)}（`{reason}`） | {count} |")
        lines += [""]
    lines += ["零工具调用任务没有可评价的工具步骤，不进入上述步级指标；任务完成率也不与步骤正确率混用。", "",
             "## 1. 各信号的覆盖与指标", "",
             "各行先报告自身可用样本；缺失值不补 0。AUROC 越大越好，Brier、ECE、NLL 越小越好。"
             "ECE 使用 10 个等宽区间，NLL 仅为避免 log(0) 将概率截到 [1e-12, 1-1e-12]。", "",
             "| 信号 | 可用 / 全部 | 正 / 负 | AUROC | Brier | ECE | NLL |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for signal, values in metrics["cohorts"]["all_steps"]["signals"].items():
        lines.append(f"| {DISPLAY[signal]} | {values['n']} / {values['n_total']} | {values['positive']} / {values['negative']} | "
                     f"{_number(values['AUROC'])} | {_number(values['Brier'])} | {_number(values['ECE'])} | {_number(values['NLL'])} |")
    for cohort_name, title in (("matched_consistency", "一致性与校准概率、常数先验的同样本比较"),
                               ("matched_all_signals", "包含 logprob 的四信号同样本比较")):
        cohort = metrics["cohorts"][cohort_name]
        lines += ["", f"### {title}", "", f"共 {cohort['n']} 步（{cohort['positive']} 正、{cohort['negative']} 负）。", "",
                  "| 信号 | AUROC | Brier | ECE | NLL |", "| --- | ---: | ---: | ---: | ---: |"]
        for signal, values in cohort["signals"].items():
            lines.append(f"| {DISPLAY[signal]} | {_number(values['AUROC'])} | {_number(values['Brier'])} | "
                         f"{_number(values['ECE'])} | {_number(values['NLL'])} |")
    if metrics["positive"] == 0 or metrics["negative"] == 0:
        lines += ["", "当前新步骤只有一个类别，AUROC 不可计算；不把不可计算写成 0.5。"]
    if metrics["negative"] < 10:
        lines += ["", f"**本轮负例仅 {metrics['negative']} 个，少于 10 个；AUROC 和错误捕获率很不稳定，"
                  "单个错误就可能显著改变数值。即使捕获率为 100%，也不能据此证明稳定的区分或泛化能力。**"]
    if metrics.get("limitations", {}).get("calibrated_brier_exceeds_training_prior"):
        compared = metrics["cohorts"]["matched_consistency"]["signals"]
        lines += ["", f"**同样本上，冻结校准概率的 Brier 为 {_number(compared['p_correct']['Brier'])}，"
                  f"高于训练集常数先验的 {_number(compared['training_prior']['Brier'])}。"
                  "当前结果不能声称新域概率优于常数基线，也不能据此宣称校准泛化成功。**"]
    lines += ["", "正斜率 sigmoid 保持原分数排序，因而通常不会提高 AUROC；校准应主要检查概率误差是否下降。"
              "常数先验固定取训练 198 步的正确率，不使用新标签估计先验。", "",
              "## 2. 验证资源分配", "",
              f"冻结规则为 `raw_consistency < 1`；缺失分数也触发验证。选中 {fixed['selected']} / {fixed['n']} 步，"
              f"覆盖率 {_number(fixed['coverage'])}；捕获 {fixed['captured_errors']} / {fixed['errors']} 个错误，"
              f"错误捕获率 {_number(fixed['error_capture_rate'])}；选中步骤中的错误占比 {_number(fixed['verification_precision'])}。", "",
              "最低 20% 采用严格 `floor(0.2 × 可用步数)` 预算，同分按预先固定种子 20261019 打乱，"
              "同时报告同分随机抽查的期望。下面各信号统一使用四信号共同可用集；"
              "若 logprob 完全缺失，则改用一致性共同可用集。", "",
              "| 信号 | 预算 / 可用 | 固定种子捕错 | 固定种子捕错率 | 同分随机期望捕错率 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    cohort = metrics["cohorts"]["matched_all_signals"]
    if not cohort["n"]:
        cohort = metrics["cohorts"]["matched_consistency"]
    for signal, budget in cohort["lowest_20_percent"].items():
        lines.append(f"| {DISPLAY[signal]} | {budget['budget']} / {budget['n']} | {budget['captured_errors_fixed_seed']} | "
                     f"{_number(budget['error_capture_rate_fixed_seed'])} | {_number(budget['error_capture_rate_random_tie_expectation'])} |")
    budget = metrics["lowest_20_percent_including_missing_first"]
    lines += ["", f"另按全部步骤分配预算（缺失分数优先），选择 {budget['budget']} / {budget['n']} 步，"
              f"同分随机期望捕错率 {_number(budget['error_capture_rate_random_tie_expectation'])}。", "",
              "## 3. 质量与解释边界", "",
              f"缺少一致性分数 {metrics['quality'].get('missing_consistency', 0)} 步，"
              f"缺少可用 logprob {metrics['quality'].get('missing_logprob', 0)} 步。"
              f"采样响应中无工具 {metrics['quality'].get('n_no_tool', 0)} 次、截断 {metrics['quality'].get('n_truncated', 0)} 次、"
              f"多工具 {metrics['quality'].get('n_multiple_tool_calls', 0)} 次；多工具仍沿用旧实验的首个调用规则。", "",
              "验证任务使用 mock 工具；分数判断的是候选工具调用在当时需求与历史依据下的正确性，"
              "不能把所有工具调用正确等同于整个任务已完成。少量任务和同任务内相关步骤会限制结论，"
              "需要结合错误类型与采集协议理解结果。", "",
              "标签审查由 Codex 模型代理在屏蔽分数的资料上完成，不等于独立人工双标。", "",
              "文件：`validation_steps.jsonl` 保存步骤与调用前上下文及紧凑采样记录，`validation_labels.jsonl` 保存二元标签及依据，"
              "`validation_protocol.json` 保存采集约定，`validation_metrics.json` 保存本报告全部数值、覆盖率和预算明细。", "",
              "公开包以 `collection_summary.json` 保留请求数和 token 汇总，未发布原始请求日志。"
              "`publication.json` 区分原始实验文件哈希与本次公开文件的实际哈希；公开离线拟合参数、OOF 预测和指标与原结果相同。", "",
              "![冻结置信度接口的新任务验证](validation_diagnostic.png)", ""]
    if run.get("model_cost"):
        cost = run["model_cost"]
        lines[-1:-1] = [f"模型采集汇总共 {cost['n_model_requests']} 次请求；有 usage 记录的总 token 为 "
                        f"{cost['total_tokens_with_usage']}。各请求类型的明细保存在指标 JSON。", ""]
    (Path(out) / "新任务验证报告.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/confidence_minimal_v1")
    args = parser.parse_args()
    out = args.out_dir
    required = ["calibrator.json", "validation_steps.jsonl", "validation_labels.jsonl", "validation_protocol.json"]
    for name in required:
        if not (out / name).is_file():
            raise FileNotFoundError("Required validation artifact is missing: " + name)
    calibration = load_calibrator(out / "calibrator.json")
    protocol = json.loads((out / "validation_protocol.json").read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("Validation protocol must be a JSON object")
    rows, quality = join_validation(read_jsonl(out / "validation_steps.jsonl"),
                                    read_jsonl(out / "validation_labels.jsonl"), calibration)
    if protocol.get("calibrator_sha256") is not None and protocol["calibrator_sha256"] != sha(out / "calibrator.json"):
        raise ValueError("Validation protocol calibrator hash does not match the frozen file")
    if isinstance(protocol.get("selected_task_ids"), list):
        allowed = set(protocol["selected_task_ids"])
        if any(row["task_id"] not in allowed for row in rows):
            raise ValueError("Validation includes a task absent from the frozen selected task list")
    metrics = evaluate_rows(rows, quality)
    metrics["collection_design"] = {field: protocol[field] for field in
                                     ("source_file", "selection_seed", "eligible_tasks", "initial_tasks", "maximum_tasks",
                                      "extension_rule", "old_level3_requirement_overlap") if field in protocol}
    metrics["run_diagnostics"] = run_diagnostics(out, metrics, rows)
    metrics["provenance"] = {name + "_sha256": sha(out / name) for name in required}
    metrics["provenance"]["evaluator_sha256"] = sha(__file__)
    for name in ("validation_trajectories.jsonl", "validation_complete.json", "validation_requests.jsonl",
                 "validation_label_review.json", "validation_task_outcomes.jsonl", "validation_labels_original.jsonl",
                 "validation_task_review.jsonl", "collection_summary.json", "publication.json"):
        if (out / name).is_file():
            metrics["provenance"][name + "_sha256"] = sha(out / name)
    metrics["calibrator"] = {field: calibration.get(field) for field in
                             ("version", "coefficient", "intercept", "fit_status", "training_prior", "fit_n")}
    (out / "validation_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    figure(out, rows, metrics)
    report(out, metrics)
    print(json.dumps({"tasks": metrics["n_tasks"], "steps": metrics["n_steps"],
                      "correct": metrics["positive"], "incorrect": metrics["negative"],
                      "signals": {name: {field: values[field] for field in ("n", "AUROC", "Brier", "ECE", "NLL")}
                                  for name, values in metrics["cohorts"]["all_steps"]["signals"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
