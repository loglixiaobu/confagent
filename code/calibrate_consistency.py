#!/usr/bin/env python
"""Fit a fixed, task-grouped 5-fold OOF sigmoid consistency calibrator on CPU.

Only the frozen legacy development labels and fast-mode scores are read.
The final JSON model is for subsequent tasks; all reported performance is OOF.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from evaluate_new_signals import join_scores, metrics, probability, read_jsonl, sha

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261019
N_FOLDS = 5
C_VALUE = 1.0
MODEL_VERSION = "consistency-platt-v1"
SIGNALS = ("raw_consistency", "oof_calibrated", "training_fold_prior")
NAMES = {"raw_consistency": "原始一致性", "oof_calibrated": "任务分组 OOF 校准", "training_fold_prior": "训练折先验常数"}


def relative(path):
    return Path(os.path.relpath(Path(path).resolve(), ROOT)).as_posix()


def sigmoid(value):
    value = float(value)
    if value >= 0:
        return 1.0/(1.0+math.exp(-value))
    exp_value = math.exp(value)
    return exp_value/(1.0+exp_value)


def predict_json(model, raw):
    """Runtime-compatible, pickle-free mapping; missing scores remain missing."""
    if raw is None:
        return None
    score = probability(raw)
    if score is None:
        raise ValueError("raw consistency must be a finite number in [0, 1]")
    if (model.get("version") != MODEL_VERSION or model.get("link") != "sigmoid"
            or model.get("signal") != "consistency"
            or model.get("score_direction") != "higher_is_more_correct"):
        raise ValueError("Unsupported calibrator metadata")
    coefficient, intercept = model.get("coefficient"), model.get("intercept")
    if (type(coefficient) not in (int, float) or not math.isfinite(coefficient) or coefficient < 0
            or type(intercept) not in (int, float) or not math.isfinite(intercept)):
        raise ValueError("Invalid or reversed calibrator coefficients")
    return sigmoid(intercept+coefficient*score)


def fit_sigmoid(x, y):
    """Fixed C=1, unstandardized input. Nonpositive fits fall back to prior."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=int)
    if len(set(y)) != 2:
        raise ValueError("Both training classes are required")
    if not np.isfinite(x).all() or ((x < 0) | (x > 1)).any():
        raise ValueError("Invalid training consistency")
    fit = LogisticRegression(C=C_VALUE, solver="lbfgs", max_iter=1000,
                             tol=1e-10, random_state=SEED).fit(x.reshape(-1, 1), y)
    unconstrained = float(fit.coef_[0, 0])
    prior = float(y.mean())
    coefficient, intercept = unconstrained, float(fit.intercept_[0])
    status = "positive_slope"
    if unconstrained <= 0:
        coefficient = 0.0
        intercept = math.log(prior/(1-prior))
        status = "intercept_only_nonpositive_slope"
    return {"version": MODEL_VERSION, "signal": "consistency", "link": "sigmoid",
            "coefficient": coefficient, "intercept": intercept,
            "unconstrained_coefficient": unconstrained, "fit_status": status,
            "score_direction": "higher_is_more_correct", "input_range": [0, 1],
            "input_standardization": "none", "training_prior": prior,
            "fit_n": len(y), "regularization_C": C_VALUE}


def assign_folds(rows):
    y = np.array([row["y"] for row in rows], dtype=int)
    groups = np.array([row["task_id"] for row in rows])
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds, seen = [], set()
    for fold_id, (train, valid) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        train_tasks, valid_tasks = set(groups[train]), set(groups[valid])
        if train_tasks & valid_tasks:
            raise AssertionError("Task leakage between calibration folds")
        if len(set(y[train])) != 2 or len(set(y[valid])) != 2:
            raise ValueError(f"Fixed fold {fold_id} lacks a class; do not tune seeds against performance")
        if seen & set(valid):
            raise AssertionError("Repeated OOF validation row")
        seen.update(valid)
        folds.append((fold_id, train, valid))
    if seen != set(range(len(rows))):
        raise AssertionError("Incomplete OOF assignment")
    return folds


def score_metrics(p, y):
    result = metrics(p, y)
    yy, qq = np.asarray(y, dtype=int), np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6)
    result["NLL"] = float(np.mean(-yy*np.log(qq)-(1-yy)*np.log(1-qq))) if len(yy) else None
    result["NLL_epsilon"] = 1e-6
    result["ECE_bins"] = 5
    return result


def verification_summary(selected, rows):
    selected = np.asarray(selected, dtype=bool)
    errors = np.array([1-r["y"] for r in rows], dtype=int)
    count, caught, total = int(selected.sum()), int(errors[selected].sum()), int(errors.sum())
    return {"n": len(rows), "verified": count, "verification_fraction": count/len(rows) if rows else 0.,
            "total_errors": total, "errors_captured": caught,
            "error_capture_fraction": caught/total if total else None,
            "error_precision_of_verified": caught/count if count else None,
            "missed_errors": total-caught,
            "correct_calls_sent_to_verification": count-caught}


def budget_selection(scores, fraction=.2, seed=SEED):
    """A fixed integer budget with a reproducible random permutation for ties."""
    values = np.asarray(scores, dtype=float)
    n = len(values)
    budget = int(math.floor(fraction*n))
    permutation = np.random.default_rng(seed).permutation(n)
    tie_rank = np.empty(n, dtype=int)
    tie_rank[permutation] = np.arange(n)
    order = np.lexsort((tie_rank, values))
    selected = np.zeros(n, dtype=bool)
    selected[order[:budget]] = True
    return selected


def oof_calibrate(rows):
    x = np.array([row["consistency"] for row in rows], dtype=float)
    y = np.array([row["y"] for row in rows], dtype=int)
    calibrated, prior = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    fold_ids = np.full(len(rows), -1, dtype=int)
    fold_records = []
    for fold_id, train, valid in assign_folds(rows):
        model = fit_sigmoid(x[train], y[train])
        calibrated[valid] = [predict_json(model, float(value)) for value in x[valid]]
        prior[valid] = model["training_prior"]
        fold_ids[valid] = fold_id
        train_tasks = sorted({rows[i]["task_id"] for i in train})
        valid_tasks = sorted({rows[i]["task_id"] for i in valid})
        fold_records.append({"fold_id": fold_id, "train_n": len(train), "valid_n": len(valid),
                             "train_correct": int(y[train].sum()), "train_errors": int((1-y[train]).sum()),
                             "valid_correct": int(y[valid].sum()), "valid_errors": int((1-y[valid]).sum()),
                             "train_tasks": train_tasks, "valid_tasks": valid_tasks,
                             "fit": model})
    if not np.isfinite(calibrated).all() or not np.isfinite(prior).all() or (fold_ids < 0).any():
        raise AssertionError("Missing OOF predictions")
    predictions = {"raw_consistency": x, "oof_calibrated": calibrated, "training_fold_prior": prior}
    policy = x < 1.0
    budgets = {name: budget_selection(value) for name, value in predictions.items()}
    stats = {name: score_metrics(value, y) for name, value in predictions.items()}
    policies = {"raw_consistency_below_one": {
        "definition": "verify when raw consistency < 1; missing runtime score also requires verification",
        "selected_without_threshold_search": True, **verification_summary(policy, rows)},
        "lowest_20pct_budget": {"fraction": .2, "rounding": "floor", "tie_seed": SEED,
                               "signals": {name: verification_summary(selected, rows) for name, selected in budgets.items()}}}
    records = []
    for i, row in enumerate(rows):
        records.append({"task_id": row["task_id"], "step_id": row["step_id"], "y": row["y"],
                        "fold_id": int(fold_ids[i]), "raw_consistency": float(x[i]),
                        "p_correct_oof": float(calibrated[i]), "p_correct_prior": float(prior[i]),
                        "verify_raw_below_one": bool(policy[i]),
                        **{f"budget_selected_{name}": bool(selected[i]) for name, selected in budgets.items()}})
    return records, fold_records, stats, policies


def make_figure(out, records, stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"raw_consistency": "#d97706", "oof_calibrated": "#2563eb", "training_fold_prior": "#64748b"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = axes[0, 0]
    for signal in SIGNALS:
        bins = stats[signal]["bins"]
        ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-",
                label=signal, color=colors[signal])
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), xlabel="Mean predicted correctness",
           ylabel="Observed correctness", title="Reliability: 5 fixed equal-width bins")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for y, color, label in [(1, "#2563eb", "Correct"), (0, "#dc2626", "Incorrect")]:
        ax.hist([r["p_correct_oof"] for r in records if r["y"] == y], bins=np.linspace(0, 1, 11),
                alpha=.55, color=color, label=label)
    ax.set(xlabel="OOF calibrated P(correct)", ylabel="Calls", title="Only held-out predictions are plotted")
    ax.legend(fontsize=8)
    for ax, metric, title in [(axes[1, 0], "Brier", "Brier score: lower is better"),
                              (axes[1, 1], "NLL", "Clipped NLL: lower is better")]:
        values = [stats[signal][metric] for signal in SIGNALS]
        ax.bar(range(3), values, color=[colors[s] for s in SIGNALS], width=.55)
        for i, value in enumerate(values):
            ax.text(i, value+max(values)*.035, f"{value:.3f}", ha="center")
        ax.set(xticks=range(3), xticklabels=["Raw", "OOF sigmoid", "Fold prior"], title=title,
               ylim=(0, max(values)*1.2))
    fig.suptitle("Consistency calibration: fixed task-grouped 5-fold OOF (development only)")
    fig.savefig(out / "calibration_reliability.png", dpi=170)
    plt.close(fig)


def write_report(out, result, model):
    stats = result["metrics"]
    lines = ["# 最小交付：步骤一致性的概率校准", "",
             f"复用清洗后的 {result['n']} 次旧工具调用（{result['correct']} 正确、{result['errors']} 错误；"
             f"{result['tasks']} 个有调用的任务），只使用 fast 模式多采样一致性一个信号。"
             "本脚本只做 CPU 统计和拟合，没有调用模型、执行工具或读取后续新验证任务。", "",
             "将原始一致性映射为当前调用正确的概率：`p_correct = sigmoid(intercept + coefficient × consistency)`。"
             "原始输入不标准化，LogisticRegression 固定 C=1，不调参。", "",
             "以下全部是固定 5 折 StratifiedGroupKFold 的 OOF 结果，按 task_id 分组、固定种子 20261019。"
             "同一任务全部步骤只在一折中验证；其概率由其他任务训练得到。训练折先验常数也只用该训练折标签。"
             "没有把全量拟合后的训练成绩当泛化成绩。", "",
             "| 方法 | AUROC | Brier ↓ | ECE（5 等宽箱）↓ | NLL ↓ |", "| --- | ---: | ---: | ---: | ---: |"]
    for name in SIGNALS:
        m = stats[name]
        lines.append(f"| {NAMES[name]} | {m['AUROC']:.4f} | {m['Brier']:.4f} | {m['ECE']:.4f} | {m['NLL']:.4f} |")
    lines += ["", "Brier/NLL 衡量概率质量，ECE 依赖这批小样本和分箱。原始 0/1 一致性在 NLL 中按 ε=1e-6 截断，"
              "因此原始分数的极端错误会产生较大 NLL；Brier 和 ECE 不做截断。"
              "单个正向 sigmoid 保持原始排序，校准主要调整概率刻度；OOF 由不同折模型给分，其总体 AUROC 可有变化。", "",
              "| 折 | 训练正确/错误 | 验证正确/错误 | 训练/验证任务数 | 斜率 | 状态 |", "| ---: | ---: | ---: | ---: | ---: | --- |"]
    for fold in result["folds"]:
        lines.append(f"| {fold['fold_id']} | {fold['train_correct']}/{fold['train_errors']} | "
                     f"{fold['valid_correct']}/{fold['valid_errors']} | {len(fold['train_tasks'])}/{len(fold['valid_tasks'])} | "
                     f"{fold['fit']['coefficient']:.4f} | {fold['fit']['fit_status']} |")
    nonpositive = sum(f["fit"]["unconstrained_coefficient"] <= 0 for f in result["folds"])
    lines += ["", f"非正斜率训练折数：{nonpositive}。预先固定规则为：拟合斜率 ≤0 时记录原始斜率，"
              "退回该训练折先验概率（coefficient=0），不翻转信号方向。", "",
              "验证触发规则预先固定为 **raw consistency < 1**；运行时若一致性因传输失败缺失，概率保持 null，要求验证。"
              "没有在错误类型或全体标签上搜索最优阈值。下表仅模拟选择哪些步骤进一步验证，尚未证明验证能纠错。", "",
              "| 选择规则 | 验证步数 | 验证比例 | 捕获错误/全部错误 | 错误捕获率 | 验证中错误比例 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    policy = result["verification"]["raw_consistency_below_one"]
    lines.append(f"| raw consistency < 1 | {policy['verified']} | {policy['verification_fraction']:.1%} | "
                 f"{policy['errors_captured']}/{policy['total_errors']} | {policy['error_capture_fraction']:.1%} | "
                 f"{policy['error_precision_of_verified']:.1%} |")
    for name, stat in result["verification"]["lowest_20pct_budget"]["signals"].items():
        lines.append(f"| 最低分 20%：{NAMES[name]} | {stat['verified']} | {stat['verification_fraction']:.1%} | "
                     f"{stat['errors_captured']}/{stat['total_errors']} | {stat['error_capture_fraction']:.1%} | "
                     f"{stat['error_precision_of_verified']:.1%} |")
    lines += ["", "20% 预算向下取整到整数步数；同分按固定种子随机顺序选取，每次重跑结果相同。"
              "这与此前按同分随机期望计算的分数诊断不同，JSON/CSV 给出实际选中的步骤。", "",
              "最终部署参数由全部旧开发步骤拟合一次并冻结到 calibrator.json："
              f"intercept={model['intercept']:.6f}，coefficient={model['coefficient']:.6f}，状态 {model['fit_status']}。"
              "这份全量模型供后续新任务使用，其训练集预测不列为评估成绩。JSON 无 pickle 依赖，可用标准 sigmoid 读取。", "",
              "一致性来自重建前置状态下的无思考采样策略（K=3，temperature=0.7），旧标签和任务已被审阅，"
              "该信号也曾根据这批任务的比较结果筛选；OOF 不消除信号选择偏差，仍是历史任务的内部开发诊断，并非新的独立测试。旧模拟器错误反馈、任务族相似、"
              "仅四档一致性和重复调用都限制泛化解释。后续必须在新的真实执行步骤上独立检验。", "",
              "![可靠性图与概率诊断](calibration_reliability.png)", "",
              "- calibrator.json：固定输入协议、正向 sigmoid 参数、验证规则和输入哈希。",
              "- oof_predictions.jsonl / .csv：逐步 OOF 概率、训练折先验、折号、固定验证选择。",
              "- calibration_metrics.json：三方法指标、可靠性分箱、每折训练/验证任务和参数、验证覆盖/捕错。", "",
              "在项目根目录复现，只运行 CPU：", "", "```bash",
              f"python code/calibrate_consistency.py --labels {result['paths']['labels']} --scores {result['paths']['scores']} --out-dir {result['paths']['out_dir']}",
              "```"]
    (out / "一致性校准报告.md").write_text("\n".join(lines)+"\n")


def calibrate(labels_path, scores_path, out_dir):
    labels_path, scores_path, out = Path(labels_path), Path(scores_path), Path(out_dir)
    labels, scores = read_jsonl(labels_path), read_jsonl(scores_path)
    summary = json.loads((labels_path.parent / "label_summary.json").read_text())
    if sha(labels_path) != summary["artifact_sha256"][labels_path.name]:
        raise ValueError("Frozen label hash changed")
    protocol_path = scores_path.parent / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    expected = {"consistency_k": 3, "consistency_thinking": False,
                "consistency_temperature": .7, "consistency_top_p": 1.0, "max_tokens": 4096}
    if any(protocol.get(name) != value for name, value in expected.items()):
        raise ValueError("This calibrator requires the fixed K=3 fast collection protocol")
    rows, audit = join_scores(labels, scores, require_history_flags=True)
    if len(rows) != 198 or sum(r["y"] for r in rows) != 158:
        raise ValueError("Expected the frozen 198-call, 158-correct development set")
    if any(row["consistency"] is None or row["consistency_k"] != 3 for row in rows):
        raise ValueError("Incomplete scores: never impute missing consistency for calibration")
    records, folds, stats, policies = oof_calibrate(rows)
    model = fit_sigmoid([row["consistency"] for row in rows], [row["y"] for row in rows])
    model.update({"fit_tasks": len({r["task_id"] for r in rows}), "collection_protocol": expected,
                  "decision_rule": {"signal": "consistency", "operator": "<", "threshold": 1.0,
                                    "missing_score": "verify"},
                  "provenance": {"labels_sha256": sha(labels_path), "scores_sha256": sha(scores_path),
                                 "collection_protocol_sha256": sha(protocol_path), "code_sha256": sha(__file__)},
                  "development_evaluation": {"method": "task-grouped-5fold-OOF", "seed": SEED,
                                             "performance_scope": "internal_development_only"}})
    result = {"version": MODEL_VERSION, "n": len(rows), "correct": sum(r["y"] for r in rows),
              "errors": sum(1-r["y"] for r in rows), "tasks": len({r["task_id"] for r in rows}),
              "input_audit": audit, "metrics": stats, "folds": folds, "verification": policies,
              "configuration": {"folds": N_FOLDS, "splitter": "StratifiedGroupKFold", "seed": SEED,
                                "C": C_VALUE, "input_standardization": "none", "hyperparameter_search": False},
              "paths": {"labels": relative(labels_path), "scores": relative(scores_path), "out_dir": relative(out)},
              "provenance": model["provenance"], "cpu_only": True, "model_requests": 0,
              "new_validation_tasks_read": False, "full_fit_predictions_used_for_metrics": False}
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "calibrator.json"
    if model_path.exists() and json.loads(model_path.read_text()) != model:
        raise ValueError("A different calibrator is already frozen here; use a new output directory")
    model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    (out / "calibration_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    (out / "oof_predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False)+"\n" for r in records))
    with (out / "oof_predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    make_figure(out, records, stats)
    write_report(out, result, model)
    print(json.dumps({"metrics": {name: {k: stats[name][k] for k in ("AUROC", "Brier", "ECE", "NLL")} for name in SIGNALS},
                      "final_coefficient": model["coefficient"], "final_intercept": model["intercept"],
                      "verify_raw_below_one": policies["raw_consistency_below_one"]}, ensure_ascii=False))
    return model, result, records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=str(ROOT / "outputs/legacy_binary_v2/calls_binary.jsonl"))
    parser.add_argument("--scores", default=str(ROOT / "outputs/new_signals_fast_v1/scores.jsonl"))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/confidence_minimal_v1"))
    args = parser.parse_args()
    calibrate(args.labels, args.scores, args.out_dir)
