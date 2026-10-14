#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
步骤级置信度的校准评估（研究内容一的核心分析）。

回答的问题
--------------------------------------------------------------------
  Q1  从推理段提取的哪一类信号，最能反映"这一步是否可靠"？
      （消融 A1：置信度来源比较）
  Q2  校准之后，置信度是否真的"名副其实"？（ECE / Brier / NLL / 可靠性图）
  Q3  置信度能否有效定位错误步骤？（错误检测 AUROC、风险—覆盖率曲线）
  Q4  哪种校准方法最有效？（消融 A2：无校准 / 温度缩放 / Platt / Isotonic）

关键约定
--------------------------------------------------------------------
  * 标签 y_t 使用 **宽松口径** matched_gold（见 apibank_env.py 的说明）；
    严格口径 strict_matched_gold 同时报告，用于**敏感性分析**。
  * 退化步骤（推理段过短、置信度无意义）默认剔除，并报告剔除比例。
  * 校准集与测试集按**任务**切分（不是按步骤切分），避免同任务步骤泄漏。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

# 可比较的候选置信度信号（值越大表示越"自信"）
SIGNAL_KEYS = [
    "geo_mean_prob",
    "p10_prob",
    "min_prob",
    "tail_geo_mean_prob",
    "frac_highprob",
]

# 熵是"越大越不确定"，比较时取负
NEG_SIGNAL_KEYS = ["mean_entropy_topk"]


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_steps(path: str, min_think_tokens: int = 5,
               drop_degenerate: bool = True) -> List[Dict[str, Any]]:
    """从 trajectory.jsonl 展开出步骤级样本。"""
    rows: List[Dict[str, Any]] = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        traj = json.loads(line)
        for s in traj.get("steps", []):
            rl = s.get("raw_logprobs") or {}
            if s.get("matched_gold") is None:
                continue
            if drop_degenerate and rl.get("degenerate"):
                continue
            if rl.get("n_tokens") is not None and rl["n_tokens"] < min_think_tokens:
                continue
            row = {
                "task_id": traj.get("task_id"),
                "step_id": s.get("step_id"),
                "api_name": (s.get("action_input") or {}).get("api_name"),
                "y": int(bool(s.get("matched_gold"))),
                "y_strict": (None if s.get("strict_matched_gold") is None
                             else int(bool(s.get("strict_matched_gold")))),
                "failure_type": s.get("failure_type"),
                "n_tokens": rl.get("n_tokens"),
                "degenerate": bool(rl.get("degenerate", False)),
                "confidence": s.get("confidence"),
                "self_report": rl.get("self_report"),
                "consistency": rl.get("consistency"),
            }
            for k in SIGNAL_KEYS:
                row[k] = rl.get(k)
            for k in NEG_SIGNAL_KEYS:
                v = rl.get(k)
                row[k] = (-v if v is not None else None)
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 校准指标
# ---------------------------------------------------------------------------
def _bin_edges(n_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def ece(confs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """期望校准误差：|准确率 − 平均置信度| 的加权和。"""
    edges = _bin_edges(n_bins)
    n = len(confs)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (confs >= lo) & (confs < hi if i < n_bins - 1 else confs <= hi)
        if m.sum() == 0:
            continue
        acc = labels[m].mean()
        conf = confs[m].mean()
        total += (m.sum() / n) * abs(acc - conf)
    return float(total)


def mce(confs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """最大校准误差。"""
    edges = _bin_edges(n_bins)
    worst = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (confs >= lo) & (confs < hi if i < n_bins - 1 else confs <= hi)
        if m.sum() == 0:
            continue
        worst = max(worst, abs(labels[m].mean() - confs[m].mean()))
    return float(worst)


def brier(confs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((confs - labels) ** 2))


def nll(confs: np.ndarray, labels: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(confs, eps, 1 - eps)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """错误检测 AUROC：scores 越大表示越可能"正确"。用秩和法计算，无需 sklearn。"""
    labels = labels.astype(int)
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # 处理并列
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def risk_coverage(confs: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """按置信度从高到低逐步"接受"步骤，返回 (覆盖率, 错误率) 曲线。

    这是本项目的核心决策曲线：验证预算有限时，优先验证低置信步骤，
    因此需要知道"接受比例 vs 剩余错误率"的取舍。
    """
    order = np.argsort(-confs)
    errs = 1 - labels[order]
    cum_err = np.cumsum(errs) / np.arange(1, len(errs) + 1)
    cov = np.arange(1, len(errs) + 1) / len(errs)
    return cov, cum_err


def aurc(confs: np.ndarray, labels: np.ndarray) -> float:
    """风险—覆盖率曲线下面积（越小越好）。"""
    cov, risk = risk_coverage(confs, labels)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(risk, cov))


# ---------------------------------------------------------------------------
# 校准方法
# ---------------------------------------------------------------------------
def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_temperature(confs: np.ndarray, labels: np.ndarray) -> float:
    """拟合温度缩放参数 T：p' = sigmoid(logit(p)/T)，最小化 NLL。"""
    z = _logit(confs)
    best_T, best_nll = 1.0, float("inf")
    for T in np.linspace(0.05, 10.0, 200):
        p = 1.0 / (1.0 + np.exp(-z / T))
        v = nll(p, labels)
        if v < best_nll:
            best_nll, best_T = v, float(T)
    return best_T


def apply_temperature(confs: np.ndarray, T: float) -> np.ndarray:
    z = _logit(confs)
    return 1.0 / (1.0 + np.exp(-z / T))


class PlattCalibrator:
    def __init__(self) -> None:
        self.model = None

    def fit(self, confs: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        if HAVE_SKLEARN:
            self.model = LogisticRegression().fit(_logit(confs).reshape(-1, 1), labels)
        return self

    def predict(self, confs: np.ndarray) -> np.ndarray:
        if self.model is None:
            return confs
        return self.model.predict_proba(_logit(confs).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    def __init__(self) -> None:
        self.model = None

    def fit(self, confs: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        if HAVE_SKLEARN:
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.model.fit(confs, labels)
        return self

    def predict(self, confs: np.ndarray) -> np.ndarray:
        if self.model is None:
            return confs
        return self.model.predict(confs)


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def summarize(confs: np.ndarray, labels: np.ndarray, name: str) -> Dict[str, float]:
    return {
        "signal": name,
        "n": int(len(confs)),
        "acc": float(labels.mean()) if len(labels) else float("nan"),
        "mean_conf": float(confs.mean()) if len(confs) else float("nan"),
        "ECE": ece(confs, labels),
        "MCE": mce(confs, labels),
        "Brier": brier(confs, labels),
        "NLL": nll(confs, labels),
        "AUROC": auroc(confs, labels),
        "AURC": aurc(confs, labels),
    }


def print_table(rows: List[Dict[str, Any]], title: str) -> None:
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)
    hdr = "%-26s %6s %8s %10s %8s %8s %8s %8s %8s" % (
        "signal", "n", "acc", "mean_conf", "ECE", "MCE", "Brier", "NLL", "AUROC")
    print(hdr)
    print("-" * 92)
    for r in rows:
        print("%-26s %6d %8.3f %10.3f %8.4f %8.4f %8.4f %8.4f %8.4f" % (
            r["signal"], r["n"], r["acc"], r["mean_conf"],
            r["ECE"], r["MCE"], r["Brier"], r["NLL"], r["AUROC"]))


def plot_reliability(results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                     out_path: str, n_bins: int = 10) -> None:
    """画可靠性图（校准曲线）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="perfect calibration")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, (name, (c, y)) in enumerate(results.items()):
        edges = np.linspace(0, 1, n_bins + 1)
        xs, ys = [], []
        for k in range(n_bins):
            m = (c >= edges[k]) & (c < edges[k + 1] if k < n_bins - 1 else c <= edges[k + 1])
            if m.sum() == 0:
                continue
            xs.append(c[m].mean())
            ys.append(y[m].mean())
        ax.plot(xs, ys, "o-", color=colors[i % len(colors)], label=name, markersize=4)
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i, (name, (c, y)) in enumerate(results.items()):
        cov, risk = risk_coverage(c, y)
        ax.plot(cov, risk, color=colors[i % len(colors)], label=name)
    ax.set_xlabel("coverage (fraction of steps accepted)")
    ax.set_ylabel("error rate among accepted")
    ax.set_title("Risk-coverage curve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print("\n图已保存:", out_path)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="步骤级置信度校准评估")
    ap.add_argument("--traj", default="/home1/lh/confagent/outputs/traj_pilot.jsonl")
    ap.add_argument("--out-dir", default="/home1/lh/confagent/outputs")
    ap.add_argument("--label", default="y", choices=["y", "y_strict"])
    ap.add_argument("--calib-ratio", type=float, default=0.5,
                    help="按任务切分，前多少比例的任务作为校准集")
    ap.add_argument("--min-think-tokens", type=int, default=5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rows = load_steps(args.traj, min_think_tokens=args.min_think_tokens)
    if not rows:
        print("[ERR] 没有可用样本。检查轨迹文件:", args.traj)
        return

    total = sum(1 for _ in open(args.traj, encoding="utf-8") if _.strip())
    print("=" * 92)
    print("步骤级置信度校准报告")
    print("=" * 92)
    print("轨迹文件      :", args.traj)
    print("轨迹条数      :", total)
    print("可用步骤样本  :", len(rows))
    print("标签口径      :", "matched_gold（宽松）" if args.label == "y"
          else "strict_matched_gold（严格）")
    y_all = np.array([r[args.label] for r in rows], dtype=int)
    print("步骤正确率    : %.3f  (正 %d / 负 %d)" % (y_all.mean(), y_all.sum(), len(y_all) - y_all.sum()))

    # ---- 标签口径敏感性分析 ----
    both = [r for r in rows if r["y_strict"] is not None]
    if both:
        agree = sum(1 for r in both if r["y"] == r["y_strict"]) / len(both)
        print("标签口径一致性: %.3f  (宽松 vs 严格，%d 条可比)" % (agree, len(both)))

    # ---- 错误类型分布 ----
    ft: Dict[str, int] = {}
    for r in rows:
        if r["failure_type"]:
            ft[r["failure_type"]] = ft.get(r["failure_type"], 0) + 1
    if ft:
        print("错误类型分布  :", json.dumps(ft, ensure_ascii=False))

    # ---- 按任务切分校准集 / 测试集 ----
    task_ids = sorted({r["task_id"] for r in rows})
    n_cal = max(1, int(len(task_ids) * args.calib_ratio))
    cal_tasks = set(task_ids[:n_cal])
    cal = [r for r in rows if r["task_id"] in cal_tasks]
    tst = [r for r in rows if r["task_id"] not in cal_tasks]
    if not tst:
        cal, tst = rows, rows
    print("校准集 / 测试集: %d / %d 步骤（按任务切分，避免同任务泄漏）" % (len(cal), len(tst)))

    y_cal = np.array([r[args.label] for r in cal], dtype=int)
    y_tst = np.array([r[args.label] for r in tst], dtype=int)

    # ---- A1：哪种信号最可校准 ----
    base_rows = []
    plot_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for key in SIGNAL_KEYS + NEG_SIGNAL_KEYS + ["self_report", "consistency"]:
        c = np.array([r[key] for r in tst if r.get(key) is not None], dtype=float)
        yy = np.array([r[args.label] for r in tst if r.get(key) is not None], dtype=int)
        if len(c) < 10 or len(set(yy)) < 2:
            continue
        base_rows.append(summarize(c, yy, key))
        plot_data[key] = (c, yy)

    if base_rows:
        print_table(base_rows, "A1 候选置信度信号比较（未校准，测试集）")
        best = max(base_rows, key=lambda r: (r["AUROC"] if not math.isnan(r["AUROC"]) else -1))
        print("\n-> AUROC 最高的是: %s (%.3f)" % (best["signal"], best["AUROC"]))

        # ---- A2：校准方法比较（用校准集拟合，测试集评估） ----
        key = best["signal"]
        c_cal = np.array([r[key] for r in cal if r.get(key) is not None], dtype=float)
        yc = np.array([r[args.label] for r in cal if r.get(key) is not None], dtype=int)
        c_tst = np.array([r[key] for r in tst if r.get(key) is not None], dtype=float)
        yt = np.array([r[args.label] for r in tst if r.get(key) is not None], dtype=int)

        methods: List[Dict[str, Any]] = [summarize(c_tst, yt, "none")]

        T = fit_temperature(c_cal, yc)
        c_t = apply_temperature(c_tst, T)
        methods.append(summarize(c_t, yt, "temperature(T=%.2f)" % T))

        # 校准集只有一个类别时，Platt / Isotonic 无法拟合（小样本常见），跳过并说明
        if len(set(yc.tolist())) < 2:
            print("\n[注意] 校准集只含单一类别（%d 条，全为 %s），"
                  "Platt / Isotonic 无法拟合，已跳过。"
                  % (len(yc), "正确" if yc[0] == 1 else "错误"))
            print("       样本量充足时不会出现该问题；当前结果仅作流程验证。")
            plot_data = {"none": (c_tst, yt), "temperature": (c_t, yt)}
        else:
            pl = PlattCalibrator().fit(c_cal, yc)
            methods.append(summarize(pl.predict(c_tst), yt, "platt"))

            iso = IsotonicCalibrator().fit(c_cal, yc)
            methods.append(summarize(iso.predict(c_tst), yt, "isotonic"))

            plot_data = {
                "none": (c_tst, yt),
                "temperature": (c_t, yt),
                "isotonic": (iso.predict(c_tst), yt),
            }

        print_table(methods, "A2 校准方法比较（信号=%s，校准集拟合 / 测试集评估）" % key)

        os.makedirs(args.out_dir, exist_ok=True)
        suffix = ("_" + args.tag) if args.tag else ""
        png = os.path.join(args.out_dir, "reliability%s.png" % suffix)
        plot_reliability(plot_data, png)

        # 保存指标
        js = os.path.join(args.out_dir, "calibration_metrics%s.json" % suffix)
        with open(js, "w", encoding="utf-8") as f:
            json.dump({"traj": args.traj, "label": args.label,
                       "n_steps": len(rows), "step_acc": float(y_all.mean()),
                       "signal_comparison": base_rows, "method_comparison": methods},
                      f, ensure_ascii=False, indent=2)
        print("指标已保存:", js)


if __name__ == "__main__":
    main()
