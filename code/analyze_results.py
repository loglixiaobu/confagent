#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
纯 Python（无第三方依赖）的结果分析 —— 用于在本地复盘已采集的轨迹数据。

计算内容：
  * 任务级：成功率、平均步数、停止原因分布
  * 步骤级：正确率、错误类型分布、错误传播
  * 置信度：正确步 vs 错误步的分布对比、ECE、AUROC、风险—覆盖率
  * 标签口径：宽松 vs 严格的一致性（敏感性分析）
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict

SIGNALS = ["geo_mean_prob", "p10_prob", "min_prob", "tail_geo_mean_prob", "frac_highprob",
           "self_report", "consistency"]


def load(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def ece(confs, labels, n_bins=10):
    n = len(confs)
    total = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        idx = [k for k, c in enumerate(confs)
               if (lo <= c < hi) or (i == n_bins - 1 and c == 1.0)]
        if not idx:
            continue
        acc = sum(labels[k] for k in idx) / len(idx)
        conf = sum(confs[k] for k in idx) / len(idx)
        total += (len(idx) / n) * abs(acc - conf)
    return total


def brier(confs, labels):
    return sum((c - y) ** 2 for c, y in zip(confs, labels)) / len(confs)


def nll(confs, labels, eps=1e-6):
    s = 0.0
    for c, y in zip(confs, labels):
        p = min(max(c, eps), 1 - eps)
        s += y * math.log(p) + (1 - y) * math.log(1 - p)
    return -s / len(confs)


def auroc(scores, labels):
    """秩和法，处理并列。scores 越大越可能 label=1。"""
    pairs = sorted(zip(scores, labels), key=lambda t: t[0])
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    pos = sum(1 for _, y in pairs if y == 1)
    neg = n - pos
    if pos == 0 or neg == 0:
        return float("nan")
    s_pos = sum(ranks[k] for k in range(n) if pairs[k][1] == 1)
    return (s_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def risk_coverage(confs, labels, n_points=20):
    order = sorted(range(len(confs)), key=lambda k: -confs[k])
    errs = [1 - labels[k] for k in order]
    out = []
    cum = 0
    for i, e in enumerate(errs, start=1):
        cum += e
        if i % max(1, len(errs) // n_points) == 0 or i == len(errs):
            out.append((i / len(errs), cum / i))
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/traj_main_clean.jsonl"
    trajs = load(path)

    print("=" * 78)
    print("结果分析:", os.path.basename(path))
    print("=" * 78)

    # ---- 任务级 ----
    n_task = len(trajs)
    n_ok = sum(1 for t in trajs if t.get("final_success"))
    steps_per = [t["n_steps"] for t in trajs]
    stop = Counter((t.get("config") or {}).get("stop_reason", "?") for t in trajs)
    print("任务数            : %d" % n_task)
    print("任务成功率        : %.1f%%  (%d/%d)" % (100 * n_ok / n_task, n_ok, n_task))
    print("平均步数/任务     : %.2f  (范围 %d–%d)"
          % (sum(steps_per) / n_task, min(steps_per), max(steps_per)))
    print("停止原因分布      : %s" % dict(stop))
    print("总 token          : %d" % sum(t.get("total_tokens", 0) for t in trajs))
    print("总工具调用        : %d" % sum(t.get("total_calls", 0) for t in trajs))

    # ---- 步骤级 ----
    steps = [s for t in trajs for s in t["steps"]]
    labeled = [s for s in steps if s.get("matched_gold") is not None]
    n_step = len(labeled)
    n_ok_step = sum(1 for s in labeled if s["matched_gold"])
    print()
    print("步骤总数          : %d" % n_step)
    print("步骤正确率        : %.1f%%  (%d 正 / %d 负)"
          % (100 * n_ok_step / n_step, n_ok_step, n_step - n_ok_step))

    ft = Counter(s.get("failure_type") for s in labeled if s.get("failure_type"))
    print("错误类型分布      : %s" % dict(ft))

    api_stat = defaultdict(lambda: [0, 0])
    for s in labeled:
        name = (s.get("action_input") or {}).get("api_name", "?")
        api_stat[name][1] += 1
        if s["matched_gold"]:
            api_stat[name][0] += 1
    print("按 API 的正确率   :")
    for name, (ok, tot) in sorted(api_stat.items(), key=lambda kv: -kv[1][1])[:8]:
        print("    %-16s %3d/%3d  (%.0f%%)" % (name, ok, tot, 100 * ok / tot))

    # ---- 标签口径敏感性 ----
    both = [s for s in labeled if s.get("strict_matched_gold") is not None]
    if both:
        agree = sum(1 for s in both if s["matched_gold"] == s["strict_matched_gold"]) / len(both)
        n_strict_ok = sum(1 for s in both if s["strict_matched_gold"])
        print()
        print("标签口径一致性    : %.1f%%  (宽松 vs 严格，%d 条)"
              % (100 * agree, len(both)))
        print("严格口径正确率    : %.1f%%" % (100 * n_strict_ok / len(both)))

    # ---- 错误传播 ----
    prop = []
    for t in trajs:
        first_err = next((i for i, s in enumerate(t["steps"])
                          if s.get("matched_gold") is False), None)
        if first_err is not None and len(t["steps"]) > 1:
            prop.append((len(t["steps"]) - first_err - 1) / (len(t["steps"]) - 1))
    if prop:
        print("错误传播率(均值)  : %.2f  （首个错误步骤之后仍继续执行的比例，%d 条失败轨迹）"
              % (sum(prop) / len(prop), len(prop)))

    # ---- 置信度 ----
    print()
    print("=" * 78)
    print("步骤级置信度分析（信号 = 推理段 token logprob）")
    print("=" * 78)

    for sig in SIGNALS:
        vals = [(s["raw_logprobs"].get(sig), int(bool(s["matched_gold"])))
                for s in labeled
                if s.get("raw_logprobs") and s["raw_logprobs"].get(sig) is not None]
        if len(vals) < 5:
            continue
        confs = [v[0] for v in vals]
        labels = [v[1] for v in vals]
        acc = sum(labels) / len(labels)
        mc = sum(confs) / len(confs)
        c_ok = [c for c, y in vals if y == 1]
        c_bad = [c for c, y in vals if y == 0]
        print()
        print("--- %s ---" % sig)
        print("  样本 %d | 实际正确率 %.3f | 平均置信度 %.3f  -> 过度自信 %+.3f"
              % (len(vals), acc, mc, mc - acc))
        print("  正确步置信度 均值 %.3f  (n=%d)" % (sum(c_ok) / len(c_ok), len(c_ok)))
        if c_bad:
            print("  错误步置信度 均值 %.3f  (n=%d)" % (sum(c_bad) / len(c_bad), len(c_bad)))
        print("  ECE %.4f | Brier %.4f | NLL %.4f | AUROC %.4f"
              % (ece(confs, labels), brier(confs, labels), nll(confs, labels),
                 auroc(confs, labels)))

    # ---- 风险—覆盖率 ----
    key = "p10_prob"
    vals = [(s["raw_logprobs"].get(key), int(bool(s["matched_gold"])))
            for s in labeled
            if s.get("raw_logprobs") and s["raw_logprobs"].get(key) is not None]
    if vals:
        confs = [v[0] for v in vals]
        labels = [v[1] for v in vals]
        print()
        print("风险—覆盖率曲线（按置信度从高到低接受步骤，信号=%s）" % key)
        print("   覆盖率   剩余错误率")
        for cov, risk in risk_coverage(confs, labels):
            print("   %5.2f    %6.3f" % (cov, risk))


if __name__ == "__main__":
    main()
