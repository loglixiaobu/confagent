#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""研究内容二专用分析：验证触发、纠错收益与等预算成本。"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


def load(paths: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        for line in open(path, encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                rec["_source"] = os.path.basename(path)
                rows.append(rec)
    return rows


def condition_key(t: Dict[str, Any]) -> str:
    cfg = t.get("config") or {}
    return "%s|policy=%s|verifier=%s|B=%s" % (
        t.get("method") or "?", cfg.get("verify_policy", "legacy"),
        cfg.get("verifier", "-"), cfg.get("verify_budget", 0))


def execution_steps(t: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in t.get("steps", []) if s.get("action_type") != "answer"]


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def safe_div(a: float, b: float) -> Any:
    return a / b if b else None


def summarize(trajs: List[Dict[str, Any]]) -> Dict[str, Any]:
    steps = [s for t in trajs for s in execution_steps(t)]
    verified = [s for s in steps if s.get("verified")]
    candidate_bad = [s for s in steps if s.get("candidate_matched_gold") is False]
    verified_bad = [s for s in verified if s.get("candidate_matched_gold") is False]
    outcomes = defaultdict(int)
    for s in steps:
        if s.get("correction_outcome"):
            outcomes[s["correction_outcome"]] += 1
    total_tokens = [float(t.get("total_tokens", 0)) for t in trajs]
    model_calls = [float(t.get("total_model_calls", 0)) for t in trajs]
    verify_calls = [float(t.get("total_verification_calls", 0)) for t in trajs]
    tool_calls = [float(t.get("total_tool_calls", t.get("total_calls", 0))) for t in trajs]
    success = sum(bool(t.get("final_success")) for t in trajs)
    clean = sum(bool(t.get("clean_success")) for t in trajs)
    recovered = sum(bool(t.get("recovered_success")) for t in trajs)
    propagation: List[float] = []
    for t in trajs:
        ts = execution_steps(t)
        first_error = next((i for i, s in enumerate(ts) if s.get("matched_gold") is False), None)
        if first_error is not None:
            propagation.append((len(ts) - first_error - 1) / max(len(ts) - 1, 1))
    corrected = outcomes["error_to_correct"]
    harmed = outcomes["correct_to_error"]
    return {
        "n_tasks": len(trajs),
        "success_rate": safe_div(success, len(trajs)),
        "clean_success_rate": safe_div(clean, len(trajs)),
        "recovered_success_rate": safe_div(recovered, len(trajs)),
        "n_steps": len(steps),
        "verified_steps": len(verified),
        "verification_rate": safe_div(len(verified), len(steps)),
        "trigger_precision": safe_div(len(verified_bad), len(verified)),
        "trigger_recall": safe_div(len(verified_bad), len(candidate_bad)),
        "error_to_correct": corrected,
        "correct_to_error": harmed,
        "correction_precision": safe_div(corrected, corrected + harmed),
        "candidate_error_count": len(candidate_bad),
        "avg_error_propagation": mean(propagation),
        "avg_tokens": mean(total_tokens),
        "avg_model_calls": mean(model_calls),
        "avg_verification_calls": mean(verify_calls),
        "avg_tool_calls": mean(tool_calls),
        "avg_latency_ms": mean([
            float(sum((s.get("latency_ms") or 0) for s in execution_steps(t))) for t in trajs
        ]),
        "outcomes": dict(outcomes),
    }


def _task_success_map(trajs: List[Dict[str, Any]]) -> Dict[Tuple[str, int], int]:
    return {(str(t.get("task_id")), int(t.get("seed") or 0)): int(bool(t.get("final_success")))
            for t in trajs}


def paired_delta_ci(base: List[Dict[str, Any]], other: List[Dict[str, Any]],
                    n_boot: int = 2000, seed: int = 0) -> Dict[str, Any]:
    a, b = _task_success_map(base), _task_success_map(other)
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"n_paired": 0}
    diffs = [b[k] - a[k] for k in keys]
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boots.append(mean(sample))
    boots.sort()
    lo = boots[max(0, int(0.025 * n_boot) - 1)]
    hi = boots[min(n_boot - 1, int(0.975 * n_boot))]
    return {"n_paired": len(keys), "success_delta": mean(diffs),
            "bootstrap_95ci": [lo, hi],
            "wins": sum(d > 0 for d in diffs), "losses": sum(d < 0 for d in diffs),
            "ties": sum(d == 0 for d in diffs)}


def fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "-" if math.isnan(v) else "%.3f" % v
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description="研究内容二：验证纠错与等预算分析")
    ap.add_argument("traj", nargs="+", help="一个或多个 trajectory JSONL")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    trajs = load(args.traj)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in trajs:
        groups[condition_key(t)].append(t)
    if not groups:
        raise SystemExit("没有读到轨迹")

    summaries = {name: summarize(ts) for name, ts in groups.items()}
    print("=" * 134)
    print("研究内容二：自主验证与纠错结果（成本均为每任务平均值）")
    print("=" * 134)
    print("%-42s %5s %7s %7s %7s %7s %7s %7s %7s %8s %8s %8s" % (
        "condition", "N", "success", "recover", "trig-P", "trig-R",
        "fix-P", "harm", "prop", "v-calls", "m-calls", "tokens"))
    print("-" * 134)
    for name, s in summaries.items():
        print("%-42s %5d %7s %7s %7s %7s %7s %7d %7.3f %8.2f %8.2f %8.0f" % (
            name[:42], s["n_tasks"], fmt(s["success_rate"]),
            fmt(s["recovered_success_rate"]), fmt(s["trigger_precision"]),
            fmt(s["trigger_recall"]), fmt(s["correction_precision"]),
            s["correct_to_error"], s["avg_error_propagation"], s["avg_verification_calls"],
            s["avg_model_calls"], s["avg_tokens"]))

    names = list(groups)
    comparisons: Dict[str, Any] = {}
    if len(names) > 1:
        base = names[0]
        print("\n配对任务成功率差（相对第一组 %s）:" % base)
        for name in names[1:]:
            cmp = paired_delta_ci(groups[base], groups[name], args.bootstrap, args.seed)
            comparisons[name] = cmp
            print("  %-42s n=%s delta=%s CI=%s W/L/T=%s/%s/%s" % (
                name[:42], cmp.get("n_paired"), fmt(cmp.get("success_delta", float("nan"))),
                cmp.get("bootstrap_95ci"), cmp.get("wins", 0), cmp.get("losses", 0),
                cmp.get("ties", 0)))

    report = {"conditions": summaries, "paired_against_first": comparisons,
              "sources": args.traj}
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        print("\nJSON 已保存:", args.json_out)


if __name__ == "__main__":
    main()
