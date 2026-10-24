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
    return "%s|policy=%s|verifier=%s|B=%s|K=%s|tau=%s|sr=%s|checks=%s|rate=%s|proto=%s" % (
        t.get("method") or "?", cfg.get("verify_policy", "legacy"),
        cfg.get("verifier", "-"), cfg.get("verify_budget", 0),
        cfg.get("consistency_k", 0), cfg.get("risk_threshold", "-"),
        bool(cfg.get("self_report", False)), cfg.get("max_checks_per_task", "-"),
        cfg.get("random_check_rate", "-"), cfg.get("q2_protocol_version", "legacy"))


def execution_steps(t: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in t.get("steps", []) if s.get("action_type") != "answer"]


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def safe_div(a: float, b: float) -> Any:
    return a / b if b else None


def _auroc(labels: List[int], scores: List[float]) -> Any:
    """AUROC by pairwise ordering; score is probability of correctness."""
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = sum((positive > negative) + 0.5 * (positive == negative)
               for positive in positives for negative in negatives)
    return wins / (len(positives) * len(negatives))


def _ece(labels: List[int], scores: List[float], bins: int = 10) -> Any:
    if not labels:
        return None
    total = len(labels)
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [(label, score) for label, score in zip(labels, scores)
                   if low <= score < high or (index == bins - 1 and score == 1.0)]
        if members:
            accuracy = mean([float(label) for label, _ in members])
            confidence = mean([score for _, score in members])
            error += len(members) / total * abs(accuracy - confidence)
    return error


def _calibration_bins(labels: List[int], scores: List[float], bins: int = 10) -> List[Dict[str, Any]]:
    rows = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [(label, score) for label, score in zip(labels, scores)
                   if low <= score < high or (index == bins - 1 and score == 1.0)]
        if members:
            rows.append({
                "lower": low, "upper": high, "count": len(members),
                "mean_confidence": mean([score for _, score in members]),
                "empirical_accuracy": mean([float(label) for label, _ in members]),
            })
    return rows


def summarize(trajs: List[Dict[str, Any]]) -> Dict[str, Any]:
    steps = [s for t in trajs for s in execution_steps(t)]
    verified = [s for s in steps if s.get("selected_for_check", s.get("verified", False))]
    available_verified = [
        s for s in verified
        if isinstance(s.get("verification_result"), dict)
        and s["verification_result"].get("available") is True
    ]
    candidate_bad = [s for s in steps if s.get("candidate_matched_gold") is False]
    verified_bad = [s for s in verified if s.get("candidate_matched_gold") is False]
    grounded_labeled = [s for s in steps if s.get("candidate_grounded_correct") is not None]
    grounded_bad = [s for s in grounded_labeled if s.get("candidate_grounded_correct") is False]
    verified_grounded = [s for s in verified if s.get("candidate_grounded_correct") is not None]
    verified_grounded_bad = [s for s in verified_grounded
                             if s.get("candidate_grounded_correct") is False]
    outcomes = defaultdict(int)
    for s in steps:
        if s.get("correction_outcome"):
            outcomes[s["correction_outcome"]] += 1
    total_tokens = [float(t.get("total_tokens", 0)) for t in trajs]
    model_calls = [float(t.get("total_model_calls", 0)) for t in trajs]
    actor_calls = [float(t.get("total_actor_calls",
                               t.get("total_model_calls", 0) - t.get("total_verification_calls", 0)))
                   for t in trajs]
    verify_calls = [float(t.get("total_verification_calls", 0)) for t in trajs]
    signal_calls = [float(sum((s.get("signal_model_calls") or 0)
                              for s in execution_steps(t))) for t in trajs]
    verifier_calls = [float(sum((s.get("verifier_model_calls") or 0)
                                for s in execution_steps(t))) for t in trajs]
    tool_calls = [float(t.get("total_tool_calls", t.get("total_calls", 0))) for t in trajs]
    success = sum(bool(t.get("final_success")) for t in trajs)
    clean = sum(bool(t.get("clean_success")) for t in trajs)
    recovered = sum(bool(t.get("recovered_success")) for t in trajs)
    replay = sum(bool(t.get("replay_success", t.get("final_success"))) for t in trajs)
    replay_propagation: List[float] = []
    grounded_propagation: List[float] = []
    for t in trajs:
        ts = execution_steps(t)
        replay_first = next((i for i, s in enumerate(ts)
                             if s.get("matched_gold") is False), None)
        grounded_first = next((i for i, s in enumerate(ts)
                               if s.get("executed_grounded_correct") is False), None)
        if replay_first is not None:
            replay_propagation.append(
                (len(ts) - replay_first - 1) / max(len(ts) - 1, 1))
        if grounded_first is not None:
            grounded_propagation.append(
                (len(ts) - grounded_first - 1) / max(len(ts) - 1, 1))
    corrected = outcomes["error_to_correct"]
    harmed = outcomes["correct_to_error"]
    grounded_corrected = sum(
        s.get("candidate_grounded_correct") is False
        and s.get("executed_grounded_correct") is True for s in steps)
    grounded_harmed = sum(
        s.get("candidate_grounded_correct") is True
        and s.get("executed_grounded_correct") is False for s in steps)
    retry_recovered = sum(s.get("retry_outcome") == "recovered" for s in steps)
    confidence_rows = [(int(bool(s.get("candidate_matched_gold"))), float(s["confidence"]))
                       for s in steps
                       if s.get("candidate_matched_gold") is not None
                       and isinstance(s.get("confidence"), (int, float))]
    confidence_labels = [label for label, _ in confidence_rows]
    confidence_scores = [score for _, score in confidence_rows]
    brier = (mean([(score - label) ** 2
                   for label, score in confidence_rows]) if confidence_rows else None)
    correct_scores = [score for label, score in confidence_rows if label == 1]
    wrong_scores = [score for label, score in confidence_rows if label == 0]
    low_steps = [s for s in steps
                 if isinstance(s.get("confidence_result"), dict)
                 and s["confidence_result"].get("needs_verification") is True
                 and s.get("candidate_matched_gold") is not None]
    high_steps = [s for s in steps
                  if isinstance(s.get("confidence_result"), dict)
                  and s["confidence_result"].get("needs_verification") is False
                  and s.get("candidate_matched_gold") is not None]
    return {
        "n_tasks": len(trajs),
        "success_rate": safe_div(success, len(trajs)),
        "clean_success_rate": safe_div(clean, len(trajs)),
        "recovered_success_rate": safe_div(recovered, len(trajs)),
        "replay_success_rate": safe_div(replay, len(trajs)),
        "n_steps": len(steps),
        "verified_steps": len(verified),
        "verification_rate": safe_div(len(verified), len(steps)),
        "available_verifications": len(available_verified),
        "verification_available_rate": safe_div(len(available_verified), len(verified)),
        "usable_verification_rate": safe_div(len(available_verified), len(steps)),
        # The primary causal label is the environment replay label. Grounding
        # is reported separately because it is also an input to the safety gate.
        "trigger_label_basis": "replay",
        "trigger_precision": safe_div(len(verified_bad), len(verified)),
        "trigger_recall": safe_div(len(verified_bad), len(candidate_bad)),
        "replay_trigger_precision": safe_div(len(verified_bad), len(verified)),
        "replay_trigger_recall": safe_div(len(verified_bad), len(candidate_bad)),
        "grounded_label_coverage": safe_div(len(grounded_labeled), len(steps)),
        "grounded_trigger_precision": safe_div(len(verified_grounded_bad), len(verified_grounded)),
        "grounded_trigger_recall": safe_div(len(verified_grounded_bad), len(grounded_bad)),
        "error_to_correct": corrected,
        "correct_to_error": harmed,
        "grounded_error_to_correct": grounded_corrected,
        "grounded_correct_to_error": grounded_harmed,
        "retry_recovered": retry_recovered,
        "correction_precision": safe_div(corrected, corrected + harmed),
        "replay_correction_precision": safe_div(corrected, corrected + harmed),
        "grounded_correction_precision": safe_div(
            grounded_corrected, grounded_corrected + grounded_harmed),
        "candidate_error_count": len(candidate_bad),
        "replay_candidate_error_count": len(candidate_bad),
        "grounded_candidate_error_count": len(grounded_bad),
        "avg_error_propagation": mean(replay_propagation),
        "grounded_avg_error_propagation": mean(grounded_propagation),
        "confidence_label_coverage": safe_div(len(confidence_rows), len(steps)),
        "confidence_auroc": _auroc(confidence_labels, confidence_scores),
        "confidence_brier": brier,
        "confidence_ece_10": _ece(confidence_labels, confidence_scores),
        "confidence_bins_10": _calibration_bins(confidence_labels, confidence_scores),
        "mean_confidence_correct": mean(correct_scores) if correct_scores else None,
        "mean_confidence_wrong": mean(wrong_scores) if wrong_scores else None,
        "low_confidence_error_rate": safe_div(
            sum(s.get("candidate_matched_gold") is False for s in low_steps), len(low_steps)),
        "high_confidence_error_rate": safe_div(
            sum(s.get("candidate_matched_gold") is False for s in high_steps), len(high_steps)),
        "avg_tokens": mean(total_tokens),
        "avg_model_calls": mean(model_calls),
        "avg_actor_calls": mean(actor_calls),
        "avg_verification_calls": mean(verify_calls),
        "avg_signal_calls": mean(signal_calls),
        "avg_verifier_calls": mean(verifier_calls),
        "fixes_per_100_extra_calls": safe_div(100.0 * corrected,
                                                sum(verify_calls)),
        "avg_tool_calls": mean(tool_calls),
        "avg_latency_ms": mean([
            float(sum((s.get("latency_ms") or 0) for s in execution_steps(t))) for t in trajs
        ]),
        "outcomes": dict(outcomes),
    }


def _task_success_map(trajs: List[Dict[str, Any]]) -> Dict[Tuple[str, int], int]:
    result: Dict[Tuple[str, int], int] = {}
    for trajectory in trajs:
        key = (str(trajectory.get("task_id")), int(trajectory.get("seed") or 0))
        if key in result:
            raise ValueError("重复的 task_id/seed: %r" % (key,))
        result[key] = int(bool(trajectory.get("final_success")))
    return result


def paired_delta_ci(base: List[Dict[str, Any]], other: List[Dict[str, Any]],
                    n_boot: int = 2000, seed: int = 0) -> Dict[str, Any]:
    a, b = _task_success_map(base), _task_success_map(other)
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"n_paired": 0, "n_base": len(a), "n_other": len(b),
                "complete_pairing": False}
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
            "n_base": len(a), "n_other": len(b),
            "complete_pairing": len(keys) == len(a) == len(b),
            "bootstrap_95ci": [lo, hi],
            "wins": sum(d > 0 for d in diffs), "losses": sum(d < 0 for d in diffs),
            "ties": sum(d == 0 for d in diffs)}


def verified_corrections(trajs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Export only successful, safety-accepted repairs for downstream Q3 review."""
    rows = []
    for trajectory in trajs:
        config = trajectory.get("config") or {}
        for step in execution_steps(trajectory):
            guard = step.get("revision_guard") or {}
            if (step.get("correction_outcome") != "error_to_correct"
                    or not step.get("correction_changed")
                    or guard.get("accepted") is not True):
                continue
            verification = step.get("verification_result") or {}
            grounding = step.get("candidate_grounding") or {}
            rows.append({
                "schema_version": "q2-verified-correction-v1",
                "task_id": trajectory.get("task_id"),
                "step_id": step.get("step_id"),
                "seed": trajectory.get("seed"),
                "manifest_id": config.get("task_manifest"),
                "candidate_action": step.get("candidate_action"),
                "corrected_action": step.get("action_input"),
                "error_types": sorted({issue.get("kind") for issue in grounding.get("issues", [])
                                       if issue.get("kind")}),
                "evidence": {
                    "grounding": grounding,
                    "revision_guard": guard,
                    "critic_issues": verification.get("issues") or [],
                },
                "outcome": "error_to_correct",
            })
    return rows


def fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "-" if math.isnan(v) else "%.3f" % v
    return str(v)


def _short_condition(name: str) -> str:
    """Compact the long protocol key for plot tick labels."""
    parts = name.split("|")
    method = parts[0]
    values = {item.split("=", 1)[0]: item.split("=", 1)[1]
              for item in parts[1:] if "=" in item}
    policy = values.get("policy", method)
    if policy == "vanilla":
        return "vanilla"
    return "%s\n%s, B=%s" % (policy, values.get("verifier", "-"), values.get("B", "0"))


def plot_summaries(summaries: Dict[str, Dict[str, Any]], path: str) -> None:
    """Create a Q2-only outcome, verification coverage, and cost dashboard."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("--plot-out 需要 matplotlib；请安装 requirements-analysis.txt") from exc

    names = list(summaries)
    labels = [_short_condition(name) for name in names]
    x = list(range(len(names)))
    width = 0.36
    fig_width = max(12.0, 2.0 * len(names) + 7.0)
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, 4.6))

    success = [100.0 * (summaries[n]["success_rate"] or 0.0) for n in names]
    replay = [100.0 * (summaries[n]["replay_success_rate"] or 0.0) for n in names]
    axes[0].bar([i - width / 2 for i in x], success, width,
                label="End-to-end success", color="#4C78A8")
    axes[0].bar([i + width / 2 for i in x], replay, width,
                label="Replay complete", color="#72B7B2")
    axes[0].set_title("Task outcome")
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_ylim(0, 105)
    axes[0].legend(fontsize=8)

    triggered = [100.0 * (summaries[n]["verification_rate"] or 0.0) for n in names]
    usable = [100.0 * (summaries[n]["usable_verification_rate"] or 0.0) for n in names]
    axes[1].bar([i - width / 2 for i in x], triggered, width,
                label="Triggered", color="#F58518")
    axes[1].bar([i + width / 2 for i in x], usable, width,
                label="Usable output", color="#54A24B")
    axes[1].set_title("Verification coverage")
    axes[1].set_ylabel("Execution steps (%)")
    axes[1].set_ylim(0, 105)
    axes[1].legend(fontsize=8)
    for index, value in enumerate(triggered):
        if value == 0:
            axes[1].text(index, 3, "0%\n(disabled)", ha="center", va="bottom",
                         fontsize=8, color="#666666",
                         bbox={"boxstyle": "round,pad=0.2", "facecolor": "#EEEEEE",
                               "edgecolor": "#AAAAAA"})

    actor = [summaries[n]["avg_actor_calls"] for n in names]
    signal = [summaries[n]["avg_signal_calls"] for n in names]
    verifier = [summaries[n]["avg_verifier_calls"] for n in names]
    axes[2].bar(x, actor, label="Actor calls", color="#B279A2")
    axes[2].bar(x, signal, bottom=actor, label="Confidence calls", color="#ECA82C")
    axes[2].bar(x, verifier,
                bottom=[a + s for a, s in zip(actor, signal)],
                label="Critic calls", color="#FF9DA6")
    axes[2].set_title("Model-call cost")
    axes[2].set_ylabel("Average calls / task")
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.set_xticks(x, labels, rotation=12)
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle("Q2 autonomous verification and correction")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="研究内容二：验证纠错与等预算分析")
    ap.add_argument("traj", nargs="+", help="一个或多个 trajectory JSONL")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--plot-out", default="",
                    help="可选：保存研究内容二汇总 PNG 图表")
    ap.add_argument("--corrections-out", default="",
                    help="可选：导出通过安全门且 error_to_correct 的 Q3 候选 JSONL")
    ap.add_argument("--require-complete-pairs", action="store_true",
                    help="任一条件与第一组不能完全按 task_id/seed 配对时失败")
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
    if args.require_complete_pairs:
        manifest_sets = {
            name: {((trajectory.get("config") or {}).get("task_manifest"))
                   for trajectory in rows}
            for name, rows in groups.items()
        }
        if any(len(values) != 1 or None in values for values in manifest_sets.values()):
            raise SystemExit("正式配对要求每个条件只包含一个非空 task manifest")
        if len({next(iter(values)) for values in manifest_sets.values()}) != 1:
            raise SystemExit("各条件使用了不同的 task manifest")
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
        print("  Q1 score: coverage=%s AUROC=%s Brier=%s ECE10=%s "
              "mean(correct/wrong)=%s/%s error(low/high)=%s/%s" % (
                  fmt(s["confidence_label_coverage"]), fmt(s["confidence_auroc"]),
                  fmt(s["confidence_brier"]), fmt(s["confidence_ece_10"]),
                  fmt(s["mean_confidence_correct"]), fmt(s["mean_confidence_wrong"]),
                  fmt(s["low_confidence_error_rate"]),
                  fmt(s["high_confidence_error_rate"])))

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
            if args.require_complete_pairs and not cmp.get("complete_pairing"):
                raise SystemExit("条件任务不完整，无法进行正式配对比较: %s" % name)

    report = {"conditions": summaries, "paired_against_first": comparisons,
              "sources": args.traj}
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        print("\nJSON 已保存:", args.json_out)
    if args.plot_out:
        plot_summaries(summaries, args.plot_out)
        print("PNG 已保存:", args.plot_out)
    if args.corrections_out:
        corrections = verified_corrections(trajs)
        os.makedirs(os.path.dirname(os.path.abspath(args.corrections_out)), exist_ok=True)
        with open(args.corrections_out, "x", encoding="utf-8") as output:
            for row in corrections:
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        print("Q3 候选已保存: %s（%d 条）" % (args.corrections_out, len(corrections)))


if __name__ == "__main__":
    main()
