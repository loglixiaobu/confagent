"""Aggregate paired decisions and write the memory experiment artifacts."""
from __future__ import annotations

from collections import Counter

from collect_new_signals import write_json

MEASURES = (
    ("fabrication_rate", "编造参数率", "Fabrication", "clarify", "fabricated"),
    ("reasonable_clarification_rate", "合理追问率", "Correct clarification", "clarify", "reasonable_clarification"),
    ("correct_call_rate", "正确调用率", "Correct call", "call", "correct"),
    ("unnecessary_clarification_rate", "不必要追问率", "Redundant question", "call",
     "unnecessary_clarification"),
)
DATASET_LABELS = {"original": ("原测试集", "Original tasks"), "stress": ("新增压力测试", "Added challenge tasks")}


def rate(rows, key):
    count = sum(row["evaluation"][key] for row in rows)
    return {"count": count, "n": len(rows), "rate": count / len(rows)}


def add_costs(costs):
    totals = dict.fromkeys(("requests", "prompt_tokens", "completion_tokens", "total_tokens"), 0)
    for cost in costs:
        for key in totals:
            totals[key] += cost[key]
    return totals


def summarize_test(results):
    grouped = {
        arm: {row["task_id"]: row for row in results if row["arm"] == arm}
        for arm in ("A", "B")
    }
    if len(results) != 40 or len(grouped["A"]) != 20 or grouped["A"].keys() != grouped["B"].keys():
        raise ValueError("Expected 20 complete A/B pairs")
    summary = {}
    for arm, decisions in grouped.items():
        cohort = {"clarify": [], "call": []}
        for row in decisions.values():
            cohort[row["expected"]["kind"]].append(row)
        if any(len(rows) != 10 for rows in cohort.values()):
            raise ValueError("Each arm needs ten missing-information and ten complete-information tasks")
        summary[arm] = {
            **{name: rate(cohort[kind], key) for name, _, _, kind, key in MEASURES},
            "cost": add_costs(row["cost"] for row in decisions.values()),
            "action_kinds": dict(Counter(row["action"]["kind"] for row in decisions.values())),
            "cases_with_memory_injected": sum(bool(row["memory_rule_ids"]) for row in decisions.values()),
        }
    paired = {}
    for name, _, _, kind, key in MEASURES:
        counts = {"both": 0, "only_A": 0, "only_B": 0, "neither": 0}
        direction = {(True, True): "both", (True, False): "only_A",
                     (False, True): "only_B", (False, False): "neither"}
        for task_id, first in grouped["A"].items():
            if first["expected"]["kind"] == kind:
                second = grouped["B"][task_id]
                counts[direction[(first["evaluation"][key], second["evaluation"][key])]] += 1
        paired[name] = {
            **counts, "change_points": 10 * (counts["only_B"] - counts["only_A"]),
        }
    return summary, paired


def plot_results(output, comparisons, attempts):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(comparisons) or 1, figsize=(9, 7 if comparisons else 4.5),
                             squeeze=False, layout="constrained")
    if comparisons:
        for axis, (key, (summary, _)) in zip(axes.flat, comparisons.items()):
            labels = [label for _, _, label, _, _ in MEASURES]
            positions = list(range(len(labels)))
            for arm, shift, color, legend in (
                ("A", -0.18, "#366589", "A · M0"), ("B", 0.18, "#13816f", "B · M1"),
            ):
                bars = axis.bar([pos + shift for pos in positions],
                                [summary[arm][measure[0]]["rate"] * 100 for measure in MEASURES],
                                width=0.32, color=color, label=legend)
                axis.bar_label(bars, fmt="%g%%", padding=3)
            axis.set_xticks(positions, labels)
            axis.set_ylim(0, 118)
            axis.set_yticks(range(0, 101, 20))
            axis.set_ylabel("Eligible tasks (%)")
            axis.set_title(DATASET_LABELS[key][1])
            if axis is axes[0, 0]:
                axis.legend(loc="upper center", ncol=2, frameon=False)
    else:
        axis = axes[0, 0]
        for shift, color, label, check in (
            (-0.18, "#366589", "Correct next actions", lambda row: row["evaluation"]["correct"]),
            (0.18, "#13816f", "Memory injected", lambda row: bool(row["memory_rule_ids"])),
        ):
            bars = axis.bar([i + shift for i in range(len(attempts))],
                            [sum(check(row) for row in a["validation"]) for a in attempts],
                            color=color, label=label, width=0.32)
            axis.bar_label(bars, padding=3)
        axis.set_xticks(range(len(attempts)), [f"Candidate {attempt['attempt']}" for attempt in attempts])
        axis.set_yticks(range(5))
        axis.set_ylim(0, 5)
        axis.set_ylabel("Validation tasks")
        axis.set_title("Memory validation results")
        axis.legend(frameon=False)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.yaxis.grid(alpha=0.2)
    fig.savefig(output / "comparison.png", dpi=170)
    plt.close(fig)


def findings(summary, paired):
    before = summary["A"]["fabrication_rate"]["count"]
    change = paired["fabrication_rate"]["change_points"]
    calls = paired["correct_call_rate"]["change_points"]
    if before == 0:
        opening = "基线在 10 个缺信息任务中没有编造参数，本组样本的基线已达零编造。"
    else:
        opening = ""
    if change == 0:
        fabrication = "加入记忆后，编造参数率持平。"
    else:
        verb = "下降" if change < 0 else "上升"
        fabrication = f"加入记忆后，编造参数率{verb} {abs(change)} 个百分点。"
    if calls == 0:
        call_rate = "信息齐全任务的正确调用率持平。"
    else:
        verb = "下降" if calls < 0 else "上升"
        call_rate = f"信息齐全任务的正确调用率{verb} {abs(calls)} 个百分点。"
    return opening + fabrication + call_rate


def write_report(output, protocol, results, memory, development, attempts, failures):
    comparisons = {
        dataset: summarize_test([row for row in results if row["dataset"] == dataset])
        for dataset in DATASET_LABELS if results
    }
    summary, paired = comparisons.get("original", ({}, {}))
    stress_groups, stress_paired = comparisons.get("stress", ({}, {}))
    costs = {
        "development_decisions": add_costs(row["cost"] for row in development),
        "development_self_report": add_costs(row["confidence"]["cost"] for row in development),
        "synthesis": add_costs(attempt["cost"] for attempt in attempts),
        "validation": add_costs(row["cost"] for attempt in attempts for row in attempt["validation"]),
    }
    status = "completed" if memory is not None else "memory_build_failed"
    write_json(output / "summary.json", {
        "status": status, "memory_rules_sha256": memory["rules_sha256"] if memory else None,
        "build_cost": costs, "groups": summary, "paired": paired, "failed_request_attempts": failures,
        "stress": {"groups": stress_groups, "paired": stress_paired},
        "test_cost": {arm: add_costs(row["cost"] for row in results if row["arm"] == arm)
                      for arm in ("A", "B")} if results else {},
    })
    plot_results(output, comparisons, attempts)

    settings = protocol["settings"]
    lines = [
        "# ConfAgent 必填参数记忆实验",
        "",
        f"使用 {settings['model']} 在 10 个开发实例和历史错误轨迹上提炼规则，"
        "用 4 个独立实例核验。原测试集和新增压力测试各 20 题，各含 10 道缺信息题和 10 道信息齐全题。"
        "压力测试保留相似对象、其他场次或候选默认值等干扰，使缺失的字段看起来可以填写。"
        "两套测试均在运行前冻结，只评测下一步动作。工具模拟返回单独记录。",
        "",
        "本轮规则按必填参数问题匹配可用工具，适用条件由模型从开发反馈中提炼。"
        "沿用上一轮的四个核验实例，要求四个动作均正确，且每条规则至少实际注入一次。"
        "上一轮的两次构建失败记录单独保留，两套测试题均未参与前轮构建。",
        "",
        f"两组沿用同一 ReAct 基础提示、单步输出约定和检查预算（最多 {settings['max_checks']} 次）。"
        f"推理温度为 {settings['temperature']}，种子为 {settings['seed']}，"
        f"每个决策的输入与输出共用 {settings['total_token_cap']} token 上限，记忆计入输入。"
        "同一任务的两个组在同一模型副本上顺序运行，先运行 A 或 B 的任务数在每类题中相同。",
        "",
        "## 经验构建与核验",
        "",
    ]
    for attempt in attempts:
        decisions = attempt["validation"]
        if not decisions:
            lines.append(f"- 候选 {attempt['attempt']} 未通过规则格式检查。")
            continue
        passed = sum(row["evaluation"]["correct"] for row in decisions)
        details = "、".join(f"{row['task_id']} {'通过' if row['evaluation']['correct'] else '未通过'}"
                            for row in decisions)
        covered = sum(bool(task_ids) for task_ids in attempt["rule_coverage"].values())
        lines.append(f"- 候选 {attempt['attempt']} 的决策通过 {passed}/4（{details}）。"
                     f"核验时实际注入了 {covered}/{len(attempt['rules'])} 条规则。")
    if memory is not None:
        lines.append(f"- 候选 {memory['accepted_attempt']} 保存为 M1，共 {len(memory['rules'])} 条。")
    lines.extend([
        "",
        "| 阶段 | 请求数 | 输入 token | 输出 token | 总 token |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    labels = ("开发任务决策", "开发自陈置信度", "经验提炼", "独立核验")
    for label, cost in zip(labels, costs.values()):
        lines.append(cost_line(label, cost))
    for dataset, (summary, paired) in comparisons.items():
        lines.extend([
            "",
            f"## {DATASET_LABELS[dataset][0]}",
            "",
            "| 指标 | A（M0） | B（M1） | 变化（百分点） |",
            "| --- | ---: | ---: | ---: |",
        ])
        for measure, label, _, _, _ in MEASURES:
            values = [summary[arm][measure] for arm in ("A", "B")]
            columns = " | ".join(f"{v['count']}/{v['n']} ({v['rate']:.0%})" for v in values)
            lines.append(f"| {label} | {columns} | {paired[measure]['change_points']:+d} |")
        lines.extend([
            "",
            findings(summary, paired),
            "",
            "| 测试组 | 请求数 | 输入 token | 输出 token | 总 token |",
            "| --- | ---: | ---: | ---: | ---: |",
            cost_line("A（M0）", summary["A"]["cost"]),
            cost_line("B（M1）", summary["B"]["cost"]),
            "",
            f"B 组在 {summary['B']['cases_with_memory_injected']}/20 个实例注入记忆。"
            f"总 token 变化为 {summary['B']['cost']['total_tokens'] - summary['A']['cost']['total_tokens']:+,}。",
        ])
        changed = changed_fabrications([row for row in results if row["dataset"] == dataset])
        if changed:
            lines.extend(["", "编造判定发生变化的任务", ""])
            lines.extend(changed)
    if comparisons:
        lines.extend(["", "每套测试的编造率分母为 10 个缺信息场景。观察到的变化以逐例配对结果为准。"])
    else:
        lines.extend(["", "经验构建失败。入库要求四个决策全部通过，且每条规则都曾实际注入。测试阶段未运行。"])
    if failures:
        lines.extend(["", f"另记录 {sum(failures.values())} 次请求异常，无服务端 token 用量。"])
    lines.extend(["", "![实验图](comparison.png)", ""])
    (output / "报告.md").write_text("\n".join(lines), encoding="utf-8")


def cost_line(label, cost):
    return (f"| {label} | {cost['requests']:,} | {cost['prompt_tokens']:,}"
            f" | {cost['completion_tokens']:,} | {cost['total_tokens']:,} |")


def changed_fabrications(rows):
    pairs = {}
    for row in rows:
        if row["expected"]["kind"] == "clarify":
            pairs.setdefault(row["task_id"], {})[row["arm"]] = row
    changed = []
    for task_id, pair in pairs.items():
        a, b = pair["A"]["evaluation"], pair["B"]["evaluation"]
        if a["fabricated"] != b["fabricated"]:
            arm, evaluation = ("A", a) if a["fabricated"] else ("B", b)
            fields = "、".join(evaluation["unsupported_fields"])
            changed.append(f"- `{task_id}` 的无依据字段由 {arm} 组填入（{fields}）。")
    return changed
