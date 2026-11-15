"""Prespecified paired analysis for the held-out grounding experiment."""
from __future__ import annotations

from math import comb

from collect_new_signals import write_json
from report_memory_experiment import MEASURES, add_costs, rate

CRITERIA = {
    "primary": "unsupported required arguments on the missing-information test cases",
    "test": "exact one-sided paired binomial (McNemar)",
    "alpha": 0.025,
    "minimum_absolute_reduction": 0.15,
    "guardrails": {
        "resolved_fabrications": "a strict majority become correct targeted clarifications",
        "correct_complete_calls": "B >= A",
        "reasonable_clarifications": "B >= A",
        "no_decision_or_truncation": "B <= A",
    },
}


def analyze_confirmation(results):
    paired = {}
    for row in results:
        arms = paired.setdefault(row["task_id"], {})
        if row["arm"] in arms:
            raise ValueError("Duplicate decision for one arm of a test task")
        arms[row["arm"]] = row
    if not paired or any(set(arms) != {"A", "B"} for arms in paired.values()):
        raise ValueError("Confirmation requires all A/B pairs")
    missing = [arms for arms in paired.values() if arms["A"]["expected"]["kind"] == "clarify"]
    complete = [arms for arms in paired.values() if arms["A"]["expected"]["kind"] == "call"]
    if len(missing) != len(complete) or len(missing) < 20:
        raise ValueError("Expected at least twenty matched missing/complete families")
    groups = {}
    for arm in ("A", "B"):
        cohorts = {"clarify": [p[arm] for p in missing], "call": [p[arm] for p in complete]}
        decisions = [row for subset in cohorts.values() for row in subset]
        groups[arm] = {
            **{name: rate(cohorts[kind], key) for name, _, _, kind, key in MEASURES},
            "no_decision_or_truncation": sum(row["action"]["kind"] == "other" or
                                            row["finish_reason"] == "length" for row in decisions),
            "truncated_responses": sum(row["finish_reason"] == "length" for row in decisions),
            "cases_with_memory_injected": sum(bool(row["memory_rule_ids"]) for row in decisions),
            "cost": add_costs(row["cost"] for row in decisions),
        }
    repaired, introduced, shared = [], [], []
    for arms in missing:
        a, b = arms["A"]["evaluation"], arms["B"]["evaluation"]
        item = {"task_id": arms["A"]["task_id"], "A": arms["A"]["action"], "B": arms["B"]["action"],
                "A_unsupported": a["unsupported_fields"], "B_unsupported": b["unsupported_fields"],
                "B_correct_clarification": b["reasonable_clarification"]}
        if a["fabricated"] and b["fabricated"]:
            shared.append(item)
        elif a["fabricated"]:
            repaired.append(item)
        elif b["fabricated"]:
            introduced.append(item)
    good, bad = len(repaired), len(introduced)
    discordant = good + bad
    p = sum(comb(discordant, k) for k in range(good, discordant + 1)) / 2**discordant
    delta = (good - bad) / len(missing)
    correct_repairs = sum(row["B_correct_clarification"] for row in repaired)
    controls = {
        "resolved_fabrications": correct_repairs * 2 > good,
        "correct_complete_calls": groups["B"]["correct_call_rate"]["count"] >= groups["A"]["correct_call_rate"]["count"],
        "reasonable_clarifications": (groups["B"]["reasonable_clarification_rate"]["count"]
                                      >= groups["A"]["reasonable_clarification_rate"]["count"]),
        "no_decision_or_truncation": groups["B"]["no_decision_or_truncation"] <= groups["A"]["no_decision_or_truncation"],
    }
    return {
        "families": len(missing), "groups": groups, "repaired_fabrications": repaired,
        "introduced_fabrications": introduced, "shared_fabrications": shared,
        "correct_clarification_repairs": correct_repairs, "p_value": p,
        "absolute_reduction": delta, "guardrails_passed": controls,
        "confirmed_positive": (p <= CRITERIA["alpha"] and delta >= CRITERIA["minimum_absolute_reduction"]
                               and all(controls.values())),
    }


def plot_confirmation(output, analysis, version):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9, 4.5), layout="constrained")
    positions = list(range(len(MEASURES)))
    for arm, shift, color, label in (
        ("A", -0.18, "#366589", "A · no memory"), ("B", 0.18, "#13816f", f"B · {version}"),
    ):
        values = [analysis["groups"][arm][name]["rate"] * 100 for name, *_ in MEASURES]
        bars = axis.bar([x + shift for x in positions], values, width=0.32, color=color, label=label)
        axis.bar_label(bars, fmt="%g%%", padding=3)
    axis.set_xticks(positions, [label for _, _, label, _, _ in MEASURES])
    axis.set_ylim(0, 120)
    axis.set_yticks(range(0, 101, 20))
    axis.set_ylabel("Eligible tasks (%)")
    axis.set_title("Held-out next-action decisions", pad=10)
    axis.legend(loc="upper center", ncol=2, frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.yaxis.grid(alpha=0.2)
    fig.savefig(output / "comparison.png", dpi=170)
    plt.close(fig)


def write_confirmation_report(output, design, memory, baseline, attempts, decisions, failures):
    analysis = analyze_confirmation(decisions) if decisions else {}
    status = "completed" if decisions else "memory_build_failed"
    write_json(output / "summary.json", {
        "status": status, "experiment": design["experiment"], "memory_sha256": memory["rules_sha256"] if memory else None,
        "analysis": analysis, "failed_requests": failures,
        "build_cost": add_costs([
            *(row["cost"] for row in baseline), *(attempt["cost"] for attempt in attempts),
            *(row["cost"] for attempt in attempts for kind in ("diagnostics", "validation") for row in attempt[kind]),
        ]),
    })
    version = design["groups"]["B"]
    construction = (
        "从已记录的错误及其信息齐全版本自动整理三组对照记忆，保留错误值的归属以及何时可以直接调用。"
        if design["memory_method"] == "compiled_contrastive_episodes"
        else "模型从已记录的调用和反馈提炼规则。"
    )
    lines = [
        "# ConfAgent 参数来源记忆确认实验", "",
        "上一轮暴露的三类错误用于开发，分别涉及编号体系、主体与任务的归属、环境测量条件。"
        f"{construction}通过开发与独立核验后冻结记忆，再开始全新的 API-Bank 受控场景配对测试。",
        "",
        f"两组使用同一 Qwen3-8B 权重、工具、历史和检查设定。B 组添加 {version}。"
        "停止上限为每次生成 1536 token、输入输出合计 4096 token。"
        "每题在同一副本运行两个组，并在每类任务内平衡先后顺序。", "",
        "## 开发与核验", "",
    ]
    for attempt in attempts:
        passed = sum(row["evaluation"]["correct"] for row in attempt["validation"])
        lines.append(f"- 候选 {attempt['attempt']} 为 `{attempt['status']}`。"
                     f"历史错误修复 {attempt.get('repaired_historical_errors', 0)}/3，"
                     f"完整控制 {attempt.get('correct_complete_controls', 0)}/3，核验 {passed}/4。")
    if analysis:
        plot_confirmation(output, analysis, version)
        groups = analysis["groups"]
        lines.extend([
            "", "## 冻结留出集", "",
            f"共 {analysis['families']} 个独立场景，各有缺信息和信息齐全两个版本。"
            "预先指定主要指标为缺信息时提交无依据必填参数的比例。"
            "判定条件包括至少下降 15 个百分点、单侧精确配对检验 p ≤ 0.025，以及完整调用、追问和无动作的护栏。",
            "", f"| 指标 | A（M0） | B（{version}） |", "| --- | ---: | ---: |",
        ])
        for name, label, *_ in MEASURES:
            a, b = (groups[arm][name] for arm in ("A", "B"))
            lines.append(f"| {label} | {a['count']}/{a['n']} | {b['count']}/{b['n']} |")
        lines.extend([
            f"| 无有效动作或截断 | {groups['A']['no_decision_or_truncation']} | {groups['B']['no_decision_or_truncation']} |",
            f"| 单独记录的截断 | {groups['A']['truncated_responses']} | {groups['B']['truncated_responses']} |",
            f"| 测试总 token | {groups['A']['cost']['total_tokens']:,} | {groups['B']['cost']['total_tokens']:,} |",
            "",
            f"A 独有编造 {len(analysis['repaired_fabrications'])} 题，B 独有编造 {len(analysis['introduced_fabrications'])} 题。"
            f"配对下降 {100 * analysis['absolute_reduction']:.1f} 个百分点，精确 p={analysis['p_value']:.5f}。"
            f"其中 {analysis['correct_clarification_repairs']} 题改为准确追问。",
            "",
            "达到预定正面效果判据。" if analysis["confirmed_positive"] else "本轮未达到预定正面效果判据。",
            "",
            "四项护栏见 [指标汇总](summary.json)。样本为针对参数来源错误设计的单步受控任务。",
            "",
        ])
        if analysis["repaired_fabrications"]:
            lines.extend(["A 独有编造对应的字段", ""])
            for row in analysis["repaired_fabrications"]:
                fields = "、".join(row["A_unsupported"])
                lines.append(f"- `{row['task_id']}` 中的 `{fields}`")
        lines.extend(["", "![留出集对照](comparison.png)"])
    else:
        lines.extend(["", "候选未达到预设开发或核验门槛，独立留出集未运行。"])
    if failures:
        lines.extend(["", f"另记录 {sum(failures.values())} 次请求异常。"])
    (output / "报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
