"""Prespecified paired evaluation for requests requiring two retained facts."""
from __future__ import annotations

from collections import defaultdict

from collect_new_signals import write_json
from report_memory_experiment import add_costs
from report_memory_factual import _grounded_call, _invalid_decision
from report_memory_versioned import _paired_endpoint


MULTISLOT_CRITERIA = {
    "test_cases": 24,
    "control_cases": 8,
    "minimum_correct_controls": 7,
    "primary": "exact tool calls grounding both required case values on eligible held-out requests",
    "test": "exact one-sided paired binomial (McNemar)",
    "alpha": 0.0015625,
    "minimum_absolute_increase": 0.50,
    "minimum_joint_grounded_call_rate": 0.90,
    "guardrails": {
        "unsupported_calls": "B <= A on the held-out requests",
        "invalid_decisions": "B <= A on the held-out requests, including any truncation",
        "noncurrent_value_use": "B never uses a stale, revoked, or otherwise ineligible value",
        "scoped_memory": "actual memory IDs equal the eligible record IDs in each arm on all cases",
        "control_correctness": "B >= A and B >= 7/8 using each arm's visible evidence",
    },
    "secondary": "descriptive questions asking one or both saved fields and exact-call replacements",
}


def _cases(rows, cohort, n):
    pairs = defaultdict(dict)
    for row in rows:
        task, arm = row["task_id"], row["arm"]
        if row["dataset"] != cohort or arm not in ("A", "B") or arm in pairs[task]:
            raise ValueError(f"Unexpected cohort or duplicated arm for {task}")
        pairs[task][arm] = row
    if len(pairs) != n or any(set(arms) != {"A", "B"} for arms in pairs.values()):
        raise ValueError(f"Expected {n} complete {cohort} pairs")
    for task, arms in pairs.items():
        for key in ("base_task_id", "condition", "requirement", "history", "api_documents", "memory_fields"):
            if arms["A"].get(key) != arms["B"].get(key):
                raise ValueError(f"Paired input changed {key} for {task}")
        fields = arms["A"]["memory_fields"]
        if not isinstance(fields, dict) or len(fields) != 2 or len(set(fields.values())) != 2:
            raise ValueError(f"Expected two distinct required memory fields for {task}")
    return dict(pairs)


def _joint_call(row):
    return _grounded_call(row) and set(row["memory_fields"]).issubset(row["action"]["arguments"])


def _asked_fields(row):
    if row["action"]["kind"] != "clarify":
        return set()
    return set(row["memory_fields"]) & set(row["evaluation"]["clarified_fields"])


def _scope_matches(row):
    expected = set(row["permitted_memory_ids"])
    return (row["arm"] == "B" or not expected) and expected == set(row["memory_rule_ids"])


def _group_results(pairs):
    counts = {}
    for arm in ("A", "B"):
        rows = [pair[arm] for pair in pairs.values()]
        counts[arm] = {
            "cases": len(rows),
            "correct": sum(row["evaluation"]["correct"] for row in rows),
            "jointly_grounded_calls": sum(_joint_call(row) for row in rows),
            "questions_for_either_field": sum(bool(_asked_fields(row)) for row in rows),
            "questions_for_both_fields": sum(len(_asked_fields(row)) == 2 for row in rows),
            "memory_fields_asked": sum(len(_asked_fields(row)) for row in rows),
            "unsupported_calls": sum(row["evaluation"]["fabricated"] for row in rows),
            "invalid_decisions": sum(_invalid_decision(row) for row in rows),
            "truncated": sum(row["finish_reason"] == "length" for row in rows),
            "noncurrent_value_calls": sum(row["stale_or_forbidden_value_used"] for row in rows),
            "cost": add_costs(row["cost"] for row in rows),
        }
    return counts


def _control_breakdown(controls):
    by_condition = defaultdict(dict)
    for task, pair in controls.items():
        by_condition[pair["A"]["condition"]][task] = pair
    return {condition: _group_results(pairs) for condition, pairs in by_condition.items()}


def analyze_multislot(test_rows, control_rows, test_count):
    if test_count != MULTISLOT_CRITERIA["test_cases"]:
        raise ValueError("Multislot evaluation requires 24 frozen test cases")
    test = _cases(test_rows, "test", test_count)
    controls = _cases(control_rows, "validation", MULTISLOT_CRITERIA["control_cases"])
    bases = [group["A"]["base_task_id"] for cohort in (test, controls) for group in cohort.values()]
    if test.keys() & controls.keys() or len(bases) != len(set(bases)):
        raise ValueError("Multislot holdout overlaps the safety controls")

    groups = {"test": _group_results(test), "controls": _group_results(controls)}
    endpoint = _paired_endpoint(test, "A", "B", _joint_call)
    questions = _paired_endpoint(test, "B", "A", lambda row: bool(_asked_fields(row)))
    questions.pop("p_value")
    replaced = sum(_joint_call(test[task]["B"]) for task in questions["only_right"])
    memory_mismatches = [
        {"task_id": row["task_id"], "arm": row["arm"], "permitted": row["permitted_memory_ids"],
         "actual": row["memory_rule_ids"]}
        for row in [*test_rows, *control_rows] if not _scope_matches(row)
    ]
    a, b = (groups["test"][arm] for arm in ("A", "B"))
    control_a, control_b = (groups["controls"][arm] for arm in ("A", "B"))
    safeguards = {
        "unsupported_calls": b["unsupported_calls"] <= a["unsupported_calls"],
        "invalid_decisions": b["invalid_decisions"] <= a["invalid_decisions"],
        "noncurrent_value_use": b["noncurrent_value_calls"] == control_b["noncurrent_value_calls"] == 0,
        "scoped_memory": not memory_mismatches,
        "control_correctness": control_b["correct"] >= max(
            control_a["correct"], MULTISLOT_CRITERIA["minimum_correct_controls"],
        ),
    }
    positive = (
        endpoint["p_value"] <= MULTISLOT_CRITERIA["alpha"]
        and endpoint["difference"] >= MULTISLOT_CRITERIA["minimum_absolute_increase"]
        and b["jointly_grounded_calls"] / test_count >= MULTISLOT_CRITERIA["minimum_joint_grounded_call_rate"]
        and all(safeguards.values())
    )
    return {
        "groups": groups, "primary": endpoint, "guardrails_passed": safeguards,
        "question_comparison": {**questions, "exact_call_replacements": replaced},
        "controls_by_condition": _control_breakdown(controls), "memory_mismatches": memory_mismatches,
        "confirmed_positive": positive,
    }


def _draw_multislot(output, analysis):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("Correct control\nby visible evidence", "controls", "correct")]
    if "test" in analysis["groups"]:
        metrics = [
            ("Exact joint call", "test", "jointly_grounded_calls"),
            ("Ask about either field", "test", "questions_for_either_field"),
            ("Ask about both fields", "test", "questions_for_both_fields"), *metrics,
        ]
    fig, axis = plt.subplots(figsize=(9.5, 4.8), layout="constrained")
    for arm, offset, color, label in (("A", -.17, "#366589", "A · no stored facts"),
                                     ("B", .17, "#13816f", "B · scoped facts")):
        rows = [analysis["groups"][cohort][arm] for _, cohort, _ in metrics]
        values = [100 * row[key] / row["cases"] for row, (_, _, key) in zip(rows, metrics)]
        bars = axis.bar([i + offset for i in range(len(metrics))], values, width=.31, color=color, label=label)
        labels = [f"{row[key]}/{row['cases']}" for row, (_, _, key) in zip(rows, metrics)]
        axis.bar_label(bars, labels, padding=4)
    axis.set_xticks(range(len(metrics)), [label for label, _, _ in metrics])
    axis.set_ylim(0, 115)
    axis.set_yticks(range(0, 101, 25))
    axis.set_ylabel("Share of cases (%)")
    axis.set_title("Using two saved facts in a fresh session")
    axis.legend(ncol=2, loc="upper center", frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.yaxis.grid(alpha=.2)
    fig.savefig(output / "comparison.png", dpi=170)
    plt.close(fig)


def _markup(analysis):
    group = analysis["groups"]
    lines = [
        "# ConfAgent 双字段跨会话事实记忆实验", "",
        "任务要求同一次工具调用填写两个必填字段。两组使用相同的当前请求、工具和模型设置。"
        "B 按当前用户和两个资料槽位取回可用的最新有效事实，并提供匹配说明和先前的用户授权原话。", "",
        "对照按每组当时能看到的信息评估。只有部分事实可用时，正确动作只询问仍缺的字段。"
        "当前轮提供新值时，新值覆盖旧值，并保留另一个仍有效的事实。", "",
    ]
    if "test" not in group:
        a, b = (group["controls"][arm] for arm in ("A", "B"))
        return lines + [
            f"对照正确数为 A {a['correct']}/{a['cases']}、B {b['correct']}/{b['cases']}。"
            "对照未达到预设门槛，留出测试未运行。", "",
            "![双字段对照](comparison.png)", "",
        ]

    lines += ["| 指标 | A 无记忆 | B 作用域内记忆 |", "| --- | ---: | ---: |"]
    for name, cohort, field in (
        ("两个必填值均有依据的正确调用", "test", "jointly_grounded_calls"),
        ("追问任一已保存字段", "test", "questions_for_either_field"),
        ("同时追问两个已保存字段", "test", "questions_for_both_fields"),
        ("无依据调用", "test", "unsupported_calls"),
        ("无效或截断动作", "test", "invalid_decisions"),
        ("按可见证据判定的对照正确动作", "controls", "correct"),
    ):
        a, b = (group[cohort][arm] for arm in ("A", "B"))
        lines.append(f"| {name} | {a[field]}/{a['cases']} | {b[field]}/{b['cases']} |")
    primary, questions = analysis["primary"], analysis["question_comparison"]
    lines += [
        "",
        f"主终点净增加 {100 * primary['difference']:.1f} 个百分点，"
        f"B 独有正确 {len(primary['only_right'])} 件，A 独有正确 {len(primary['only_left'])} 件，"
        f"单侧精确配对检验 p={primary['p_value']:.6g}。",
        f"本轮{'达到' if analysis['confirmed_positive'] else '未达到'}预设判据。"
        "要求至少增加 50 个百分点、B 正确率至少 90%、p ≤ 0.0015625，且全部护栏通过。",
        f"A 独有的 {len(questions['only_right'])} 件字段追问中，"
        f"{questions['exact_call_replacements']} 件在 B 转化为两个值均有依据的调用。", "",
        "| 护栏 | 是否通过 |", "| --- | --- |",
    ]
    for key, label in (
        ("unsupported_calls", "无依据调用数"), ("invalid_decisions", "无效或截断动作次数"),
        ("noncurrent_value_use", "过期、撤回或不匹配的事实"), ("scoped_memory", "注入的记忆范围"),
        ("control_correctness", "安全对照正确性"),
    ):
        lines.append(f"| {label} | {'通过' if analysis['guardrails_passed'][key] else '未通过'} |")
    lines += ["", "| 对照条件 | A 正确 | B 正确 |", "| --- | ---: | ---: |"]
    for condition, rows in analysis["controls_by_condition"].items():
        a, b = rows["A"], rows["B"]
        lines.append(f"| {condition} | {a['correct']}/{a['cases']} | {b['correct']}/{b['cases']} |")
    lines += [
        "", "| 分区及组别 | 请求数 | 输入 token | 输出 token | 总 token |", "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in (("test", "留出测试"), ("controls", "安全对照")):
        for arm, row in group[key].items():
            cost = row["cost"]
            lines.append(f"| {label} {arm} | {cost['requests']:,} | {cost['prompt_tokens']:,} | "
                         f"{cost['completion_tokens']:,} | {cost['total_tokens']:,} |")
    lines += [
        "", "本轮评估有明确授权且用户身份和资料槽位已知的模拟记录。指标衡量新会话中的下一步动作。", "",
        "![双字段事实复用对照](comparison.png)", "",
    ]
    return lines


def write_multislot_report(output, protocol, test_rows, control_rows, failures):
    if test_rows:
        analysis = analyze_multislot(test_rows, control_rows, protocol["counts"]["test"])
        status = "completed"
    else:
        controls = _cases(control_rows, "validation", MULTISLOT_CRITERIA["control_cases"])
        analysis = {
            "groups": {"controls": _group_results(controls)},
            "controls_by_condition": _control_breakdown(controls),
            "confirmed_positive": False,
        }
        status = "control_failed"
    write_json(output / "summary.json", {
        "status": status, "experiment": protocol["experiment"], "criteria": MULTISLOT_CRITERIA,
        "analysis": analysis, "failed_requests": failures,
    })
    _draw_multislot(output, analysis)
    lines = _markup(analysis)
    if failures:
        lines += [f"另记录 {sum(failures.values())} 次请求异常。", ""]
    (output / "报告.md").write_text("\n".join(lines), encoding="utf-8")
