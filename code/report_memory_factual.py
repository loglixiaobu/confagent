"""Predeclared paired analysis for reuse of explicitly saved profile facts."""
from __future__ import annotations

from math import comb

from collect_new_signals import write_json
from report_memory_experiment import add_costs


FACTUAL_CRITERIA = {
    "test_cases": 24,
    "control_cases": 12,
    "primary": "targeted clarification about the saved field on fresh-session test cases",
    "test": "exact one-sided paired binomial (McNemar)",
    "alpha": 0.0125,
    "minimum_absolute_reduction": 0.25,
    "minimum_exact_call_replacement": 0.90,
    "minimum_correct_controls": 11,
    "guardrails": {
        "exact_calls_replace_questions": "at least 90% of eliminated questions yield exact grounded calls",
        "unsupported_calls": "B <= A on test cases",
        "invalid_decisions": "B <= A on test cases, including any truncation",
        "control_correctness": "B >= A and B >= 11/12",
        "permitted_memory_only": "neither arm injects a disallowed record in any case",
    },
}


def _pairs(rows, dataset, count):
    expected_arms = {"A", "B"}
    pairs = {}
    for row in rows:
        if row["dataset"] != dataset:
            raise ValueError(f"Unexpected factual experiment dataset: {row['dataset']}")
        arms = pairs.setdefault(row["task_id"], {})
        if row["arm"] not in expected_arms or row["arm"] in arms:
            raise ValueError(f"Unexpected or duplicate arm for {row['task_id']}")
        arms[row["arm"]] = row
    if len(pairs) != count or any(arms.keys() != expected_arms for arms in pairs.values()):
        raise ValueError(f"Expected {count} complete A/B pairs for {dataset}")
    for task_id, arms in pairs.items():
        for key in ("requested_slot", "memory_field", "condition", "requirement", "history", "api_documents"):
            if arms["A"].get(key) != arms["B"].get(key):
                raise ValueError(f"Paired factual input changed {key} for {task_id}")
    return pairs


def _requested_field_question(row):
    return (row["action"]["kind"] == "clarify"
            and row["memory_field"] in row["evaluation"].get("clarified_fields", []))


def _grounded_call(row):
    return (row["action"]["kind"] == "call" and row["evaluation"]["correct"]
            and not row["evaluation"]["fabricated"])


def _invalid_decision(row):
    return (row["action"]["kind"] not in {"call", "clarify"}
            or row["finish_reason"] == "length"
            or row["evaluation"].get("reason") in {"invalid_argument_schema", "required_value_empty", "wrong_tool"})


def _forbidden_memory(row):
    injected = set(row["memory_rule_ids"])
    permitted = set(row["permitted_memory_ids"]) if row["arm"] == "B" else set()
    return sorted(injected - permitted)


def _group_stats(rows, controls):
    return {
        "test_cases": len(rows),
        "targeted_questions": sum(_requested_field_question(row) for row in rows),
        "exact_grounded_calls": sum(_grounded_call(row) for row in rows),
        "unsupported_calls": sum(row["evaluation"]["fabricated"] for row in rows),
        "invalid_decisions": sum(_invalid_decision(row) for row in rows),
        "truncated_responses": sum(row["finish_reason"] == "length" for row in rows),
        "test_cases_with_memory": sum(bool(row["memory_rule_ids"]) for row in rows),
        "control_cases": len(controls),
        "correct_controls": sum(row["evaluation"]["correct"] for row in controls),
        "control_cases_with_memory": sum(bool(row["memory_rule_ids"]) for row in controls),
        "test_cost": add_costs(row["cost"] for row in rows),
        "control_cost": add_costs(row["cost"] for row in controls),
    }


def analyze_factual(test_rows, validation_rows):
    tests = _pairs(test_rows, "factual_test", FACTUAL_CRITERIA["test_cases"])
    controls = _pairs(validation_rows, "factual_validation", FACTUAL_CRITERIA["control_cases"])
    if tests.keys() & controls.keys():
        raise ValueError("A factual test case was also used as a control")
    groups = {
        arm: _group_stats([pair[arm] for pair in tests.values()], [pair[arm] for pair in controls.values()])
        for arm in ("A", "B")
    }
    eliminated, introduced = [], []
    both = neither = 0
    for task_id, pair in tests.items():
        a, b = (_requested_field_question(pair[arm]) for arm in ("A", "B"))
        if a and b:
            both += 1
        elif not a and not b:
            neither += 1
        else:
            detail = {
                "task_id": task_id, "requested_slot": pair["A"]["requested_slot"], "memory_field": pair["A"]["memory_field"],
                "A": pair["A"]["action"], "B": pair["B"]["action"],
                "B_exact_grounded_call": _grounded_call(pair["B"]),
            }
            (eliminated if a else introduced).append(detail)
    wins, losses = len(eliminated), len(introduced)
    discordant = wins + losses
    p_value = sum(comb(discordant, k) for k in range(wins, discordant + 1)) / 2**discordant
    absolute_reduction = (wins - losses) / len(tests)
    exact_replacements = sum(row["B_exact_grounded_call"] for row in eliminated)
    disallowed = [
        {"task_id": row["task_id"], "arm": row["arm"], "memory_ids": ids}
        for row in [*test_rows, *validation_rows] if (ids := _forbidden_memory(row))
    ]
    a, b = groups["A"], groups["B"]
    guardrails = {
        "exact_calls_replace_questions": (
            wins > 0 and exact_replacements / wins >= FACTUAL_CRITERIA["minimum_exact_call_replacement"]
        ),
        "unsupported_calls": b["unsupported_calls"] <= a["unsupported_calls"],
        "invalid_decisions": b["invalid_decisions"] <= a["invalid_decisions"],
        "control_correctness": b["correct_controls"] >= max(a["correct_controls"], FACTUAL_CRITERIA["minimum_correct_controls"]),
        "permitted_memory_only": not disallowed,
    }
    return {
        "groups": groups, "p_value": p_value, "absolute_reduction": absolute_reduction,
        "eliminated_questions": eliminated, "introduced_questions": introduced,
        "questions_in_both": both, "questions_in_neither": neither,
        "exact_call_replacements": exact_replacements, "disallowed_memory": disallowed,
        "guardrails_passed": guardrails,
        "confirmed_positive": (
            p_value <= FACTUAL_CRITERIA["alpha"]
            and absolute_reduction >= FACTUAL_CRITERIA["minimum_absolute_reduction"]
            and all(guardrails.values())
        ),
    }


def _plot_factual(output, analysis):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    measures = (
        ("Targeted question", "targeted_questions", "test_cases"),
        ("Exact grounded call", "exact_grounded_calls", "test_cases"),
        ("Unsupported call", "unsupported_calls", "test_cases"),
        ("Correct control", "correct_controls", "control_cases"),
    )
    has_test = analysis["groups"]["A"]["test_cases"] > 0
    if not has_test:
        measures = measures[-1:]
    fig, axis = plt.subplots(figsize=(8.8, 4.6), layout="constrained")
    for arm, shift, color, label in (("A", -0.18, "#366589", "A · no saved fact"),
                                     ("B", 0.18, "#13816f", "B · saved fact when applicable")):
        row = analysis["groups"][arm]
        values = [100 * row[numerator] / row[denominator] for _, numerator, denominator in measures]
        bars = axis.bar([x + shift for x in range(len(measures))], values, width=0.33, color=color, label=label)
        axis.bar_label(bars, fmt="%.1f%%", padding=4)
    axis.set_xticks(range(len(measures)), [label for label, _, _ in measures])
    axis.set_ylim(0, 120)
    axis.set_yticks(range(0, 101, 20))
    axis.set_ylabel("Share of cases (%)")
    axis.set_title("Reusing saved profile facts in a fresh session" if has_test else "Factual memory controls")
    axis.legend(loc="upper center", ncol=2, frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.yaxis.grid(alpha=0.2)
    fig.savefig(output / "comparison.png", dpi=170)
    plt.close(fig)


def write_factual_report(output, protocol, test_rows, validation_rows, failures):
    if not test_rows:
        _write_validation_failure(output, protocol, validation_rows, failures)
        return
    analysis = analyze_factual(test_rows, validation_rows)
    write_json(output / "summary.json", {
        "status": "completed", "experiment": protocol["experiment"],
        "memory_sha256": protocol["memory_sha256"],
        "criteria": FACTUAL_CRITERIA, "analysis": analysis, "failed_requests": failures,
    })
    _plot_factual(output, analysis)
    a, b = (analysis["groups"][arm] for arm in ("A", "B"))
    tests, controls = FACTUAL_CRITERIA["test_cases"], FACTUAL_CRITERIA["control_cases"]
    eliminated, introduced = len(analysis["eliminated_questions"]), len(analysis["introduced_questions"])
    passed = analysis["guardrails_passed"]
    status = "达到" if analysis["confirmed_positive"] else "未达到"
    lines = [
        "# ConfAgent 跨会话事实记忆实验", "",
        f"使用 Qwen3-8B 对 {tests} 个独立测试请求和 {controls} 个对照请求逐例配对。"
        "两组使用相同请求和推理设置。B 组在用户身份、适用范围和事实状态匹配时注入已明确保存的资料。",
        "",
        "测试衡量能否复用用户已提供的资料，减少新会话对同一字段的重复追问。"
        "资料、会话和工具返回均为受控模拟，结论限于下一步动作和本轮测试。",
        "",
        "| 指标 | A（无资料记忆） | B（资料记忆） |",
        "| --- | ---: | ---: |",
        f"| 针对已保存字段的追问 | {a['targeted_questions']}/{tests} | {b['targeted_questions']}/{tests} |",
        f"| 正确且有依据的调用 | {a['exact_grounded_calls']}/{tests} | {b['exact_grounded_calls']}/{tests} |",
        f"| 无依据调用 | {a['unsupported_calls']}/{tests} | {b['unsupported_calls']}/{tests} |",
        f"| 无效动作或截断 | {a['invalid_decisions']}/{tests} | {b['invalid_decisions']}/{tests} |",
        f"| 对照组正确动作 | {a['correct_controls']}/{controls} | {b['correct_controls']}/{controls} |",
        "",
        f"A 独有追问 {eliminated} 题，B 独有追问 {introduced} 题。"
        f"追问率配对下降 {100 * analysis['absolute_reduction']:.1f} 个百分点，"
        f"单侧精确配对检验 p={analysis['p_value']:.6g}。"
        f"A 独有追问中有 {analysis['exact_call_replacements']}/{eliminated} 题在 B 组变成正确调用。",
        "",
        f"本轮{status}预设判据。阈值为追问率至少下降 25 个百分点、p ≤ 0.0125，"
        "且至少 90% 的减少追问转化为正确且有依据的调用。其余条件要求无依据调用和无效动作不增加，"
        "B 组对照正确数不少于 A 组且至少为 11/12，两个组都只注入允许使用的记忆。",
        "",
        "| 护栏 | 是否通过 |",
        "| --- | --- |",
    ]
    for key, label in (
        ("exact_calls_replace_questions", "减少的追问转化为正确调用"),
        ("unsupported_calls", "无依据调用数"),
        ("invalid_decisions", "无效动作或截断数"),
        ("control_correctness", "对照请求正确性"),
        ("permitted_memory_only", "记忆适用范围"),
    ):
        lines.append(f"| {label} | {'通过' if passed[key] else '未通过'} |")
    lines.extend(["", "| 阶段和组别 | 请求数 | 输入 token | 输出 token | 总 token |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for label, key in (("主测试", "test_cost"), ("对照", "control_cost")):
        for arm, row in (("A", a), ("B", b)):
            cost = row[key]
            lines.append(f"| {label} {arm} | {cost['requests']:,} | {cost['prompt_tokens']:,} |"
                         f" {cost['completion_tokens']:,} | {cost['total_tokens']:,} |")
    if failures:
        lines.extend(["", f"另记录 {sum(failures.values())} 次请求异常。"])
    lines.extend(["", "![跨会话事实复用对照](comparison.png)", ""])
    (output / "报告.md").write_text("\n".join(lines), encoding="utf-8")


def _write_validation_failure(output, protocol, validation_rows, failures):
    controls = _pairs(validation_rows, "factual_validation", FACTUAL_CRITERIA["control_cases"])
    groups = {arm: _group_stats([], [pair[arm] for pair in controls.values()]) for arm in ("A", "B")}
    mismatches = [
        {"task_id": row["task_id"], "arm": row["arm"], "injected": row["memory_rule_ids"],
         "permitted": row["permitted_memory_ids"]}
        for row in validation_rows if set(row["memory_rule_ids"]) != set(row["permitted_memory_ids"])
    ]
    analysis = {"groups": groups, "control_memory_mismatches": mismatches, "confirmed_positive": False}
    write_json(output / "summary.json", {
        "status": "validation_failed", "experiment": protocol["experiment"],
        "memory_sha256": protocol["memory_sha256"],
        "criteria": FACTUAL_CRITERIA, "analysis": analysis, "failed_requests": failures,
    })
    _plot_factual(output, analysis)
    lines = [
        "# ConfAgent 跨会话事实记忆实验", "",
        "对照未达到预设门槛，留出测试未运行。", "",
        "| 对照组 | 正确动作 | 请求数 | 总 token |",
        "| --- | ---: | ---: | ---: |",
    ]
    for arm, row in groups.items():
        cost = row["control_cost"]
        lines.append(f"| {arm} | {row['correct_controls']}/{row['control_cases']} | "
                     f"{cost['requests']:,} | {cost['total_tokens']:,} |")
    lines.extend([
        "", f"有 {len(mismatches)} 条决策的实际记忆注入与对照预期不一致。",
        "", "![事实记忆对照](comparison.png)", "",
    ])
    (output / "报告.md").write_text("\n".join(lines), encoding="utf-8")
