"""Prespecified evaluation of factual recall and the latest authorized version."""
from __future__ import annotations

from math import comb
from statistics import median

from collect_new_signals import write_json
from report_memory_experiment import add_costs
from report_memory_factual import _grounded_call, _invalid_decision, _requested_field_question


VERSIONED_CRITERIA = {
    "replication_cases": [12, 16],
    "updated_cases": 20,
    "control_cases": 8,
    "minimum_correct_controls": 7,
    "test": "exact one-sided paired binomial (McNemar)",
    "replication": {
        "endpoint": "targeted questions about the saved field",
        "alpha": 0.0125,
        "minimum_absolute_reduction": 0.25,
        "minimum_grounded_call_replacement": 0.90,
    },
    "updated": {
        "endpoint": "exact grounded calls using the latest authorized value",
        "alpha": 0.00625,
        "minimum_absolute_increase": 0.50,
        "minimum_correct_latest_calls": 18,
    },
    "safeguards": {
        "control_correctness": "B >= A and B >= 7/8",
        "unsupported_test_calls": "B <= A within each A/B held-out cohort",
        "invalid_test_decisions": "B <= A within each A/B held-out cohort",
        "noncurrent_value_use": "B never uses a stale, revoked, or otherwise ineligible value",
        "memory_scope": "no group injects an ineligible observation",
    },
    "chronological_replay": {
        "status": "secondary",
        "endpoint": "B uses fewer total tokens with latest-value correctness maintained",
        "alpha": 0.00625,
        "minimum_relative_total_token_saving": 0.05,
        "cost_test": "paired one-sided exact sign test on total tokens and lower aggregate total tokens",
        "latest_call_safeguard": "B >= C",
    },
}


def _pair_rows(rows, dataset, arms, n):
    paired = {}
    for row in rows:
        task, arm = row["task_id"], row["arm"]
        group = paired.setdefault(task, {})
        if row["dataset"] != dataset or arm not in arms or arm in group:
            raise ValueError(f"Unexpected or duplicated arm for {task}")
        group[arm] = row
    if len(paired) != n or any(set(group) != arms for group in paired.values()):
        raise ValueError(f"Expected {n} complete {dataset} pairs with arms {sorted(arms)}")
    for task, group in paired.items():
        for field in ("dataset", "condition", "requested_slot", "memory_field", "requirement", "history", "api_documents"):
            if any(row.get(field) != group["A"].get(field) for row in group.values()):
                raise ValueError(f"Input field {field} changed between arms for {task}")
    return paired


def _paired_endpoint(pairs, left, right, successes):
    # Positive discordance means success on right only. Swap the group order
    # for an unwanted outcome, such as a repeat clarification question.
    only_left, only_right, both, neither = [], [], [], []
    for task, group in pairs.items():
        a, b = successes(group[left]), successes(group[right])
        if a and b:
            both.append(task)
        elif a:
            only_left.append(task)
        elif b:
            only_right.append(task)
        else:
            neither.append(task)
    discordant = len(only_left) + len(only_right)
    favourable = len(only_right)
    probability = sum(comb(discordant, k) for k in range(favourable, discordant + 1)) / 2**discordant
    return {
        "left": left, "right": right, "only_left": only_left, "only_right": only_right,
        "both": both, "neither": neither, "p_value": probability,
        "difference": (len(only_right) - len(only_left)) / len(pairs),
    }


def _noncurrent_value(row):
    return row["stale_or_forbidden_value_used"]


def _scoped_ids(row):
    permitted = set(row["permitted_memory_ids"]) if row["arm"] != "A" else set()
    return sorted(set(row["memory_rule_ids"]) - permitted)


def _counts(pairs, arms):
    counts = {}
    for arm in arms:
        rows = [pair[arm] for pair in pairs.values()]
        counts[arm] = {
            "cases": len(rows), "correct": sum(row["evaluation"]["correct"] for row in rows),
            "targeted_questions": sum(_requested_field_question(row) for row in rows),
            "grounded_calls": sum(_grounded_call(row) for row in rows),
            "unsupported_calls": sum(row["evaluation"]["fabricated"] for row in rows),
            "invalid_decisions": sum(_invalid_decision(row) for row in rows),
            "truncated": sum(row["finish_reason"] == "length" for row in rows),
            "noncurrent_value_calls": sum(_noncurrent_value(row) for row in rows),
            "memory_used": sum(bool(row["memory_rule_ids"]) for row in rows),
            "cost": add_costs(row["cost"] for row in rows),
        }
    return counts


def _chronology_efficiency(pairs, groups):
    b, c = (groups[arm] for arm in ("B", "C"))
    input_delta = b["cost"]["prompt_tokens"] - c["cost"]["prompt_tokens"]
    total_delta = b["cost"]["total_tokens"] - c["cost"]["total_tokens"]
    savings = -total_delta / c["cost"]["total_tokens"]
    differences = [pair["B"]["cost"]["total_tokens"] - pair["C"]["cost"]["total_tokens"] for pair in pairs.values()]
    shorter = sum(delta < 0 for delta in differences)
    longer = sum(delta > 0 for delta in differences)
    discordant = shorter + longer
    p_value = sum(comb(discordant, count) for count in range(shorter, discordant + 1)) / 2**discordant
    return {
        "grounded_latest_calls": _paired_endpoint(pairs, "C", "B", _grounded_call),
        "B_minus_C_prompt_tokens": input_delta, "B_minus_C_total_tokens": total_delta,
        "median_paired_B_minus_C_total_tokens": median(differences), "paired_total_token_sign_p_value": p_value,
        "relative_total_token_saving": savings, "B_lower_cost_cases": shorter,
        "C_lower_cost_cases": longer, "equal_cost_cases": len(pairs) - shorter - longer,
        "efficiency_with_correctness": (
            b["grounded_calls"] >= c["grounded_calls"]
            and b["noncurrent_value_calls"] <= c["noncurrent_value_calls"]
            and b["invalid_decisions"] <= c["invalid_decisions"]
            and savings >= VERSIONED_CRITERIA["chronological_replay"]["minimum_relative_total_token_saving"]
            and p_value <= VERSIONED_CRITERIA["chronological_replay"]["alpha"]
        ),
    }


def analyze_versioned(replication_rows, updated_rows, control_rows, replication_count=12):
    criteria = VERSIONED_CRITERIA
    if replication_count not in criteria["replication_cases"]:
        raise ValueError("Versioned replication expects a frozen size of 12 or 16")
    replication = _pair_rows(replication_rows, "replication", {"A", "B"}, replication_count)
    updated = _pair_rows(updated_rows, "updated", {"A", "B", "C"}, criteria["updated_cases"])
    controls = _pair_rows(control_rows, "validation", {"A", "B"}, criteria["control_cases"])
    base_ids = [pair["A"].get("base_task_id", task) for section in (replication, updated, controls)
                for task, pair in section.items()]
    if len(set(base_ids)) != len(base_ids):
        raise ValueError("Versioned experimental partitions share a base task")

    groups = {"replication": _counts(replication, ("A", "B")), "updated": _counts(updated, ("A", "B", "C")),
              "controls": _counts(controls, ("A", "B"))}
    repeated = _paired_endpoint(replication, "B", "A", _requested_field_question)
    replaced = sum(_grounded_call(replication[task]["B"]) for task in repeated["only_right"])
    removed = len(repeated["only_right"])
    latest = _paired_endpoint(updated, "A", "B", _grounded_call)
    disallowed = [
        {"task_id": row["task_id"], "arm": row["arm"], "memory_ids": ids}
        for row in [*replication_rows, *updated_rows, *control_rows] if (ids := _scoped_ids(row))
    ]
    safeguards = {
        "control_correctness": (
            groups["controls"]["B"]["correct"] >= max(criteria["minimum_correct_controls"], groups["controls"]["A"]["correct"])
            and all(set(row["memory_rule_ids"]) == set(row["permitted_memory_ids"]) for row in control_rows)
        ),
        "unsupported_test_calls": all(groups[section]["B"]["unsupported_calls"] <= groups[section]["A"]["unsupported_calls"]
                                      for section in ("replication", "updated")),
        "invalid_test_decisions": all(groups[section]["B"]["invalid_decisions"] <= groups[section]["A"]["invalid_decisions"]
                                      for section in ("replication", "updated")),
        "noncurrent_value_use": all(groups[section]["B"]["noncurrent_value_calls"] == 0
                                    for section in ("replication", "updated", "controls")),
        "memory_scope": not disallowed,
    }
    primary_replication = (repeated["p_value"] <= criteria["replication"]["alpha"]
                           and repeated["difference"] >= criteria["replication"]["minimum_absolute_reduction"]
                           and removed > 0 and replaced / removed >= criteria["replication"]["minimum_grounded_call_replacement"])
    primary_updated = (latest["p_value"] <= criteria["updated"]["alpha"]
                       and latest["difference"] >= criteria["updated"]["minimum_absolute_increase"]
                       and groups["updated"]["B"]["grounded_calls"] >= criteria["updated"]["minimum_correct_latest_calls"])
    return {
        "groups": groups, "guardrails_passed": safeguards, "disallowed_memory": disallowed,
        "replication": {**repeated, "grounded_replacements": replaced,
                        "primary_passed": primary_replication, "confirmed_positive": primary_replication and all(safeguards.values())},
        "updated": {**latest, "primary_passed": primary_updated,
                    "confirmed_positive": primary_updated and all(safeguards.values())},
        "chronological_replay": _chronology_efficiency(updated, groups["updated"]),
        "both_confirmed_positive": primary_replication and primary_updated and all(safeguards.values()),
    }


def _draw_comparison(output, analysis):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = [("Safety controls", "controls", "correct")]
    if "replication" in analysis:
        specs = [
            ("Fresh-session repeat questions", "replication", "targeted_questions"),
            ("Correct calls after a saved update", "updated", "grounded_calls"), *specs,
        ]
    fig, axes = plt.subplots(1, len(specs), figsize=(4 * len(specs), 4.1), squeeze=False, layout="constrained")
    colors = {"A": "#366589", "B": "#13816f", "C": "#8c5919"}
    for axis, (title, group, field) in zip(axes[0], specs):
        rows = analysis["groups"][group]
        bars = axis.bar(list(rows), [100 * row[field] / row["cases"] for row in rows.values()],
                        color=[colors[name] for name in rows], width=0.5)
        axis.bar_label(bars, [f"{row[field]}/{row['cases']}" for row in rows.values()], padding=5)
        axis.set_ylim(0, 115)
        axis.set_yticks(range(0, 101, 25))
        axis.set_title(title)
        axis.set_ylabel("Share of cases (%)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.yaxis.grid(alpha=0.2)
    fig.savefig(output / "comparison.png", dpi=170)
    plt.close(fig)


def _markdown_report(analysis):
    groups = analysis["groups"]
    a, b = (groups["controls"][arm] for arm in ("A", "B"))
    lines = [
        "# ConfAgent 版本化事实记忆实验", "",
        "使用 Qwen3-8B 评估三个受控分区。A 组没有跨会话事实记忆。B 组根据已知用户和事实槽位选取最新有效记录。"
        "更新分区另设 C 组，按时间顺序提供同一槽位的真实历史记录，并交由模型判断当前有效的值。",
        "",
        f"安全对照中，A 正确 {a['correct']}/{a['cases']}，B 正确 {b['correct']}/{b['cases']}。"
        "对照包含撤回、用户不匹配、槽位不匹配和当前轮覆盖历史值。",
        "",
    ]
    if "replication" not in analysis:
        return lines + ["安全对照未达到预设要求，两个留出分区未运行。", "", "![安全对照](comparison.png)", ""]
    replication, updated = analysis["replication"], analysis["updated"]
    ar, br = (groups["replication"][arm] for arm in ("A", "B"))
    au, bu, cu = (groups["updated"][arm] for arm in ("A", "B", "C"))
    c = analysis["chronological_replay"]
    lines += [
        "| 结果 | A（无记忆） | B（最新有效值） | C（历史重放） |",
        "| --- | ---: | ---: | ---: |",
        f"| 复现组针对已保存字段的追问 | {ar['targeted_questions']}/{ar['cases']} | {br['targeted_questions']}/{br['cases']} | — |",
        f"| 更新组正确使用最新值 | {au['grounded_calls']}/{au['cases']} | {bu['grounded_calls']}/{bu['cases']} | {cu['grounded_calls']}/{cu['cases']} |",
        f"| 更新组使用无效版本 | {au['noncurrent_value_calls']} | {bu['noncurrent_value_calls']} | {cu['noncurrent_value_calls']} |",
        "",
        "## 预设主要终点", "",
        f"- 复现组追问净减少 {100 * replication['difference']:.1f} 个百分点，"
        f"单侧精确配对检验 p={replication['p_value']:.6g}。"
        f"{replication['grounded_replacements']}/{len(replication['only_right'])} 个减少的追问改为正确调用。"
        f"{'达到' if replication['confirmed_positive'] else '未达到'}该终点的预设判据。",
        f"- 更新组正确调用净增加 {100 * updated['difference']:.1f} 个百分点，"
        f"单侧精确配对检验 p={updated['p_value']:.6g}。"
        f"{'达到' if updated['confirmed_positive'] else '未达到'}该终点的预设判据。",
        "",
        "| 护栏 | 是否通过 |",
        "| --- | --- |",
    ]
    for key, label in (
        ("control_correctness", "安全对照正确性"), ("unsupported_test_calls", "无依据调用数"),
        ("invalid_test_decisions", "无效动作或截断数"), ("noncurrent_value_use", "无效版本调用"),
        ("memory_scope", "观察记录适用范围"),
    ):
        lines.append(f"| {label} | {'通过' if analysis['guardrails_passed'][key] else '未通过'} |")
    lines += [
        "", "## 历史重放与资源", "",
        f"在更新组中，B 与 C 的输入 token 差值为 {c['B_minus_C_prompt_tokens']:+,}，"
        f"总 token 差值为 {c['B_minus_C_total_tokens']:+,}。B 在 {c['B_lower_cost_cases']}/{au['cases']} 个配对中开销更低。"
        f"单侧配对符号检验 p={c['paired_total_token_sign_p_value']:.6g}。"
        f"在观测到的正确调用数和无效动作护栏下，{'达到' if c['efficiency_with_correctness'] else '未达到'}预设的次要效率判据。",
        "",
        "| 分区与组别 | 请求数 | 输入 token | 输出 token | 总 token |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for section, label in (("replication", "复现"), ("updated", "更新"), ("controls", "安全对照")):
        for arm, row in groups[section].items():
            cost = row["cost"]
            lines.append(f"| {label} {arm} | {cost['requests']:,} | {cost['prompt_tokens']:,} |"
                         f" {cost['completion_tokens']:,} | {cost['total_tokens']:,} |")
    lines += [
        "", "结果适用于已知用户、已知事实槽位以及按规则保存的模拟事实。A 与 B 的更新组对比衡量能否使用已解析的版本化事实，"
        "B 与 C 的对比衡量预先解析记录与让模型读取时间线的表现。", "",
        "![版本化事实复用对照](comparison.png)", "",
    ]
    return lines


def write_versioned_report(output, protocol, replication_rows, updated_rows, control_rows, failures):
    if bool(replication_rows) != bool(updated_rows):
        raise ValueError("Both held-out partitions are required to produce the versioned summary")
    if replication_rows:
        analysis = analyze_versioned(replication_rows, updated_rows, control_rows, protocol["counts"]["replication"])
        status = "completed"
    else:
        controls = _pair_rows(control_rows, "validation", {"A", "B"}, VERSIONED_CRITERIA["control_cases"])
        analysis = {"groups": {"controls": _counts(controls, ("A", "B"))}, "both_confirmed_positive": False}
        status = "control_failed"
    write_json(output / "summary.json", {
        "status": status, "experiment": protocol["experiment"], "criteria": VERSIONED_CRITERIA,
        "analysis": analysis, "failed_requests": failures,
    })
    _draw_comparison(output, analysis)
    lines = _markdown_report(analysis)
    if failures:
        lines += [f"另记录 {sum(failures.values())} 次请求异常。", ""]
    (output / "报告.md").write_text("\n".join(lines), encoding="utf-8")
