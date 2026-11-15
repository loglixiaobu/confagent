"""Predeclared comparisons for versioned memory carrying authorized user evidence."""
from __future__ import annotations

from copy import deepcopy

from collect_new_signals import write_json
from report_memory_factual import _grounded_call
from report_memory_versioned import (
    VERSIONED_CRITERIA, _counts, _draw_comparison, _pair_rows, _paired_endpoint, _scoped_ids, analyze_versioned,
)


PROVENANCE_CRITERIA = deepcopy(VERSIONED_CRITERIA)
PROVENANCE_CRITERIA.pop("chronological_replay")
PROVENANCE_CRITERIA["replication_cases"] = 16
for endpoint in ("replication", "updated"):
    PROVENANCE_CRITERIA[endpoint]["alpha"] = 0.0015625
PROVENANCE_CRITERIA["provenance"] = {
    "status": "exploratory",
    "comparison": "B active value with an exact scope-match statement and verbatim user authorization versus C bare active value",
    "endpoint": "exact grounded calls pooled across 16 recall and 20 updated cases",
    "test": "exact one-sided paired binomial (McNemar)",
    "safeguard": "zero use of stale or forbidden values in B across test and controls",
}


def analyze_provenance(replication_rows, updated_rows, control_rows):
    replication = _pair_rows(replication_rows, "replication", {"A", "B", "C"}, PROVENANCE_CRITERIA["replication_cases"])
    updated = _pair_rows(updated_rows, "updated", {"A", "B", "C"}, PROVENANCE_CRITERIA["updated_cases"])
    result = analyze_versioned(
        [row for row in replication_rows if row["arm"] != "C"], updated_rows, control_rows,
        PROVENANCE_CRITERIA["replication_cases"],
    )
    result["groups"]["replication"] = _counts(replication, ("A", "B", "C"))
    result.pop("chronological_replay")
    disallowed_c = [
        {"task_id": row["task_id"], "arm": "C", "memory_ids": ids}
        for row in replication_rows if row["arm"] == "C" and (ids := _scoped_ids(row))
    ]
    result["disallowed_memory"].extend(disallowed_c)
    result["guardrails_passed"]["memory_scope"] = not result["disallowed_memory"]
    for key in ("replication", "updated"):
        endpoint = result[key]
        endpoint["primary_passed"] &= endpoint["p_value"] <= PROVENANCE_CRITERIA[key]["alpha"]
        endpoint["confirmed_positive"] = endpoint["primary_passed"] and all(result["guardrails_passed"].values())
    result["both_confirmed_positive"] = all(result[key]["confirmed_positive"] for key in ("replication", "updated"))

    pooled = {**replication, **updated}
    if len(pooled) != PROVENANCE_CRITERIA["replication_cases"] + PROVENANCE_CRITERIA["updated_cases"]:
        raise ValueError("Provenance test partitions share a case")
    comparison = _paired_endpoint(pooled, "C", "B", _grounded_call)
    groups = result["groups"]
    comparison["B_grounded_calls"] = sum(groups[stage]["B"]["grounded_calls"] for stage in ("replication", "updated"))
    comparison["C_grounded_calls"] = sum(groups[stage]["C"]["grounded_calls"] for stage in ("replication", "updated"))
    comparison["B_stale_or_forbidden_calls"] = sum(
        groups[stage]["B"]["noncurrent_value_calls"] for stage in ("replication", "updated", "controls")
    )
    comparison["C_stale_or_forbidden_calls"] = sum(
        groups[stage]["C"]["noncurrent_value_calls"] for stage in ("replication", "updated")
    )
    comparison["safeguard_passed"] = comparison["B_stale_or_forbidden_calls"] == 0
    result["provenance"] = comparison
    return result


def _provenance_markdown(analysis):
    groups = analysis["groups"]
    controls = groups["controls"]
    lines = [
        "# ConfAgent 带来源依据的版本化记忆实验", "",
        "所有组收到相同的当前请求和工具说明。A 无跨会话记忆。B 注入最新有效事实，并附上该事实与当前用户、"
        "所需资料的精确匹配说明，以及授权保存该事实的用户原话。"
        "C 注入同一个最新版的字段和值。三组共用模型、推理和证据边界设置。", "",
        f"安全对照共 8 件，A 正确 {controls['A']['correct']}/8，B 正确 {controls['B']['correct']}/8。"
        "对照覆盖撤回、用户不匹配、槽位不匹配和当前轮覆盖历史值。", "",
    ]
    if "replication" not in analysis:
        return lines + ["对照未达到预设门槛，留出分区未运行。", "", "![安全对照](comparison.png)", ""]
    replication = groups["replication"]
    updated = groups["updated"]
    lines += [
        "| 分区和指标 | A 无记忆 | B 带来源依据 | C 仅字段和值 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, group, field in (
        ("复现组追问已保存的字段", replication, "targeted_questions"),
        ("复现组正确调用", replication, "grounded_calls"),
        ("更新组正确使用最新值", updated, "grounded_calls"),
        ("更新组使用无效版本", updated, "noncurrent_value_calls"),
    ):
        values = [f"{group[arm][field]}/{group[arm]['cases']}" for arm in ("A", "B", "C")]
        lines.append(f"| {label} | {' | '.join(values)} |")
    recall, current, source = (analysis[key] for key in ("replication", "updated", "provenance"))
    lines += [
        "", "## 预设主要终点", "",
        "两个 A/B 终点共用无依据调用、无效或截断动作、过期或越权值以及对照正确性护栏。"
        "各自的单侧精确配对检验门槛为 p ≤ 0.0015625。", "",
        f"- 复现组减少追问 {100 * recall['difference']:.1f} 个百分点，p={recall['p_value']:.6g}。"
        f"少追问的 {len(recall['only_right'])} 件中，有 {recall['grounded_replacements']} 件变为正确调用。"
        f"{'达到' if recall['confirmed_positive'] else '未达到'}预设判据。",
        f"- 更新组正确调用增加 {100 * current['difference']:.1f} 个百分点，p={current['p_value']:.6g}。"
        f"{'达到' if current['confirmed_positive'] else '未达到'}预设判据。",
        "", "| 护栏 | 是否通过 |", "| --- | --- |",
    ]
    for key, label in (
        ("control_correctness", "安全对照正确性"), ("unsupported_test_calls", "无依据调用数"),
        ("invalid_test_decisions", "无效动作或截断数"), ("noncurrent_value_use", "无效版本调用"),
        ("memory_scope", "观察记录适用范围"),
    ):
        lines.append(f"| {label} | {'通过' if analysis['guardrails_passed'][key] else '未通过'} |")
    lines += [
        "", "## 来源关联记忆的探索性比较", "",
        f"合并两个留出分区，B 正确调用 {source['B_grounded_calls']}/36，C 正确调用 {source['C_grounded_calls']}/36。"
        f"B 独有正确 {len(source['only_right'])} 件，C 独有正确 {len(source['only_left'])} 件，"
        f"单侧精确配对检验 p={source['p_value']:.6g}。",
        f"B 在 36 个测试和 8 个对照中使用无效事实 {source['B_stale_or_forbidden_calls']} 次。"
        f"C 在 36 个测试中使用无效事实 {source['C_stale_or_forbidden_calls']} 次。",
        "",
        "| 分区与组别 | 请求数 | 输入 token | 输出 token | 总 token |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for stage, label in (("replication", "复现"), ("updated", "更新"), ("controls", "安全对照")):
        for arm, row in groups[stage].items():
            cost = row["cost"]
            lines.append(f"| {label} {arm} | {cost['requests']:,} | {cost['prompt_tokens']:,} | "
                         f"{cost['completion_tokens']:,} | {cost['total_tokens']:,} |")
    lines += [
        "", "结果适用于已知用户和槽位，以及已有明确授权的受控跨会话事实。评测范围为当前请求的下一步动作。", "",
        "![来源关联记忆对照](comparison.png)", "",
    ]
    return lines


def write_provenance_report(output, protocol, replication_rows, updated_rows, control_rows, failures):
    if bool(replication_rows) != bool(updated_rows):
        raise ValueError("Both held-out partitions are required")
    if replication_rows:
        analysis = analyze_provenance(replication_rows, updated_rows, control_rows)
        status = "completed"
    else:
        controls = _pair_rows(control_rows, "validation", {"A", "B"}, PROVENANCE_CRITERIA["control_cases"])
        analysis = {"groups": {"controls": _counts(controls, ("A", "B"))}, "both_confirmed_positive": False}
        status = "control_failed"
    write_json(output / "summary.json", {
        "status": status, "experiment": protocol["experiment"], "criteria": PROVENANCE_CRITERIA,
        "analysis": analysis, "failed_requests": failures,
    })
    _draw_comparison(output, analysis)
    lines = _provenance_markdown(analysis)
    if failures:
        lines += [f"另记录 {sum(failures.values())} 次请求异常。", ""]
    (output / "报告.md").write_text("\n".join(lines), encoding="utf-8")
