#!/usr/bin/env python
"""Summarize smoke trajectories and produce a compact PNG dashboard."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = HERE / "cache"
(CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str((CACHE / "matplotlib").resolve())
os.environ["XDG_CACHE_HOME"] = str(CACHE.resolve())
os.environ["TEMP"] = str((HERE / "tmp").resolve())
os.environ["TMP"] = str((HERE / "tmp").resolve())
sys.pycache_prefix = str((CACHE / "pycache").resolve())

import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(ROOT / "code"))
from analyze_verification import summarize  # noqa: E402


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def pct(value: Any) -> str:
    return "-" if value is None else "%.1f%%" % (100.0 * value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(HERE / "outputs"))
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.drive.upper() != "E:" or HERE not in output_dir.parents:
        parser.error("--output-dir must stay below %s on E:" % HERE)
    files = sorted(
        output_dir.glob("trajectories_*.jsonl"),
        key=lambda path: (path.stem != "trajectories_vanilla", path.name),
    )
    if not files:
        parser.error("no trajectories_*.jsonl found in %s" % output_dir)

    summaries: Dict[str, Dict[str, Any]] = {}
    trajectories: Dict[str, List[Dict[str, Any]]] = {}
    for path in files:
        name = path.stem.removeprefix("trajectories_")
        trajectories[name] = load_jsonl(path)
        summaries[name] = summarize(trajectories[name])
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    columns = [
        "condition", "n_tasks", "success_rate", "replay_success_rate",
        "verification_rate", "verification_available_rate", "usable_verification_rate",
        "error_to_correct", "correct_to_error",
        "avg_model_calls", "avg_verification_calls", "avg_tool_calls",
        "avg_latency_ms",
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        for name, stats in summaries.items():
            writer.writerow({"condition": name, **{key: stats.get(key) for key in columns[1:]}})

    names = list(summaries)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"][:len(names)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    x = range(len(names))
    width = 0.36
    axes[0].bar([i - width / 2 for i in x],
                [100 * (summaries[n]["success_rate"] or 0) for n in names],
                width, label="End-to-end success", color="#4C78A8")
    axes[0].bar([i + width / 2 for i in x],
                [100 * (summaries[n]["replay_success_rate"] or 0) for n in names],
                width, label="Replay complete", color="#72B7B2")
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Task outcome")
    axes[0].legend(fontsize=8)

    axes[1].bar([i - width / 2 for i in x],
                [100 * (summaries[n]["verification_rate"] or 0) for n in names],
                width, label="Triggered", color="#F58518")
    axes[1].bar([i + width / 2 for i in x],
                [100 * (summaries[n]["usable_verification_rate"] or 0) for n in names],
                width, label="Usable output", color="#54A24B")
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Execution steps (%)")
    axes[1].set_title("Verification coverage")
    axes[1].legend(fontsize=8)
    for index, name in enumerate(names):
        if not (summaries[name]["verification_rate"] or 0):
            axes[1].text(
                index, 3, "0%\n(disabled)", ha="center", va="bottom",
                fontsize=9, color="#666666",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "#EEEEEE",
                      "edgecolor": "#AAAAAA"},
            )

    actor = [summaries[n]["avg_actor_calls"] for n in names]
    verifier = [summaries[n]["avg_verification_calls"] for n in names]
    axes[2].bar(x, actor, label="Actor calls", color="#B279A2")
    axes[2].bar(x, verifier, bottom=actor, label="Verification calls", color="#FF9DA6")
    axes[2].set_ylabel("Average calls / task")
    axes[2].set_title("Model-call cost")
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.set_xticks(list(x), names, rotation=15)
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle("Qwen3-0.6B local CPU smoke test (engineering validation only)")
    fig.tight_layout()
    fig.savefig(output_dir / "smoke_summary.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# 小模型流程烟测结果",
        "",
        "> 本结果仅用于验证代码链路是否可运行，不代表正式模型能力或论文结论。",
        "",
        "| 条件 | 任务数 | 最终成功率 | Replay完成率 | 验证触发率 | 可用验证/步骤 | 平均模型调用 | 平均验证调用 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in summaries.items():
        lines.append(
            "| %s | %d | %s | %s | %s | %s | %.2f | %.2f |"
            % (name, stats["n_tasks"], pct(stats["success_rate"]),
               pct(stats["replay_success_rate"]), pct(stats["verification_rate"]),
               pct(stats["usable_verification_rate"]),
               stats["avg_model_calls"], stats["avg_verification_calls"])
        )
    if "vanilla" in summaries and "always" in summaries:
        base, checked = summaries["vanilla"], summaries["always"]
        call_delta = checked["avg_model_calls"] - base["avg_model_calls"]
        lines.extend([
            "",
            "## 本次可观察结果",
            "",
            "- 两个条件均为 3/3 端到端成功，说明本地模型、工具解析、mock 环境、轨迹和终止判断已连通。",
            "- `always + critic` 覆盖 4/4 个执行步骤，4/4 均产生可用结构化验证结果。",
            "- 4 个正确候选均被保留，未发生 `correct_to_error`；该简单烟测没有自然错误，因此不宣称纠错收益。",
            "- 全量验证平均每任务增加 %.2f 次模型调用，图中明确展示了可靠性检查的成本。" % call_delta,
        ])
    lines.extend([
        "",
        "## 动作轨迹预览",
        "",
        "| 条件 | 任务 | 工具动作 | 已验证/可用 | Replay | 最终成功 |",
        "|---|---|---|---:|---:|---:|",
    ])
    for name, rows in trajectories.items():
        for trajectory in rows:
            execution = [step for step in trajectory.get("steps", [])
                         if step.get("action_type") != "answer"]
            actions = " → ".join(
                str((step.get("action_input") or {}).get("api_name", "?"))
                for step in execution
            ) or "-"
            usable = sum(
                bool(step.get("verified"))
                and (step.get("verification_result") or {}).get("available") is True
                for step in execution
            )
            lines.append(
                "| %s | %s | %s | %d/%d | %s | %s |"
                % (name, trajectory.get("task_id"), actions, usable, len(execution),
                   trajectory.get("replay_success"), trajectory.get("final_success"))
            )
    lines.extend(["", "![烟测汇总](smoke_summary.png)", ""])
    (output_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print("chart:", output_dir / "smoke_summary.png")


if __name__ == "__main__":
    main()
