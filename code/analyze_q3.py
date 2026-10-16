"""Recompute Q3 outcomes from retained trajectories, with paired task bootstrap."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from q3_tasks import evaluate_trajectory


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def paired_comparison(baseline, condition):
    """Task is the resampling cluster; deterministic seeds are not 72 IID tasks."""
    if baseline.keys() != condition.keys():
        raise ValueError("Missing paired task/seed outcomes")
    tasks = sorted({task for task, _ in baseline})
    deltas = np.array([np.mean([int(condition[key]) - int(baseline[key])
                               for key in baseline if key[0] == task]) for task in tasks])
    rng = np.random.default_rng(20261016)
    sampled = deltas[rng.integers(0, len(deltas), size=(10000, len(deltas)))].mean(axis=1)
    return {"delta_pp": float(deltas.mean() * 100),
            "task_cluster_bootstrap_95ci_pp": [float(x * 100) for x in np.quantile(sampled, [.025, .975])],
            "task_clusters": len(tasks), "wins": sum(condition[k] and not baseline[k] for k in baseline),
            "losses": sum(baseline[k] and not condition[k] for k in baseline),
            "ties": sum(baseline[k] == condition[k] for k in baseline)}


def summarize(rows):
    n = len(rows)
    labels = [r["independent_evaluation"] for r in rows]
    actions = sum(v["action_correct"] + v["action_errors"] for v in labels)
    return {"episodes": n, "strict_successes": sum(v["strict_success"] for v in labels),
            "strict_success_rate": sum(v["strict_success"] for v in labels) / n,
            "clean_successes": sum(v["clean_success"] for v in labels),
            "recovered_successes": sum(v["recovered_success"] for v in labels),
            "substantive_actions": actions, "action_errors": sum(v["action_errors"] for v in labels),
            "action_error_rate": sum(v["action_errors"] for v in labels) / actions if actions else None,
            "pr1_final_successes": sum(r.get("final_success") is True for r in rows),
            "pr1_false_successes": sum(r.get("final_success") is True and not v["strict_success"]
                                       for r, v in zip(rows, labels)),
            "retrieved_episodes": sum(any(s.get("memory_ids") for s in r["steps"]) for r in rows),
            "logical_model_calls": sum(r["total_model_calls"] for r in rows),
            "logical_tokens": sum(r["total_tokens"] for r in rows),
            "mean_model_calls": sum(r["total_model_calls"] for r in rows) / n,
            "mean_tokens": sum(r["total_tokens"] for r in rows) / n,
            "model_error_episodes": sum(str(r["config"].get("stop_reason", "")).startswith("model_error") for r in rows)}


def analyze(folder, refresh_evaluator=False):
    folder = Path(folder)
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    suite = json.loads((folder / "suite.json").read_text(encoding="utf-8"))
    run = json.loads((folder / "run_summary.json").read_text(encoding="utf-8"))
    revision_path = folder / "measurement_revision.json"
    evaluator_hash = hashlib.sha256(Path(__file__).with_name("q3_tasks.py").read_bytes()).hexdigest()
    revision = json.loads(revision_path.read_text(encoding="utf-8")) if revision_path.exists() else None
    if refresh_evaluator:
        if revision is None:
            revision = {"reason": "Fix false negatives for explicit unit/field-before-value final answers; no new model calls",
                        "initial_evaluator_sha256": protocol["code_sha256"]["q3_tasks.py"],
                        "initial_metrics": json.loads((folder / "metrics.json").read_text(encoding="utf-8"))}
        revision["current_evaluator_sha256"] = evaluator_hash
        revision_path.write_text(json.dumps(revision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not run.get("complete"):
        raise ValueError("Cannot analyze an incomplete run as completed evidence")
    tasks = {t["_q3"]["id"]: t for t in suite["test"]}
    expected = {(task, seed) for task in protocol["split_ids"]["test"] for seed in protocol["seeds"]}
    all_rows, outcomes, summary = {}, {}, {}
    for name in run["conditions"]:
        rows = read_rows(folder / "evaluation" / (name + ".jsonl"))
        keys = [(r["task_id"], r["seed"]) for r in rows]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError("Duplicated or missing episodes in " + name)
        for row in rows:
            recomputed = evaluate_trajectory(tasks[row["task_id"]], row)
            if recomputed != row["independent_evaluation"]:
                if not revision or revision["current_evaluator_sha256"] != evaluator_hash:
                    raise ValueError("Stored oracle evidence changed: " + row["task_id"])
                # Preserve raw trajectories and their initial labels on disk.
                row["independent_evaluation"] = recomputed
        all_rows[name] = rows
        outcomes[name] = {(r["task_id"], r["seed"]): r["independent_evaluation"]["strict_success"] for r in rows}
        summary[name] = summarize(rows)
        summary[name]["by_family"] = {family: summarize([r for r in rows if r["q3_family"] == family])
                                       for family in sorted({r["q3_family"] for r in rows})}
        summary[name]["successes_by_seed"] = {str(seed): sum(r["independent_evaluation"]["strict_success"]
            for r in rows if r["seed"] == seed) for seed in protocol["seeds"]}
    comparisons = {name: paired_comparison(outcomes["no_memory"], values)
                   for name, values in outcomes.items() if name != "no_memory"}
    round_stats = []
    for stats in run["rounds"]:
        updated = dict(stats)
        source_tasks = {t["_q3"]["id"]: t for t in suite["construction"]}
        episodes = read_rows(folder / ("round_%d" % stats["round"]) / "construction.jsonl")
        updated["construction_strict_success"] = sum(evaluate_trajectory(source_tasks[r["task_id"]], r)["strict_success"] for r in episodes)
        round_stats.append(updated)
    results = {"protocol_version": protocol["version"], "reported_final_round": protocol["reported_final_round"],
               "conditions": summary, "paired_vs_no_memory": comparisons, "construction_rounds": round_stats,
               "measurement_revision": "measurement_revision.json" if revision else None,
               "actual_model_cost": run["actual_model_cost"],
               "cache_hits": run["deterministic_cache_hits_this_process"],
               "interpretation": "Synthetic within-family instance holdout. Temperature zero; seeds are decoding checks, "
                 "not independent data replicates. Task-cluster bootstrap is exploratory. Logical condition costs "
                 "include reused deterministic responses; actual cost is separately recorded."}
    (folder / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (folder / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [key for key, value in summary["no_memory"].items() if not isinstance(value, dict)]
        writer = csv.DictWriter(stream, fieldnames=["condition"] + fields)
        writer.writeheader()
        for name, values in summary.items():
            writer.writerow({"condition": name, **{key: values[key] for key in fields}})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.8))
    names = list(summary)
    ax.bar(names, [summary[name]["strict_success_rate"] * 100 for name in names], color="#3B7EA1")
    ax.set(ylabel="Strict goal completion (%)", ylim=(0, 105), title="Local Qwen3.5-0.8B Q4_K_M / synthetic instance holdout")
    ax.tick_params(axis="x", labelrotation=15)
    for i, name in enumerate(names):
        ax.text(i, summary[name]["strict_success_rate"] * 100 + 1,
                f'{summary[name]["strict_successes"]}/{summary[name]["episodes"]}', ha="center")
    fig.tight_layout()
    fig.savefig(folder / "strict_success.png", dpi=180)
    fig.savefig(folder / "strict_success.svg")
    svg = folder / "strict_success.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)
    print(json.dumps({"stage": "analysis", "metrics": str(folder / "metrics.json")}), flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--refresh-evaluator", action="store_true")
    args = parser.parse_args()
    analyze(args.folder, args.refresh_evaluator)
