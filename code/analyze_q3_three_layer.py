"""Recompute Q3 results on CPU; optionally verify local runtime artifacts."""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from q3_three_layer.backend import digest, file_digest, write_json
from q3_three_layer.learning import RidgePolicy, build_memory, features
from q3_three_layer.run import score
from q3_three_layer.tasks import TaskWorld, independent_correct


def analyze(directory, check_runtime=False):
    root = Path(directory)
    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))
    suite, selection, rows = read("suite.json"), read("selection_frozen.json"), read("evaluation.json")
    protocol, training = read("protocol.json"), read("training.json")
    heldout = suite["source_test"] + suite["chain_test"] + suite["qa_test"]
    task_by_id = {t["task_id"]: t for t in heldout}
    labels = {f"candidate_M{m}P{p}L{l}" for m in (0, 1) for p in (0, 1) for l in (0, 1)}
    labels |= {"accepted", "always_verify", "visible_conversion_rule"}
    assert len(rows) == 198 and {r["condition"] for r in rows} == labels
    assert digest(suite) == protocol["suite_sha256"]
    assert digest(protocol) == selection["protocol_sha256"]
    if check_runtime:
        for name, checksum in selection["implementation_hashes"].items():
            assert file_digest(Path(__file__).resolve().parent / "q3_three_layer" / name) == checksum
    assert len({(r["condition"], r["task_id"]) for r in rows}) == 198
    for label in labels:
        assert {r["task_id"] for r in rows if r["condition"] == label} == set(task_by_id)
    for row in rows:
        task = task_by_id[row["task_id"]]
        assert row["success"] is independent_correct(task, row["prediction"])
        world = TaskWorld(task)
        world.prepare()
        world.execute(row["prediction"])
        assert world.history == row["history"]
        assert world.evaluate() == row["assessment"]
        expected_cost = {"keep": (0, 0), "memory": (0, 1), "verify": (1, 0)}[row["action"]]
        assert expected_cost == (row["verifier_calls"], row["extra_model_calls"])
        assert math.isclose(row["utility"], float(row["success"]) - .1 * expected_cost[0] - .02 * expected_cost[1])
        assert row["assessment"]["overwrites"] == 0
        if task["shape"] == "read_write":
            assert sum(e["kind"] == "read" for e in row["history"]) == 1
        if row["action"] == "memory":
            assert row["M"] and row["memory_id"]
    source = read("source_experiences.json")
    assert len(source) == 18
    construction_ids = {t["task_id"] for t in suite["construction"]}
    assert {r["task_id"] for r in source} == construction_ids
    assert construction_ids.isdisjoint(task_by_id)
    bank = read("candidate_memory.json")
    assert build_memory(source) == bank and digest(bank) == selection["versions"]["M_c"]
    for row in source:
        task = next(t for t in suite["construction"] if t["task_id"] == row["task_id"])
        assert row["certified"] is True and independent_correct(task, row["prediction"])
        assert row["origin"] in ("tool_assisted_repair", "model_first_pass_success", "model_self_repair")
        if row["origin"] == "tool_assisted_repair":
            assert row["first"]["success"] is False and row["final"]["success"] is True
        else:
            assert row["first"]["success"] is True
    table = read("policy_training_rows.json")
    assert {r["task_id"] for r in table} <= construction_ids
    policy = RidgePolicy.from_dict(read("policy.json"))
    fitted = RidgePolicy.fit(table)
    assert digest(policy.to_dict()) == selection["versions"]["P_c"]
    for action in policy.coefficients:
        assert np.allclose(policy.coefficients[action], fitted.coefficients[action], rtol=1e-10, atol=1e-12)
    by_source = {r["task_id"]: r for r in source}
    for row in table:
        task = next(t for t in suite["construction"] if t["task_id"] == row["task_id"])
        contract = by_source[row["task_id"]]["contract"]
        assert row["features"] == features(contract, row["initial_draft"], row["memory_available"],
                                             budget=1, adapter_active=row["actor"] == "adapter")
        assert row["success"] is independent_correct(task, row["prediction"])
    steps = [json.loads(s) for s in (root / "training_steps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(steps) == 12 and [s["step"] for s in steps] == list(range(1, 13))
    assert all(math.isfinite(s["loss"]) and s["gradient_norm"] > 0 for s in steps)
    assert [s["task_id"] for s in steps] == training["actual_optimized_ids"]
    assert set(training["eligible_pool_task_ids"]) <= construction_ids
    assert set(training["actual_optimized_ids"]) <= construction_ids
    assert training["adapter_changed"] and training["frozen_unchanged"] and training["adapter_delta_l2"] > 0
    assert training["frozen_before"] == training["frozen_after"]
    assert training["adapter_after"] == selection["versions"]["L_c"]
    if check_runtime:
        for name, checksum in training["adapter_files"].items():
            assert file_digest(root / "adapter" / name) == checksum
    reload = read("reload_check.json")
    assert reload["pass"] and reload["same_greedy_tokens"] and reload["exact_tensor_roundtrip"]
    assert reload["original_state_hash"] == reload["restored_state_hash"]
    summaries = {label: {"all": score([r for r in rows if r["condition"] == label]),
        **{shape: score([r for r in rows if r["condition"] == label and r["shape"] == shape])
            for shape in ("single_write", "read_write", "text_qa")}} for label in sorted(labels)}
    baseline = {r["task_id"]: r for r in rows if r["condition"] == "candidate_M0P0L0"}
    for label in summaries:
        current = [r for r in rows if r["condition"] == label]
        summaries[label]["F"] = sum(not baseline[r["task_id"]]["success"] and r["success"] for r in current)
        summaries[label]["H"] = sum(baseline[r["task_id"]]["success"] and not r["success"] for r in current)
    # Published snapshots include result data, not weights or raw runtime logs.
    actual = None
    cost_source = None
    if check_runtime:
        requests = [json.loads(s) for s in (root / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
        computational = [json.loads(s) for s in (root / "computational_calls.jsonl").read_text(encoding="utf-8").splitlines()]
        environments = [json.loads(s) for s in (root / "environment_calls.jsonl").read_text(encoding="utf-8").splitlines()]
        collected = Counter(r["task_id"] for r in computational if r["stage"] == "construction_check")
        logged = Counter(r["task_id"] for r in environments if r["stage"] == "construction_collection")
        unlogged = sum((collected[r["task_id"]] - logged[r["task_id"]]) * r["final"]["environment_calls"] for r in source)
        assert unlogged >= 0
        for row in rows:
            assert any(r["key"] == row["generation_key"] for r in requests)
        actual = {"model_forwards": len(requests), "unique_generation_keys": len({r["key"] for r in requests}),
            "prompt_tokens": sum(r["prompt_tokens"] for r in requests),
            "completion_tokens": sum(r["completion_tokens"] for r in requests),
            "model_latency_s": sum(r["latency_s"] for r in requests),
            "computational_calls": len(computational),
            "environment_calls_logged": sum(r["count"] for r in environments),
            "source_only_initial_environment_calls_unlogged": unlogged,
            "environment_calls_total": sum(r["count"] for r in environments) + unlogged,
            "source_origins": dict(Counter(r["origin"] for r in source))}
        cost_source = "recomputed from local runtime logs"
    elif (root / "experiment.json").exists():
        actual = read("experiment.json")["actual_costs"]
        cost_source = "recorded experiment totals; raw logs are local only"
    report = {"validation_scope": "results_and_runtime" if check_runtime else "results_only",
              "rows": len(rows), "conditions": summaries, "actual_costs": actual,
              "cost_source": cost_source,
              "selected": selection["selected"], "training": training,
              "cost_note": "Recorded totals include collection, training checks and interrupted execution, not just one deployment."}
    write_json(root / "metrics.json", report)
    with (root / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["condition", "source_success", "chain_success", "qa_success", "all_success", "utility", "verifier_calls", "extra_model_calls", "F", "H"])
        for label, summary in summaries.items():
            writer.writerow([label, *(summary[s]["successes"] for s in ("single_write", "read_write", "text_qa")),
                summary["all"]["successes"], summary["all"]["mean_utility"], summary["all"]["verifier_calls"],
                summary["all"]["extra_model_calls"], summary["F"], summary["H"]])
    print(json.dumps({"validation_scope": report["validation_scope"], "rows": len(rows),
                      "selected": selection["selected"], "cost_source": cost_source}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--check-runtime", action="store_true",
                        help="Also require matching source code, raw request logs and adapter files")
    args = parser.parse_args()
    analyze(args.directory, check_runtime=args.check_runtime)
