#!/usr/bin/env python3
"""Test combining separately authorized facts in a single API call."""
from __future__ import annotations

import argparse
import fcntl
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from threading import Lock

import httpx
from openai import OpenAI

from collect_new_signals import digest, sha, write_json
from memory_agent import QwenPromptCounter, matching_rules
from memory_experiment_data import case_tools
from report_memory_multislot import MULTISLOT_CRITERIA, write_multislot_report
from run_memory_experiment import MODEL_PATH, ROOT, collect_decision, plan_pairs, require_frozen
from run_memory_versioned import compile_facts, used_stale_or_forbidden_value
from run_memory_versioned_boundaries import FramedRequests, SETTINGS, SHARED_BOUNDARY, framed
from run_memory_versioned_provenance import source_linked_rules

OUTPUT = ROOT / "outputs/profile_fact_memory_v9"
GROUPS = {"A": "M0", "B": "separately authorized source-linked facts"}


def scoped_rules(case, timelines):
    """Resolve each requested field against its own active authorized timeline."""
    tool_specs = case_tools(case)
    bindings = []
    for field, slot in case["memory_fields"].items():
        events = timelines.get((case["profile_id"], slot), [])
        if events:
            scoped = {**case, "memory_field": field}
            candidates = source_linked_rules(scoped, events[-1])
            bindings.extend((field, rule) for rule in matching_rules(candidates, tool_specs))
    return bindings


def presented_case(case, visible_memory_fields):
    """Apply the same evidence rule to both arms before scoring a next action."""
    sources = {
        field: {"source": "missing"} if source["source"] == "history" and field not in visible_memory_fields else source
        for field, source in case["parameter_sources"].items()
    }
    missing = [field for field, source in sources.items() if source["source"] == "missing"]
    if missing:
        return {**case, "parameter_sources": sources, "expected": {"kind": "clarify", "fields": missing}}
    return {**case, "parameter_sources": sources}


def used_disallowed_fact(case, action):
    return any(used_stale_or_forbidden_value({**case, "memory_field": field}, action)
               for field in case["memory_fields"])


def collect_stage(cases, plan, requests, counter, timelines, replica_count):
    by_id = {case["task_id"]: case for case in cases}
    count_lock = Lock()

    def count_tokens(payload):
        with count_lock:
            return counter(framed(payload))

    def on_replica(replica):
        rows = []
        for index, scheduled in enumerate(plan):
            if scheduled["replica_index"] != replica:
                continue
            case = by_id[scheduled["task_id"]]
            approved = scoped_rules(case, timelines)
            dataset = scheduled["dataset"]
            for arm in scheduled["arms"]:
                bindings = approved if arm == "B" else []
                visible = [field for field, _ in bindings]
                row = collect_decision(
                    presented_case(case, visible), requests, count_tokens, SETTINGS,
                    f"{dataset}:{case['task_id']}:{arm}", [rule for _, rule in bindings], replica,
                )
                rows.append((index, {
                    **row, "dataset": dataset, "arm": arm, "replica_index": replica,
                    "condition": case["condition"], "memory_fields": case["memory_fields"],
                    "visible_memory_fields": visible,
                    "permitted_memory_ids": case["permitted_memory_ids"] if arm == "B" else [],
                    "stale_or_forbidden_value_used": used_disallowed_fact(case, row["action"]),
                }))
            print(f"{dataset} {case['task_id']} ({index + 1}/{len(plan)}, GPU {replica})", flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=replica_count) as executor:
        rows = [item for batch in executor.map(on_replica, range(replica_count)) for item in batch]
    return [row for _, row in sorted(rows, key=lambda item: item[0])]


def controls_pass(rows):
    scores = {arm: sum(r["evaluation"]["correct"] for r in rows if r["arm"] == arm) for arm in ("A", "B")}
    return (scores["B"] >= max(scores["A"], MULTISLOT_CRITERIA["minimum_correct_controls"])
            and all(set(row["permitted_memory_ids"]) == set(row["memory_rule_ids"]) for row in rows)
            and not any(row["stale_or_forbidden_value_used"] for row in rows if row["arm"] == "B"))


def run(args):
    from memory_multislot_data import MULTISLOT_PRIOR_CASE_FILES, MULTISLOT_SCHEMA_CORPORA, load_multislot_cases

    splits = load_multislot_cases(args.cases)
    counts = {label: len(cases) for label, cases in splits.items()}
    if counts != {"validation": 8, "test": 24}:
        raise ValueError(f"Unexpected multislot split sizes: {counts}")
    endpoints = [args.base_url, *args.replica_base_url]
    if len(endpoints) > 2 or len(endpoints) != len(set(endpoints)):
        raise ValueError("Supply one or two distinct replicas")
    timelines = compile_facts(splits)
    memory = {"version": "M9", "facts": [records[-1] for records in timelines.values() if records]}
    controls_plan = plan_pairs({"validation": splits["validation"]}, len(endpoints))
    test_plan = plan_pairs({"test": splits["test"]}, len(endpoints))
    code_files = (
        "run_memory_multislot.py", "memory_multislot_data.py", "report_memory_multislot.py",
        "run_memory_versioned.py", "run_memory_versioned_boundaries.py",
        "run_memory_versioned_provenance.py", "run_memory_experiment.py", "run_memory_confirmation.py",
        "run_memory_factual.py", "report_memory_factual.py", "report_memory_experiment.py",
        "report_memory_versioned.py", "report_memory_versioned_provenance.py",
        "memory_agent.py", "memory_experiment_data.py", "memory_versioned_boundaries_data.py",
        "memory_versioned_data.py", "memory_versioned_provenance_data.py",
        "minimal_validation_data.py", "collect_new_signals.py", "apibank_env.py", "react_agent.py",
    )
    protocol = {
        "experiment": "jointly-recalled-authorized-profile-memory-v9", "groups": GROUPS,
        "settings": asdict(SETTINGS), "criteria": MULTISLOT_CRITERIA, "counts": counts,
        "shared_boundary": SHARED_BOUNDARY, "dataset_path": str(args.cases.resolve()),
        "dataset_sha256": sha(args.cases), "resolved_cases_sha256": digest(splits),
        "memory_sha256": digest(memory), "checkpoint": str(args.model_path.resolve()),
        "code_sha256": {name: sha(ROOT / "code" / name) for name in code_files},
        "prior_cases_sha256": {path.name: sha(path) for path in MULTISLOT_PRIOR_CASE_FILES},
        "schema_corpora_sha256": {key: sha(path) for key, path in MULTISLOT_SCHEMA_CORPORA.items()},
        "model_metadata_sha256": {name: sha(args.model_path / name) for name in (
            "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "model.safetensors.index.json",
        )},
        "controls_plan": controls_plan, "test_plan": test_plan, "replica_endpoints": endpoints,
        "temperature_monitor": args.temperature_health is not None,
        "scoring": "field-level visible evidence; current values override recalled facts",
    }
    require_frozen(args.output / "protocol.json", protocol)
    require_frozen(args.output / "memory_M9.json", memory)
    if args.prepare_only:
        print(f"Frozen V9 protocol: {args.output / 'protocol.json'}")
        return
    counter = QwenPromptCounter(args.model_path)
    with ExitStack() as stack:
        clients = []
        for endpoint in endpoints:
            client = stack.enter_context(OpenAI(
                base_url=endpoint, api_key=os.getenv("OPENAI_API_KEY") or "EMPTY",
                http_client=httpx.Client(trust_env=False, timeout=300), max_retries=0,
            ))
            served = next((row for row in client.models.list().data if row.id == SETTINGS.model), None)
            if (served is None or not served.root or Path(served.root).resolve() != args.model_path.resolve()
                    or (served.model_extra or {}).get("max_model_len") != 8192):
                raise ValueError(f"Unmatched inference replica: {endpoint}")
            clients.append(client)
        requests = FramedRequests(clients, args.output, args.temperature_health)
        controls = collect_stage(splits["validation"], controls_plan, requests, counter, timelines, len(clients))
        write_json(args.output / "validation_results.json", controls)
        rows = []
        if controls_pass(controls):
            print("Controls passed. Running paired two-fact cases.", flush=True)
            rows = collect_stage(splits["test"], test_plan, requests, counter, timelines, len(clients))
            write_json(args.output / "test_results.json", rows)
        write_multislot_report(args.output, protocol, rows, controls, requests.failed)


def main():
    from memory_multislot_data import DEFAULT_MULTISLOT_CASES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cases", type=Path, default=DEFAULT_MULTISLOT_CASES)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--temperature-health", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--replica-base-url", action="append", default=[])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
