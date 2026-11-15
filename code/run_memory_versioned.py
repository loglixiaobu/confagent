#!/usr/bin/env python3
"""Evaluate fresh recall and later revisions of explicitly saved facts."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from threading import Lock

import httpx
from openai import OpenAI

from collect_new_signals import digest, sha, write_json
from memory_agent import PROBLEM_TYPE, QwenPromptCounter
from memory_experiment_data import SCHEMA_CORPORA
from minimal_validation_data import argument_equal
from report_memory_versioned import VERSIONED_CRITERIA, write_versioned_report
from run_memory_confirmation import TemperatureGatedRequests
from run_memory_experiment import MODEL_PATH, ROOT, collect_decision, plan_pairs, require_frozen
from run_memory_factual import SETTINGS

OUTPUT = ROOT / "outputs/profile_fact_memory_v6"
GROUPS = {"A": "M0", "B": "M6 active fact", "C": "scoped chronological replay"}
ORDERS = ("ABC", "BCA", "CAB", "ACB", "CBA", "BAC")


def compile_facts(splits):
    """Build per-profile and per-field timelines starting after any revocation."""
    by_scope = {}
    by_id = {}
    for cases in splits.values():
        for case in cases:
            for event in case["memory_events"]:
                event_id = event["record_id"]
                if event_id in by_id:
                    if by_id[event_id] != event:
                        raise ValueError(f"Conflicting event ID: {event_id}")
                    continue
                by_id[event_id] = event
                key = (event["profile_id"], event["slot"])
                sequence = event["sequence"]
                scope = by_scope.setdefault(key, {})
                if type(sequence) is not int or sequence in scope:
                    raise ValueError(f"Invalid or conflicting sequence for: {key}")
                scope[sequence] = event
    timelines = {}
    for key, events in by_scope.items():
        active_history = []
        for _, event in sorted(events.items()):
            if event["kind"] == "revoke":
                active_history.clear()
            elif event["kind"] == "save":
                active_history.append(event)
            else:
                raise ValueError(f"Unknown memory operation: {event['kind']}")
        timelines[key] = active_history
    return timelines


def fact_rules(case, events):
    """Use the same fact formatting for active-state and chronological replay."""
    field = case["memory_field"]
    apis = [doc["name"] for doc in case["api_documents"] if field in doc["input_parameters"]]
    if not apis:
        raise ValueError(f"Missing API field {field} for {case['task_id']}")
    rules = []
    for index, event in enumerate(events):
        value = json.dumps(event["value"], ensure_ascii=False)
        handling = f"From the current user's saved profile, revision {event['sequence']} for {field} is {value}."
        if index == len(events) - 1:
            handling += (
                " Use the latest confirmed revision for a request referring to this saved detail. "
                "An explicit value in the current request takes precedence."
            )
        rules.append({
            "rule_id": event["record_id"],
            "applicability": {"problem_type": PROBLEM_TYPE, "tools": apis, "required_fields": [field]},
            "handling": handling,
            "source_ids": [event["record_id"]],
        })
    return rules


def presented_case(case, rules):
    if rules or case["condition"] not in {"saved", "updated"}:
        return case
    field = case["memory_field"]
    return {
        **case, "parameter_sources": {**case["parameter_sources"], field: {"source": "missing"}},
        "expected": {"kind": "clarify", "fields": [field]},
    }


def used_stale_or_forbidden_value(case, action):
    if action.get("kind") != "call" or action.get("api_name") != case["target_tool"]:
        return False
    field = case["memory_field"]
    if field not in action["arguments"]:
        return False
    metadata = next(doc["input_parameters"][field] for doc in case["api_documents"]
                    if doc["name"] == case["target_tool"])
    value = action["arguments"][field]
    expected = case["parameter_sources"][field]
    if expected["source"] in {"history", "requirement"} and argument_equal(
        field, value, expected["value"], metadata,
    ):
        return False
    return any(event["kind"] == "save" and argument_equal(field, value, event["value"], metadata)
               for event in case["memory_events"])


def schedule(splits, replica_count):
    controls = plan_pairs({"validation": splits["validation"]}, replica_count)
    replication = plan_pairs({"replication": splits["replication"]}, replica_count)
    updated = []
    for index, case in enumerate(splits["updated"]):
        round_index, replica = divmod(index, replica_count)
        arms = list(ORDERS[(round_index + replica * 2) % len(ORDERS)])
        updated.append({"dataset": "updated", "task_id": case["task_id"], "replica_index": replica, "arms": arms})
    return controls, replication + updated


def collect_cases(splits, plan, lane, counter, timelines, replica_count):
    by_id = {case["task_id"]: case for cases in splits.values() for case in cases}
    guard = Lock()

    def count_tokens(payload):
        with guard:
            return counter(payload)

    def run_replica(replica):
        rows = []
        for index, scheduled in enumerate(plan):
            if scheduled["replica_index"] != replica:
                continue
            case = by_id[scheduled["task_id"]]
            history = timelines.get((case["profile_id"], case["requested_slot"]), [])
            dataset = scheduled["dataset"]
            for arm in scheduled["arms"]:
                events = {"A": [], "B": history[-1:], "C": history}[arm]
                rules = fact_rules(case, events) if events else []
                permitted = {
                    "A": [],
                    "B": case["permitted_memory_ids"],
                    "C": sorted(event["record_id"] for event in case["memory_events"]
                                if event["kind"] == "save" and event["profile_id"] == case["profile_id"]
                                and event["slot"] == case["requested_slot"]),
                }[arm]
                row = collect_decision(presented_case(case, rules), lane, count_tokens, SETTINGS,
                                       f"{dataset}:{case['task_id']}:{arm}", rules, replica)
                rows.append((index, {
                    **row, "arm": arm, "dataset": dataset, "replica_index": replica,
                    "memory_field": case["memory_field"], "requested_slot": case["requested_slot"],
                    "condition": case["condition"], "permitted_memory_ids": permitted,
                    "stale_or_forbidden_value_used": used_stale_or_forbidden_value(case, row["action"]),
                }))
            print(f"{dataset} {case['task_id']} ({index + 1}/{len(plan)}, GPU {replica})", flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=replica_count) as pool:
        rows = [item for batch in pool.map(run_replica, range(replica_count)) for item in batch]
    return [row for _, row in sorted(rows, key=lambda item: item[0])]


def controls_pass(rows):
    counts = {arm: sum(r["evaluation"]["correct"] for r in rows if r["arm"] == arm) for arm in ("A", "B")}
    return (counts["B"] >= max(counts["A"], VERSIONED_CRITERIA["minimum_correct_controls"])
            and all(set(r["memory_rule_ids"]) == set(r["permitted_memory_ids"]) for r in rows)
            and not any(r["stale_or_forbidden_value_used"] for r in rows if r["arm"] == "B"))


def run(args):
    from memory_versioned_data import load_versioned_cases

    splits = load_versioned_cases(args.cases)
    counts = {name: len(rows) for name, rows in splits.items()}
    if counts.get("validation") != 8 or counts.get("updated") != 20 or counts.get("replication") not in {12, 16}:
        raise ValueError(f"Unexpected versioned-memory case counts: {counts}")
    endpoints = [args.base_url, *args.replica_base_url]
    if len(endpoints) > 2 or len(endpoints) != len(set(endpoints)):
        raise ValueError("Supply one or two distinct local replicas")
    timelines = compile_facts(splits)
    store = {"version": "M6", "facts": [records[-1] for records in timelines.values() if records]}
    controls_plan, test_plan = schedule(splits, len(endpoints))
    code = ("run_memory_versioned.py", "memory_versioned_data.py", "report_memory_versioned.py",
            "run_memory_factual.py", "run_memory_confirmation.py", "run_memory_experiment.py",
            "memory_experiment_data.py", "memory_agent.py", "minimal_validation_data.py", "apibank_env.py", "react_agent.py")
    protocol = {
        "experiment": "versioned-profile-fact-memory-v6", "groups": GROUPS, "counts": counts,
        "replication_cases": counts["replication"],
        "settings": asdict(SETTINGS), "criteria": VERSIONED_CRITERIA,
        "dataset_path": str(args.cases.resolve()), "dataset_sha256": sha(args.cases),
        "resolved_cases_sha256": digest(splits), "source": "local API-Bank definitions; independent controlled scenarios",
        "schema_corpora_sha256": {name: sha(path) for name, path in SCHEMA_CORPORA.items()},
        "code_sha256": {name: sha(ROOT / "code" / name) for name in code},
        "memory_sha256": digest(store),
        "checkpoint": str(args.model_path.resolve()),
        "model_metadata_sha256": {name: sha(args.model_path / name) for name in (
            "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "model.safetensors.index.json",
        )},
        "controls_plan": controls_plan, "test_plan": test_plan, "replica_endpoints": endpoints,
        "temperature_monitor": args.temperature_health is not None,
        "retention": "exact profile and slot, latest explicit save after the last explicit revocation",
        "replay": "all compatible post-revocation versions in chronological order, update cases only",
    }
    require_frozen(args.output / "protocol.json", protocol)
    require_frozen(args.output / "memory_M6.json", store)
    if args.prepare_only:
        print(f"Frozen V6 protocol: {args.output / 'protocol.json'}")
        return
    counter = QwenPromptCounter(args.model_path)
    with ExitStack() as stack:
        clients = []
        for endpoint in endpoints:
            client = stack.enter_context(OpenAI(base_url=endpoint, api_key=os.getenv("OPENAI_API_KEY") or "EMPTY",
                                               http_client=httpx.Client(trust_env=False, timeout=300), max_retries=0))
            serving = next((row for row in client.models.list().data if row.id == SETTINGS.model), None)
            if (serving is None or not serving.root or Path(serving.root).resolve() != args.model_path.resolve()
                    or (serving.model_extra or {}).get("max_model_len") != 8192):
                raise ValueError(f"Unmatched inference replica: {endpoint}")
            clients.append(client)
        lane = TemperatureGatedRequests(clients, args.output, args.temperature_health)
        controls = collect_cases(splits, controls_plan, lane, counter, timelines, len(clients))
        write_json(args.output / "validation_results.json", controls)
        rows = []
        if controls_pass(controls):
            print("Control gate passed. Running both frozen test cohorts.", flush=True)
            rows = collect_cases(splits, test_plan, lane, counter, timelines, len(clients))
            write_json(args.output / "test_results.json", rows)
        write_versioned_report(args.output, protocol, [r for r in rows if r["dataset"] == "replication"],
                               [r for r in rows if r["dataset"] == "updated"], controls, lane.failed)


def main():
    from memory_versioned_data import DEFAULT_VERSIONED_CASES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cases", type=Path, default=DEFAULT_VERSIONED_CASES)
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
