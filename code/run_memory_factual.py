#!/usr/bin/env python3
"""Confirm whether explicitly retained profile facts avoid repeated questions."""
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
from memory_agent import AgentSettings, PROBLEM_TYPE, QwenPromptCounter
from report_memory_factual import FACTUAL_CRITERIA, write_factual_report
from run_memory_confirmation import TemperatureGatedRequests
from run_memory_experiment import MODEL_PATH, ROOT, collect_decision, plan_pairs, require_frozen

OUTPUT = ROOT / "outputs/profile_fact_memory_v5"
SETTINGS = AgentSettings(max_new_tokens=1024)
GROUPS = {"A": "M0", "B": "M5"}
VALIDATION_MINIMUM_CORRECT = 11


def build_store(splits):
    """Store facts from recorded earlier user messages, independent of later decisions."""
    by_key, record_ids = {}, set()
    for cases in splits.values():
        for case in cases:
            record = case["memory_record"]
            if record is None or record["confirmed"] is not True:
                continue
            key = (record["profile_id"], record["slot"])
            if key in by_key or record["record_id"] in record_ids:
                if by_key.get(key) == record:
                    continue
                raise ValueError("Conflicting or duplicate stored fact")
            record_ids.add(record["record_id"])
            by_key[key] = record
    return by_key


def fact_rules(case, store):
    """Retrieve by current profile and requested slot, before viewing any action."""
    record = store.get((case["profile_id"], case["requested_slot"]))
    if record is None:
        return []
    field = case["memory_field"]
    supported_tools = [doc["name"] for doc in case["api_documents"] if field in doc["input_parameters"]]
    if not supported_tools:
        return []
    value = json.dumps(record["value"], ensure_ascii=False)
    return [{
        "rule_id": record["record_id"],
        "applicability": {"problem_type": PROBLEM_TYPE, "tools": supported_tools, "required_fields": [field]},
        "handling": (
            f"The current user explicitly asked to save their {record['slot']} in an earlier session. "
            f"The confirmed saved value for {field} is {value}. Use this value if the current request "
            "calls for that saved detail. A value supplied in the current request takes precedence."
        ),
        "source_ids": [record["record_id"]],
    }]


def presented_case(case, rules):
    """Label decisions against the evidence actually available to that arm."""
    if case["condition"] != "saved" or rules:
        return case
    field = case["memory_field"]
    return {
        **case, "parameter_sources": {**case["parameter_sources"], field: {"source": "missing"}},
        "expected": {"kind": "clarify", "fields": [field]},
    }


def collect_factual_cases(cases, plan, requests, counter, store, replica_count):
    by_id = {case["task_id"]: case for case in cases}
    counter_lock = Lock()

    def count_tokens(payload):
        with counter_lock:
            return counter(payload)

    def on_replica(replica):
        rows = []
        for index, scheduled in enumerate(plan):
            if scheduled["replica_index"] != replica:
                continue
            case = by_id[scheduled["task_id"]]
            permitted = ([case["memory_record"]["record_id"]]
                         if case["condition"] in {"saved", "current_update"} else [])
            for arm in scheduled["arms"]:
                rules = fact_rules(case, store) if arm == "B" else []
                row = collect_decision(presented_case(case, rules), requests, count_tokens, SETTINGS,
                                       f"{scheduled['dataset']}:{case['task_id']}:{arm}", rules, replica)
                rows.append((index, {
                    **row, "arm": arm, "dataset": scheduled["dataset"], "replica_index": replica,
                    "memory_version": GROUPS[arm], "condition": case["condition"],
                    "requested_slot": case["requested_slot"], "memory_field": case["memory_field"],
                    "permitted_memory_ids": permitted if arm == "B" else [],
                }))
            print(f"{scheduled['dataset']} pair {index + 1}/{len(plan)}: {case['task_id']}", flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=replica_count) as executor:
        rows = [item for batch in executor.map(on_replica, range(replica_count)) for item in batch]
    return [row for _, row in sorted(rows, key=lambda item: item[0])]


def passes_validation(rows):
    scores = {arm: sum(row["evaluation"]["correct"] for row in rows if row["arm"] == arm)
              for arm in ("A", "B")}
    return (scores["B"] >= max(scores["A"], VALIDATION_MINIMUM_CORRECT)
            and all(row["memory_rule_ids"] == row["permitted_memory_ids"] for row in rows))


def run(args):
    from memory_factual_data import load_factual_cases

    endpoints = [args.base_url, *args.replica_base_url]
    if len(endpoints) > 2 or len(endpoints) != len(set(endpoints)):
        raise ValueError("Supply one or two distinct inference replicas")
    splits = load_factual_cases(args.cases)
    test, controls = splits["test"], splits["validation"]
    if len(test) != 24 or len(controls) != 12:
        raise ValueError("Expected 24 saved-fact tasks and 12 independent controls")
    store = build_store(splits)
    memory = {"version": "M5", "facts": list(store.values())}
    validation_plan = plan_pairs({"factual_validation": controls}, len(endpoints))
    test_plan = plan_pairs({"factual_test": test}, len(endpoints))
    protocol = {
        "experiment": "cross-session-profile-fact-memory-v5", "groups": GROUPS,
        "settings": asdict(SETTINGS), "confirmation": FACTUAL_CRITERIA,
        "checkpoint": str(args.model_path.resolve()),
        "model_metadata_sha256": {name: sha(args.model_path / name) for name in (
            "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "model.safetensors.index.json",
        )},
        "code_sha256": {name: sha(ROOT / "code" / name) for name in (
            "run_memory_factual.py", "memory_factual_data.py", "report_memory_factual.py",
            "run_memory_confirmation.py", "run_memory_experiment.py", "memory_agent.py",
            "memory_experiment_data.py", "apibank_env.py", "react_agent.py",
        )},
        "dataset_path": str(args.cases.resolve()), "dataset_sha256": sha(args.cases),
        "resolved_cases_sha256": digest(splits), "memory_sha256": digest(memory),
        "validation_plan": validation_plan, "test_plan": test_plan, "replicas": endpoints,
        "temperature_monitor": args.temperature_health is not None,
        "retrieval": "confirmed earlier observation with exact profile and requested semantic slot",
        "scoring": "same visible-source policy; without retrieval the saved field is ungrounded and should be asked",
    }
    require_frozen(args.output / "protocol.json", protocol)
    require_frozen(args.output / "memory_M0.json", {"version": "M0", "facts": []})
    require_frozen(args.output / "memory_M5.json", memory)
    if args.prepare_only:
        print(f"Frozen factual-memory design: {args.output / 'protocol.json'}")
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
            context = (served.model_extra or {}).get("max_model_len") if served is not None else None
            if (served is None or not served.root or Path(served.root).resolve() != args.model_path.resolve()
                    or context != 8192):
                raise ValueError(f"Replica {endpoint} does not serve the frozen checkpoint and context")
            clients.append(client)
        requests = TemperatureGatedRequests(clients, args.output, args.temperature_health)
        validation = collect_factual_cases(controls, validation_plan, requests, counter, store, len(clients))
        write_json(args.output / "validation_results.json", validation)
        rows = []
        if passes_validation(validation):
            print("Factual-memory controls passed. Starting frozen held-out test.", flush=True)
            rows = collect_factual_cases(test, test_plan, requests, counter, store, len(clients))
            write_json(args.output / "test_results.json", rows)
        write_factual_report(args.output, protocol, rows, validation, requests.failed)


def main():
    from memory_factual_data import DEFAULT_FACTUAL_CASES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cases", type=Path, default=DEFAULT_FACTUAL_CASES)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--replica-base-url", action="append", default=[])
    parser.add_argument("--temperature-health", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
