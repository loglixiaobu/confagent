#!/usr/bin/env python3
"""Test active saved facts with and without the user's authorization wording."""
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
from memory_agent import QwenPromptCounter
from report_memory_versioned_provenance import PROVENANCE_CRITERIA, write_provenance_report
from run_memory_experiment import MODEL_PATH, ROOT, collect_decision, plan_pairs, require_frozen
from run_memory_versioned import ORDERS, compile_facts, controls_pass, fact_rules, presented_case, used_stale_or_forbidden_value
from run_memory_versioned_boundaries import FramedRequests, SETTINGS, SHARED_BOUNDARY, framed

OUTPUT = ROOT / "outputs/profile_fact_memory_v8"
GROUPS = {"A": "M0", "B": "active fact with authorization", "C": "active fact only"}


def source_linked_rules(case, record):
    rule, = fact_rules(case, [record])
    evidence, = [turn["content"] for turn in record["observation"] if turn["role"] == "user"]
    source = json.dumps(evidence, ensure_ascii=False)
    scope = record["slot"].replace("_", " ")
    return [{
        **rule, "handling": (
            f"Profile lookup has already matched this record to the exact saved detail referenced "
            f"by the current requester, {scope}. The earlier user authorization was {source}. "
            f"{rule['handling']}"
        ),
    }]


def schedule(splits, replica_count):
    controls = plan_pairs({"validation": splits["validation"]}, replica_count)
    tests = []
    for dataset in ("replication", "updated"):
        for index, case in enumerate(splits[dataset]):
            cycle, replica = divmod(index, replica_count)
            arms = list(ORDERS[(cycle + 2 * replica) % len(ORDERS)])
            tests.append({"dataset": dataset, "task_id": case["task_id"], "replica_index": replica, "arms": arms})
    return controls, tests


def collect_stage(splits, plan, requests, counter, timelines, replica_count):
    by_id = {case["task_id"]: case for cases in splits.values() for case in cases}
    lock = Lock()

    def count_tokens(payload):
        with lock:
            return counter(framed(payload))

    def on_replica(replica):
        rows = []
        for i, scheduled in enumerate(plan):
            if scheduled["replica_index"] != replica:
                continue
            case = by_id[scheduled["task_id"]]
            timeline = timelines.get((case["profile_id"], case["requested_slot"]), [])
            latest = timeline[-1] if timeline else None
            dataset = scheduled["dataset"]
            for arm in scheduled["arms"]:
                rules = []
                if latest and arm != "A":
                    rules = source_linked_rules(case, latest) if arm == "B" else fact_rules(case, [latest])
                decision = collect_decision(presented_case(case, rules), requests, count_tokens, SETTINGS,
                                            f"{dataset}:{case['task_id']}:{arm}", rules, replica)
                rows.append((i, {
                    **decision, "dataset": dataset, "arm": arm, "replica_index": replica,
                    "condition": case["condition"], "memory_field": case["memory_field"],
                    "requested_slot": case["requested_slot"],
                    "permitted_memory_ids": case["permitted_memory_ids"] if arm != "A" else [],
                    "stale_or_forbidden_value_used": used_stale_or_forbidden_value(case, decision["action"]),
                }))
            print(f"{dataset} {case['task_id']} ({i + 1}/{len(plan)}, GPU {replica})", flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=replica_count) as pool:
        pairs = [item for batch in pool.map(on_replica, range(replica_count)) for item in batch]
    return [row for _, row in sorted(pairs, key=lambda item: item[0])]


def run(args):
    from memory_versioned_provenance_data import (
        PROVENANCE_PRIOR_CASE_FILES, PROVENANCE_SCHEMA_CORPORA, load_provenance_cases,
    )

    splits = load_provenance_cases(args.cases)
    counts = {name: len(rows) for name, rows in splits.items()}
    if counts != {"validation": 8, "replication": 16, "updated": 20}:
        raise ValueError(f"Unexpected independent case counts: {counts}")
    endpoints = [args.base_url, *args.replica_base_url]
    if len(endpoints) > 2 or len(set(endpoints)) != len(endpoints):
        raise ValueError("Supply one or two distinct replicas")
    timelines = compile_facts(splits)
    memory = {"version": "M8", "facts": [records[-1] for records in timelines.values() if records]}
    controls_plan, test_plan = schedule(splits, len(endpoints))
    paths = ("run_memory_versioned_provenance.py", "memory_versioned_provenance_data.py",
             "memory_versioned_data.py", "run_memory_versioned.py", "run_memory_factual.py",
             "run_memory_versioned_boundaries.py", "memory_versioned_boundaries_data.py",
             "run_memory_confirmation.py", "run_memory_experiment.py", "memory_experiment_data.py",
             "memory_agent.py", "minimal_validation_data.py", "collect_new_signals.py",
             "apibank_env.py", "react_agent.py", "report_memory_versioned_provenance.py",
             "report_memory_versioned.py", "report_memory_factual.py")
    protocol = {
        "experiment": "source-linked-versioned-memory-v8",
        "counts": counts, "settings": asdict(SETTINGS), "groups": GROUPS, "criteria": PROVENANCE_CRITERIA,
        "shared_boundary": SHARED_BOUNDARY, "dataset_sha256": sha(args.cases), "dataset": str(args.cases.resolve()),
        "resolved_cases_sha256": digest(splits), "memory_sha256": digest(memory),
        "checkpoint": str(args.model_path.resolve()),
        "model_metadata_sha256": {name: sha(args.model_path / name) for name in (
            "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "model.safetensors.index.json",
        )},
        "code_sha256": {name: sha(ROOT / "code" / name) for name in paths},
        "schema_corpora_sha256": {name: sha(path) for name, path in PROVENANCE_SCHEMA_CORPORA.items()},
        "prior_cases_sha256": {path.name: sha(path) for path in PROVENANCE_PRIOR_CASE_FILES},
        "controls_plan": controls_plan, "test_plan": test_plan, "replica_endpoints": endpoints,
        "temperature_monitor": args.temperature_health is not None,
    }
    require_frozen(args.output / "protocol.json", protocol)
    require_frozen(args.output / "memory_M8.json", memory)
    if args.prepare_only:
        print(f"Frozen V8 protocol: {args.output / 'protocol.json'}")
        return
    counter = QwenPromptCounter(args.model_path)
    with ExitStack() as stack:
        clients = []
        for endpoint in endpoints:
            client = stack.enter_context(OpenAI(base_url=endpoint, api_key=os.getenv("OPENAI_API_KEY") or "EMPTY",
                                               http_client=httpx.Client(trust_env=False, timeout=300), max_retries=0))
            served = next((row for row in client.models.list().data if row.id == SETTINGS.model), None)
            if (served is None or not served.root or Path(served.root).resolve() != args.model_path.resolve()
                    or (served.model_extra or {}).get("max_model_len") != 8192):
                raise ValueError(f"Unmatched inference replica: {endpoint}")
            clients.append(client)
        requests = FramedRequests(clients, args.output, args.temperature_health)
        controls = collect_stage(splits, controls_plan, requests, counter, timelines, len(clients))
        write_json(args.output / "validation_results.json", controls)
        rows = []
        if controls_pass(controls):
            print("Controls passed. Running source-linked memory comparisons.", flush=True)
            rows = collect_stage(splits, test_plan, requests, counter, timelines, len(clients))
            write_json(args.output / "test_results.json", rows)
        write_provenance_report(args.output, protocol, [r for r in rows if r["dataset"] == "replication"],
                                [r for r in rows if r["dataset"] == "updated"], controls, requests.failed)


def main():
    from memory_versioned_provenance_data import DEFAULT_PROVENANCE_CASES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cases", type=Path, default=DEFAULT_PROVENANCE_CASES)
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
