#!/usr/bin/env python3
"""Repeat the versioned-memory comparison with shared evidence rules."""
from __future__ import annotations

import argparse
import fcntl
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from threading import Lock

import httpx
from openai import OpenAI

from collect_new_signals import digest, sha, write_json
from memory_agent import QwenPromptCounter
from report_memory_versioned import VERSIONED_CRITERIA, _draw_comparison, _markdown_report, analyze_versioned
from run_memory_confirmation import TemperatureGatedRequests
from run_memory_experiment import MODEL_PATH, ROOT, collect_decision, require_frozen
from run_memory_versioned import (
    SETTINGS as PRIOR_SETTINGS,
    compile_facts, controls_pass, fact_rules, presented_case, schedule, used_stale_or_forbidden_value,
)

SETTINGS = replace(PRIOR_SETTINGS, max_new_tokens=1536)
GROUPS = {"A": "M0", "B": "M7 active fact", "C": "scoped chronological replay"}
OUTPUT = ROOT / "outputs/profile_fact_memory_v7"
ALPHA = 0.003125
SHARED_BOUNDARY = (
    "Rules for evidence in this session: a confirmed saved profile value supplied in the system "
    "context is evidence from the user's earlier authorized messages. Use it when the current "
    "request refers to that saved detail. An explicit value in the current request takes precedence. "
    "ToolSearcher can discover API definitions only, and cannot read the user's notes or profile. "
    "If a required value is unavailable from the current conversation, saved profile context, "
    "or a relevant exposed tool, ask for that specific value. Return only the immediate next action.\n\n"
)


def framed(payload):
    system, *rest = payload["messages"]
    return {**payload, "messages": [{**system, "content": SHARED_BOUNDARY + system["content"]}, *rest]}


class FramedRequests(TemperatureGatedRequests):
    def send(self, request_id, kind, payload, *, replica_index=0):
        return super().send(request_id, kind, framed(payload), replica_index=replica_index)


def collect_stage(splits, plan, requests, counter, timelines, replica_count):
    by_id = {case["task_id"]: case for cases in splits.values() for case in cases}
    gate = Lock()

    def count_tokens(payload):
        with gate:
            return counter(framed(payload))

    def on_replica(replica):
        rows = []
        for i, scheduled in enumerate(plan):
            if scheduled["replica_index"] != replica:
                continue
            case = by_id[scheduled["task_id"]]
            active_history = timelines.get((case["profile_id"], case["requested_slot"]), [])
            archive_ids = sorted(event["record_id"] for event in case["memory_events"]
                                 if event["kind"] == "save" and event["profile_id"] == case["profile_id"]
                                 and event["slot"] == case["requested_slot"])
            for arm in scheduled["arms"]:
                records = {"A": [], "B": active_history[-1:], "C": active_history}[arm]
                rules = fact_rules(case, records) if records else []
                permitted = {"A": [], "B": case["permitted_memory_ids"], "C": archive_ids}[arm]
                dataset = scheduled["dataset"]
                decision = collect_decision(presented_case(case, rules), requests, count_tokens, SETTINGS,
                                            f"{dataset}:{case['task_id']}:{arm}", rules, replica)
                rows.append((i, {
                    **decision, "dataset": dataset, "arm": arm, "replica_index": replica,
                    "condition": case["condition"], "memory_field": case["memory_field"],
                    "requested_slot": case["requested_slot"], "permitted_memory_ids": permitted,
                    "stale_or_forbidden_value_used": used_stale_or_forbidden_value(case, decision["action"]),
                }))
            print(f"{scheduled['dataset']} {case['task_id']} ({i + 1}/{len(plan)}, GPU {replica})", flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=replica_count) as pool:
        pairs = [item for batch in pool.map(on_replica, range(replica_count)) for item in batch]
    return [row for _, row in sorted(pairs, key=lambda item: item[0])]


def write_report(output, protocol, rows, controls, failures):
    from report_memory_versioned import _counts, _pair_rows

    if rows:
        analysis = analyze_versioned([r for r in rows if r["dataset"] == "replication"],
                                     [r for r in rows if r["dataset"] == "updated"], controls, 16)
        for endpoint in ("replication", "updated"):
            result = analysis[endpoint]
            result["primary_passed"] &= result["p_value"] <= ALPHA
            result["confirmed_positive"] &= result["p_value"] <= ALPHA
        analysis["both_confirmed_positive"] = all(analysis[key]["confirmed_positive"]
                                                    for key in ("replication", "updated"))
        chronology = analysis["chronological_replay"]
        chronology["efficiency_with_correctness"] &= chronology["paired_total_token_sign_p_value"] <= ALPHA
        status = "completed"
    else:
        paired = _pair_rows(controls, "validation", {"A", "B"}, 8)
        analysis = {"groups": {"controls": _counts(paired, ("A", "B"))}, "both_confirmed_positive": False}
        status = "control_failed"
    write_json(output / "summary.json", {
        "status": status, "analysis": analysis, "criteria": protocol["criteria"], "failed_requests": failures,
    })
    _draw_comparison(output, analysis)
    lines = _markdown_report(analysis)
    lines[0] = "# ConfAgent 版本化事实记忆与证据边界"
    lines[2:2] = [
        "此轮使用独立新题。所有组均明确保存事实是有效依据，以及 ToolSearcher 仅检索工具说明。"
        "三组使用相同的输入与输出总预算，单次输出上限为 1536 token。",
        "",
        f"三个比较的单侧预设显著性水平为 {ALPHA}。", "",
    ]
    if failures:
        lines += [f"另记录 {sum(failures.values())} 次请求异常。", ""]
    (output / "报告.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    from memory_versioned_boundaries_data import BOUNDARIES_SCHEMA_CORPORA, PRIOR_CASE_FILES, load_boundaries_cases

    splits = load_boundaries_cases(args.cases)
    counts = {name: len(rows) for name, rows in splits.items()}
    if counts != {"validation": 8, "replication": 16, "updated": 20}:
        raise ValueError(f"Unexpected independent case counts: {counts}")
    endpoints = [args.base_url, *args.replica_base_url]
    if len(endpoints) > 2 or len(set(endpoints)) != len(endpoints):
        raise ValueError("Supply one or two distinct replicas")
    timelines = compile_facts(splits)
    memory = {"version": "M7", "facts": [records[-1] for records in timelines.values() if records]}
    controls_plan, test_plan = schedule(splits, len(endpoints))
    criteria = deepcopy(VERSIONED_CRITERIA)
    criteria["replication"]["alpha"] = criteria["updated"]["alpha"] = ALPHA
    criteria["chronological_replay"]["alpha"] = ALPHA
    paths = ("run_memory_versioned_boundaries.py", "memory_versioned_boundaries_data.py",
             "memory_versioned_data.py", "run_memory_versioned.py", "run_memory_factual.py",
             "run_memory_confirmation.py", "run_memory_experiment.py", "memory_experiment_data.py",
             "memory_agent.py", "minimal_validation_data.py", "collect_new_signals.py",
             "apibank_env.py", "react_agent.py", "report_memory_versioned.py", "report_memory_factual.py")
    protocol = {
        "experiment": "versioned-memory-with-explicit-evidence-boundaries-v7",
        "counts": counts, "settings": asdict(SETTINGS), "groups": GROUPS, "criteria": criteria,
        "shared_boundary": SHARED_BOUNDARY, "dataset_sha256": sha(args.cases), "dataset": str(args.cases.resolve()),
        "resolved_cases_sha256": digest(splits), "memory_sha256": digest(memory),
        "checkpoint": str(args.model_path.resolve()),
        "model_metadata_sha256": {name: sha(args.model_path / name) for name in (
            "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "model.safetensors.index.json",
        )},
        "code_sha256": {name: sha(ROOT / "code" / name) for name in paths},
        "schema_corpora_sha256": {name: sha(path) for name, path in BOUNDARIES_SCHEMA_CORPORA.items()},
        "prior_cases_sha256": {path.name: sha(path) for path in PRIOR_CASE_FILES},
        "controls_plan": controls_plan, "test_plan": test_plan, "replica_endpoints": endpoints,
        "temperature_monitor": args.temperature_health is not None,
    }
    require_frozen(args.output / "protocol.json", protocol)
    require_frozen(args.output / "memory_M7.json", memory)
    if args.prepare_only:
        print(f"Frozen V7 protocol: {args.output / 'protocol.json'}")
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
            print("Controls passed. Running new frozen test cases.", flush=True)
            rows = collect_stage(splits, test_plan, requests, counter, timelines, len(clients))
            write_json(args.output / "test_results.json", rows)
        write_report(args.output, protocol, rows, controls, requests.failed)


def main():
    from memory_versioned_boundaries_data import DEFAULT_BOUNDARIES_CASES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cases", type=Path, default=DEFAULT_BOUNDARIES_CASES)
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
