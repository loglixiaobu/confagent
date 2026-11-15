#!/usr/bin/env python3
"""Learn from exposed grounding errors and confirm the effect on fresh pairs."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from threading import Lock

import httpx
from openai import OpenAI

from collect_new_signals import digest, read_jsonl, sha, write_json
from memory_agent import AgentSettings, QwenPromptCounter, RequestBudget
from memory_experiment_data import SCHEMA_CORPORA, evaluate_action, load_cases, load_stress_cases
from report_memory_confirmation import CRITERIA, write_confirmation_report
from run_memory_experiment import (
    MODEL_PATH, ROOT, SYNTHESIS_ATTEMPTS, SYNTHESIS_TOKEN_CAP, LoggedRequests,
    collect_decision, collect_test, parse_rules, plan_pairs, require_frozen,
)

DEFAULT_OUTPUT = ROOT / "outputs/missing_parameter_memory_v3"
EPISODE_OUTPUT = ROOT / "outputs/missing_parameter_memory_v4"
DEFAULT_FRESH = ROOT / "data/memory_grounding_confirmation_cases.json"
PREVIOUS = ROOT / "outputs/missing_parameter_memory_v2"
CONFIRMATION_SETTINGS = AgentSettings(max_new_tokens=1536)
SYNTHESIS_OUTPUT_CAP = 1400
GROUPS = {"A": "M0", "B": "M3"}
SYNTHESIS_PROMPT = """Write one to three transferable memory instructions for an API assistant.
The evidence contains calls, expected actions, and failure diagnoses from development tasks.
The new cases will use tools, fields, values and entities you have never seen. Learn a short
decision procedure for checking evidence immediately before emitting a call. A value appearing
in the conversation or matching a field's type is only a candidate. Infer how to decide if it
actually fills the requested parameter for the item and circumstances in question. In the
instructions, include concrete guidance for catching the distinct failures in the evidence,
and an explicit transition from the evidence check to the next step. Preserve the successful
cases: an unambiguous supported value can be used directly, and an available lookup may be the
appropriate next step. Give the agent a useful mental check and an actionable result. Avoid
repeating the same rule, naming real APIs, or memorizing names, IDs or values from the evidence.

Output only a JSON object {"rules":[...]}. Every rule has exactly these keys:
- "rule_id": a unique short string.
- "applicability": exactly {"problem_type":"missing_required_parameter","tools":[],"required_fields":[]}.
- "handling": a short self-contained English instruction. Include the trigger and the specific
  behavior, preferably with one short abstract example if it helps differentiate the sources.
- "source_ids": a nonempty list of distinct evidence record IDs from the supplied JSON."""
RETRY_PROMPT = """
The first candidate did not meet the frozen development or validation gate. Write fresh rules
that test whether each plausible value belongs to the required role for the current request.
The rule should tell the agent exactly what evidence to look for and what to do after the check,
including the ordinary case in which all values have evidence."""


class TemperatureGatedRequests(LoggedRequests):
    def __init__(self, clients, output, health):
        super().__init__(clients, output)
        self.health = health

    def send(self, request_id, kind, payload, *, replica_index=0):
        if self.health is not None and request_id not in self.completed:
            while True:
                state = json.loads(self.health.read_text(encoding="utf-8"))
                if (abs(time.time() - state["updated_at"]) > 3
                        or state["authorized_gpus"] != [0, 1]
                        or state["status"] in {"stopped", "error"}
                        or state["error"]):
                    raise RuntimeError("The temperature monitor is unavailable or unhealthy")
                if state["can_submit"]:
                    break
                time.sleep(0.5)
        return super().send(request_id, kind, payload, replica_index=replica_index)


def previous_development():
    """Use V2's three shared failures for development after exposing their outcomes."""
    original = load_cases()
    stress = load_stress_cases(existing_cases=original)
    stress_by_id = {row["task_id"]: row for row in stress}
    prior_rows = json.loads((PREVIOUS / "test_results.json").read_text(encoding="utf-8"))
    by_id = {}
    for row in prior_rows:
        if row["dataset"] == "stress":
            by_id.setdefault(row["task_id"], {})[row["arm"]] = row
    shared = [case for case in stress if case["expected"]["kind"] == "clarify"
              and all(evaluate_action(case, by_id[case["task_id"]][arm]["action"])["fabricated"]
                      for arm in ("A", "B"))]
    if len(shared) != 3:
        raise ValueError("Expected the three jointly fabricated V2 stress cases")
    old_evidence = json.loads((PREVIOUS / "memory_sources.json").read_text(encoding="utf-8"))
    evidence = list(old_evidence)
    diagnostics = []
    for missing in shared:
        complete = next(case for case in stress_by_id.values()
                        if case["base_task_id"] == missing["base_task_id"] and case["expected"]["kind"] == "call")
        for case, arm in ((missing, "A"), (complete, "B")):
            record = by_id[case["task_id"]][arm]
            label = evaluate_action(case, record["action"])
            if label != record["evaluation"]:
                raise ValueError(f"Historical development label changed for {case['task_id']}")
            evidence.append({
                "source_id": case["task_id"], "source": "exposed V2 stress trajectory",
                "requirement": case["requirement"], "history": case["history"],
                "tool_fields": {doc["name"]: doc["input_parameters"] for doc in case["api_documents"]},
                "action": record["action"], "expected": case["expected"], "feedback": label,
                "evidence_boundary": case["provenance"]["evidence_boundary"],
            })
            diagnostics.append(case)
    return original, stress, evidence, diagnostics


def candidate_decisions(cases, lane, count_tokens, rules, prefix, replica_count):
    def on_replica(replica):
        return [
            (index, collect_decision(case, lane, count_tokens, CONFIRMATION_SETTINGS,
                                     f"{prefix}:{case['task_id']}", rules, replica))
            for index, case in enumerate(cases) if index % replica_count == replica
        ]

    with ThreadPoolExecutor(max_workers=replica_count) as executor:
        rows = [pair for batch in executor.map(on_replica, range(replica_count)) for pair in batch]
    return [row for _, row in sorted(rows)]


def episode_rules(evidence):
    """Compile observed false bindings and their grounded counterparts into reusable episodes."""
    rows = [row for row in evidence if row["source"] == "exposed V2 stress trajectory"]
    if len(rows) != 6:
        raise ValueError("The contrastive memory requires all three exposed scenario pairs")
    rules = []
    for index, (missing, complete) in enumerate(zip(rows[::2], rows[1::2]), start=1):
        field, = missing["expected"]["fields"]
        wrong = missing["action"]["arguments"][field]
        right = complete["expected"]["arguments"][field]
        handling = (
            "Before a call, ask yourself which exact phrase or tool result assigns the candidate "
            "to the required role for this target, source and situation. If the only candidate describes "
            "something else, the field is still missing. Use a relevant lookup if available or ask the user "
            "for that exact field. Call directly when all required values have matching evidence. "
            f"Past failure: {missing['requirement']} "
            f"The assistant filled {field} with {json.dumps(wrong)}. {missing['evidence_boundary']} "
            f"The correct next step was to ask for {field}. "
            f"When the user supplied {json.dumps(right)} for that exact field, the call became grounded. "
            "Transfer the source check to the new request."
        )
        rules.append({
            "rule_id": f"source_binding_episode_{index}",
            "applicability": {"problem_type": "missing_required_parameter", "tools": [], "required_fields": []},
            "handling": handling, "source_ids": [missing["source_id"], complete["source_id"]],
        })
    return rules


def build_candidate(output, evidence, diagnostics, validation, lane, count_tokens, replica_count,
                    episodes=False):
    require_frozen(output / "memory_sources.json", evidence)
    require_frozen(output / "memory_M0.json", {"version": "M0", "rules": []})
    baseline = candidate_decisions(diagnostics, lane, count_tokens, (), "development:A", replica_count)
    write_json(output / "development_baseline.json", baseline)
    attempts = []
    source_ids = {row["source_id"] for row in evidence}
    for number in range(1, 2 if episodes else SYNTHESIS_ATTEMPTS + 1):
        budget = RequestBudget(SYNTHESIS_TOKEN_CAP, lane.send, count_tokens)
        attempt = {"attempt": number, "diagnostics": [], "validation": []}
        attempts.append(attempt)
        try:
            if episodes:
                rules = episode_rules(evidence)
            else:
                prompt = SYNTHESIS_PROMPT if number == 1 else SYNTHESIS_PROMPT + RETRY_PROMPT
                response = budget.complete(f"synthesis:{number}", "memory_synthesis", {
                    "model": CONFIRMATION_SETTINGS.model, "temperature": CONFIRMATION_SETTINGS.temperature,
                    "top_p": CONFIRMATION_SETTINGS.top_p, "seed": CONFIRMATION_SETTINGS.seed,
                    "max_tokens": SYNTHESIS_OUTPUT_CAP,
                    "messages": [{"role": "system", "content": prompt},
                                 {"role": "user", "content": json.dumps(
                                     evidence, ensure_ascii=False, separators=(",", ":"))}],
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                })
                rules = parse_rules(response, source_ids)
        except ValueError as exc:
            attempt.update(status="invalid_rules", error=str(exc))
        else:
            rows = candidate_decisions(diagnostics, lane, count_tokens, rules,
                                       f"development:B:{number}", replica_count)
            repairs = sum(row["evaluation"]["reasonable_clarification"] for row in rows
                          if row["expected"]["kind"] == "clarify")
            controls = sum(row["evaluation"]["correct"] for row in rows if row["expected"]["kind"] == "call")
            attempt.update(rules=rules, diagnostics=rows, repaired_historical_errors=repairs,
                           correct_complete_controls=controls, status="failed_diagnostics")
            if repairs >= 2 and controls == 3:
                checks = candidate_decisions(validation, lane, count_tokens, rules,
                                             f"validation:{number}", replica_count)
                all_ids = {rule["rule_id"] for rule in rules}
                passed = all(row["evaluation"]["correct"] and set(row["memory_rule_ids"]) == all_ids
                             for row in checks)
                attempt.update(validation=checks, status="passed" if passed else "failed_validation")
        attempt["cost"] = budget.cost
        write_json(output / "memory_attempts.json", attempts)
        print(f"Candidate {number}: {attempt['status']}, diagnostic repairs "
              f"{attempt.get('repaired_historical_errors', 0)}/3", flush=True)
        if attempt["status"] == "passed":
            version = "M4" if episodes else "M3"
            memory = {
                "version": version, "rules": attempt["rules"], "rules_sha256": digest(attempt["rules"]),
                "source_sha256": digest(evidence), "accepted_attempt": number,
            }
            require_frozen(output / f"memory_{version}.json", memory)
            return memory, baseline, attempts
    if any((output / f"memory_{version}.json").exists() for version in ("M3", "M4")):
        raise ValueError("Stored memory conflicts with reconstructed development checks")
    return None, baseline, attempts


def freeze_design(args, evidence, diagnostics, validation, endpoints):
    code_files = ("run_memory_confirmation.py", "report_memory_confirmation.py",
                  "run_memory_experiment.py", "memory_agent.py", "memory_experiment_data.py",
                  "apibank_env.py", "react_agent.py", "collect_new_signals.py",
                  "step_confidence.py", "step_monitor.py", "minimal_validation_data.py")
    design = {
        "experiment": "contrastive-episode-memory-v4" if args.episodes else "source-binding-memory-v3",
        "settings": asdict(CONFIRMATION_SETTINGS),
        "memory_method": "compiled_contrastive_episodes" if args.episodes else "model_summary",
        "synthesis_attempt_limit": 0 if args.episodes else SYNTHESIS_ATTEMPTS,
        "synthesis_output_cap": SYNTHESIS_OUTPUT_CAP,
        "evidence_sha256": digest(evidence), "diagnostic_cases_sha256": digest(diagnostics),
        "validation_cases_sha256": digest(validation),
        "development_gate": {"minimum_repaired_prior_errors": 2, "complete_controls": 3, "validation": 4},
        "confirmation": CRITERIA, "replica_endpoints": endpoints,
        "groups": {"A": "M0", "B": "M4"} if args.episodes else GROUPS,
        "temperature_monitor": args.temperature_health is not None,
        "model_path": str(args.model_path.resolve()),
        "model_metadata_sha256": {name: sha(args.model_path / name) for name in (
            "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "model.safetensors.index.json",
        )},
        "code_sha256": {name: sha(ROOT / "code" / name) for name in code_files},
    }
    if args.episodes:
        earlier = DEFAULT_OUTPUT / "requests.jsonl"
        if any(row["request_id"].startswith("test:") for row in read_jsonl(earlier)):
            raise ValueError("This follow-up requires an untouched V3 confirmation cohort")
        design["prior_attempt_requests_sha256"] = sha(earlier)
    require_frozen(args.output / "design.json", design)
    return design


def run(args):
    endpoints = [args.base_url, *args.replica_base_url]
    if len(endpoints) > 2 or len(endpoints) != len(set(endpoints)):
        raise ValueError("Supply one or two distinct replica endpoints")
    prior, stress, evidence, diagnostics = previous_development()
    validation = prior["validation"]
    design = freeze_design(args, evidence, diagnostics, validation, endpoints)
    if args.prepare_only:
        print(f"Frozen development and analysis design: {args.output / 'design.json'}")
        return
    counter = QwenPromptCounter(args.model_path)
    token_lock = Lock()

    def count_tokens(payload):
        with token_lock:
            return counter(payload)

    with ExitStack() as stack:
        clients, serving = [], []
        for endpoint in endpoints:
            client = stack.enter_context(OpenAI(
                base_url=endpoint, api_key=os.getenv("OPENAI_API_KEY") or "EMPTY",
                http_client=httpx.Client(trust_env=False, timeout=300), max_retries=0,
            ))
            served = next((row for row in client.models.list().data if row.id == CONFIRMATION_SETTINGS.model), None)
            context = (served.model_extra or {}).get("max_model_len") if served is not None else None
            if (served is None or not served.root or Path(served.root).resolve() != args.model_path.resolve()
                    or context != SYNTHESIS_TOKEN_CAP):
                raise ValueError(f"Replica {endpoint} does not serve the frozen checkpoint and context")
            serving.append({"endpoint": endpoint, "checkpoint": served.root, "max_model_len": context})
            clients.append(client)
        require_frozen(args.output / "serving_model.json", {"replicas": serving})
        lane = TemperatureGatedRequests(clients, args.output, args.temperature_health)
        memory, baseline, attempts = build_candidate(
            args.output, evidence, diagnostics, validation, lane, count_tokens, len(clients), args.episodes,
        )
        if memory is None:
            write_confirmation_report(args.output, design, memory, baseline, attempts, [], lane.failed)
            return
        if args.construct_only:
            print(f"Frozen memory: {memory['version']}")
            return

        from memory_confirmation_data import load_confirmation_cases

        fresh = load_confirmation_cases(args.fresh_cases, existing_cases={**prior, "stress": stress})
        test_sets = {"confirmation": fresh}
        plan = plan_pairs(test_sets, len(clients))
        protocol = {
            **design, "dataset": str(args.fresh_cases.resolve()), "dataset_sha256": sha(args.fresh_cases),
            "resolved_cases_sha256": digest(fresh), "dataset_loader_sha256": sha(ROOT / "code/memory_confirmation_data.py"),
            "schema_corpora_sha256": {name: sha(path) for name, path in SCHEMA_CORPORA.items()},
            "test_plan": plan, "memory_sha256": memory["rules_sha256"],
        }
        require_frozen(args.output / "protocol.json", protocol)
        decisions = collect_test(test_sets, lane, count_tokens, CONFIRMATION_SETTINGS,
                                 plan, memory, design["groups"], len(clients))
        write_json(args.output / "test_results.json", decisions)
        write_confirmation_report(args.output, protocol, memory, baseline, attempts, decisions, lane.failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episodes", action="store_true", help="Use exposed error/repair pairs as memory episodes")
    parser.add_argument("--fresh-cases", type=Path, default=DEFAULT_FRESH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--replica-base-url", action="append", default=[])
    parser.add_argument("--temperature-health", type=Path, help="Health file for the live GPU 0/1 monitor")
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--prepare-only", action="store_true")
    stage.add_argument("--construct-only", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        args.output = EPISODE_OUTPUT if args.episodes else DEFAULT_OUTPUT
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()
