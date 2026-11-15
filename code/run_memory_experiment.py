#!/usr/bin/env python3
"""Build and gate an episodic memory, then compare paired one-step decisions."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from functools import partial
from pathlib import Path
from threading import Lock

from collect_new_signals import digest, read_jsonl, sha, write_json
from memory_agent import (AgentSettings, QwenPromptCounter, RequestBudget,
                          run_next_step, self_report_for_action)
from memory_experiment_data import (DEFAULT_CASES, SCHEMA_CORPORA, evaluate_action,
                                    load_cases, simulate_action)
from report_memory_experiment import write_report

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("CONFAGENT_MODEL_PATH") or ROOT / "models" / "Qwen3-8B").expanduser()
DEFAULT_OUTPUT = ROOT / "outputs/missing_parameter_memory_v2"
LEGACY_LABELS = ROOT / "outputs/legacy_binary_v2/calls_binary.jsonl"
LEGACY_FEEDBACK = ROOT / "outputs/legacy_binary_v2/score_blind_review.jsonl"
LEGACY_SCORES = ROOT / "outputs/new_signals_fast_v1/scores.jsonl"
LEGACY_IDS = ("l3-010:call-06", "l3-037:call-02", "l3-035:call-01")
RULE_LIMIT = 3
SYNTHESIS_ATTEMPTS = 2
SYNTHESIS_TOKEN_CAP = 8192
SYNTHESIS_OUTPUT_CAP = 2048
PROBLEM_TYPE = "missing_required_parameter"
SYNTHESIS_PROMPT = """Extract transferable memory rules for an API assistant from the supplied
development evidence. Each record connects a user's request, an observed next action, and
feedback. Self-reported confidence prioritizes review and is not proof of correctness.
Focus on unsupported required arguments. Compare error and success cases, including the
evidence about values available from prior messages and tools. Infer a decision procedure
that is usable with tool schemas that never appear in these records. Describe the condition
and next action in the handling text using general terms. Avoid example values, individual
tool names, and individual field names in the text. Write one to three concise rules.
Return only a JSON object with the single key "rules", a nonempty array. Each array item
must have exactly these keys:
- "rule_id": a short unique identifier;
- "applicability": the fixed object {"problem_type":"missing_required_parameter",
  "tools":[],"required_fields":[]}. Copy this object exactly for every rule. Both
  arrays must be empty so the rule can apply to unseen APIs;
- "handling": a short English instruction stating the condition and the action;
- "source_ids": a nonempty list of evidence record IDs that support the rule."""


class LoggedRequests:
    """Keep raw requests and responses and replay completed requests on restart."""

    def __init__(self, clients, output):
        self.clients = clients
        self.path = output / "requests.jsonl"
        self.lock = Lock()
        self.completed = {}
        self.failed = {}
        for row in read_jsonl(self.path):
            if "response" in row:
                if row["request_id"] in self.completed:
                    raise ValueError(f"Duplicate completed request {row['request_id']}")
                self.completed[row["request_id"]] = row
            else:
                prefix = row["request_id"].split(":", 1)[0]
                self.failed[prefix] = self.failed.get(prefix, 0) + 1

    def send(self, request_id, kind, payload, *, replica_index=0):
        signature = digest(payload)
        with self.lock:
            cached = self.completed.get(request_id)
        if cached is not None:
            if (cached["payload_sha256"] != signature or cached["kind"] != kind
                    or cached["replica_index"] != replica_index):
                raise ValueError(f"Frozen request changed: {request_id}")
            return cached["response"]
        row = {"request_id": request_id, "kind": kind, "payload_sha256": signature,
               "request": payload, "replica_index": replica_index}
        started = time.monotonic()
        try:
            response = self.clients[replica_index].chat.completions.create(**payload)
            row["response"] = response.model_dump()
        except (Exception, KeyboardInterrupt) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            row["latency_s"] = round(time.monotonic() - started, 4)
            with self.lock:
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                if "response" in row:
                    self.completed[request_id] = row
        return row["response"]


def require_frozen(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Frozen artifact changed: {path}. Use a new output directory.")
    else:
        write_json(path, value)


def historical_evidence():
    labels = {row["sample_id"]: row for row in read_jsonl(LEGACY_LABELS)}
    audits = {row["event_id"]: row for row in read_jsonl(LEGACY_FEEDBACK)}
    scores = {(row["task_id"], row["step_id"]): row["self_report"]
              for row in read_jsonl(LEGACY_SCORES)}
    evidence = []
    for source_id in LEGACY_IDS:
        label = labels[source_id]
        audit = audits[source_id]
        if label["revision"]["y"] != 0 or not label["revision"]["reason_code"].startswith("fabricated_"):
            raise ValueError(f"Historical evidence must be a confirmed fabrication: {source_id}")
        evidence.append({
            "source_id": source_id, "source": "audited historical API-Bank trajectory",
            "requirement": audit["requirement"], "action": label["action"],
            "confidence": scores.get((label["task_id"], label["step_id"])),
            "feedback": label["revision"]["reason"],
            "evidence": label["revision"]["evidence"],
            "tool_fields": list(audit["api_document"]["input_parameters"]),
        })
    return evidence


def collect_decision(case, requests, count_tokens, settings, request_id, rules=(), replica_index=0):
    send = requests.send if replica_index == 0 else partial(requests.send, replica_index=replica_index)
    decision = run_next_step(
        case, send=send, count_prompt_tokens=count_tokens, config=settings,
        request_id=request_id, rules=rules,
    )
    action = decision["action"]
    evaluation = evaluate_action(case, action)
    return {
        **decision, "task_id": case["task_id"], "base_task_id": case["base_task_id"],
        "requirement": case["requirement"], "target_tool": case["target_tool"],
        "api_documents": case["api_documents"], "history": case["history"],
        "parameter_sources": case["parameter_sources"], "expected": case["expected"],
        "provenance": case["provenance"], "evaluation": evaluation,
        "mock_observation": simulate_action(case, action),
    }


def format_evidence(development, historical):
    evidence = list(historical)
    for row in development:
        evidence.append({
            "source_id": row["task_id"], "source": "controlled development task",
            "requirement": row["requirement"], "history": row["history"],
            "action": row["action"], "confidence": row["confidence"]["self_report"],
            "tool_fields": {doc["name"]: list(doc["input_parameters"])
                            for doc in row["api_documents"]},
            "parameter_sources": row["parameter_sources"],
            "expected": row["expected"], "feedback": row["evaluation"],
        })
    # Scores only prioritize review. Confirmed errors and all dev examples are retained.
    return sorted(evidence, key=lambda row: -(row["confidence"] if row["confidence"] is not None else -1))


def parse_rules(response, source_ids):
    choice = response["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ValueError("Memory generation hit its token limit")
    content = (choice["message"].get("content") or "").strip()
    if content.startswith("```") and content.endswith("```"):
        opening, _, body = content.partition("\n")
        if opening in ("```", "```json"):
            content = body[:-3].strip()
    document = json.loads(content)
    if not isinstance(document, dict) or set(document) != {"rules"}:
        raise ValueError("Expected a JSON object containing only rules")
    rules = document["rules"]
    if not isinstance(rules, list) or not 1 <= len(rules) <= RULE_LIMIT:
        raise ValueError("Expected one to three memory rules")
    all_ids = set()
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) != {"rule_id", "applicability", "handling", "source_ids"}:
            raise ValueError("Invalid rule fields")
        rule_id = rule["rule_id"]
        if (not isinstance(rule_id, str) or not rule_id.strip() or len(rule_id) > 64
                or rule_id in all_ids):
            raise ValueError("Missing, duplicate, or oversized rule_id")
        all_ids.add(rule_id)
        handling = rule["handling"]
        if not isinstance(handling, str) or not handling.strip() or len(handling) > 900:
            raise ValueError("Expected a concise, nonempty rule")
        applicable = rule["applicability"]
        if applicable != {"problem_type": PROBLEM_TYPE, "tools": [], "required_fields": []}:
            raise ValueError("Invalid applicability")
        cites = rule["source_ids"]
        if (not isinstance(cites, list) or not cites
                or not all(isinstance(value, str) and value in source_ids for value in cites)
                or len(set(cites)) != len(cites)):
            raise ValueError("Each rule must cite distinct supporting evidence")
    return rules


def build_memory(output, cases, requests, count_tokens, settings, historical):
    development = []
    for case in cases["development"]:
        request_id = f"development:{case['task_id']}"
        row = collect_decision(case, requests, count_tokens, settings, request_id)
        confidence = self_report_for_action(
            case, row["action"], send=requests.send, count_prompt_tokens=count_tokens,
            config=settings, request_id=f"{request_id}:confidence",
        )
        development.append({**row, "confidence": confidence})
    write_json(output / "development.json", development)
    evidence = format_evidence(development, historical)
    write_json(output / "memory_sources.json", evidence)
    source_ids = {row["source_id"] for row in evidence}
    write_json(output / "memory_M0.json", {"version": "M0", "rules": []})
    attempts = []
    for number in range(1, SYNTHESIS_ATTEMPTS + 1):
        prompt = SYNTHESIS_PROMPT
        if number > 1:
            prompt += "\nA previous candidate failed to pass the acceptance gate. Generate a fresh candidate from the same development evidence."
        budget = RequestBudget(SYNTHESIS_TOKEN_CAP, requests.send, count_tokens)
        payload = {
            "model": settings.model,
            "messages": [{"role": "system", "content": prompt},
                         {"role": "user", "content": json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))}],
            "temperature": settings.temperature, "top_p": settings.top_p,
            "seed": settings.seed, "max_tokens": SYNTHESIS_OUTPUT_CAP,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        response = budget.complete(f"synthesis:{number}", "memory_synthesis", payload)
        attempt = {"attempt": number, "cost": budget.cost, "validation": [], "rule_coverage": {}}
        attempts.append(attempt)
        try:
            rules = parse_rules(response, source_ids)
        except ValueError as exc:
            attempt.update(status="invalid_rules", error=str(exc))
        else:
            attempt["rules"] = rules
            attempt["validation"] = [
                collect_decision(case, requests, count_tokens, settings,
                                 f"validation:{number}:{case['task_id']}", rules)
                for case in cases["validation"]
            ]
            attempt["rule_coverage"] = {
                rule["rule_id"]: [row["task_id"] for row in attempt["validation"]
                                  if rule["rule_id"] in row["memory_rule_ids"]]
                for rule in rules
            }
            passed = (all(row["evaluation"]["correct"] for row in attempt["validation"])
                      and all(attempt["rule_coverage"].values()))
            attempt["status"] = "passed" if passed else "failed_validation"
        write_json(output / "memory_attempts.json", attempts)
        validation_passes = sum(row["evaluation"]["correct"] for row in attempt["validation"])
        print(f"Memory candidate {number}: {attempt['status']} ({validation_passes}/4)", flush=True)
        if attempt["status"] == "passed":
            memory = {
                "version": "M1", "rules": attempt["rules"], "rules_sha256": digest(attempt["rules"]),
                "source_sha256": digest(evidence), "accepted_attempt": number,
                "validation_task_ids": [row["task_id"] for row in attempt["validation"]],
            }
            require_frozen(output / "memory_M1.json", memory)
            return memory, development, attempts
    if (output / "memory_M1.json").exists():
        raise ValueError("Stored M1 conflicts with reconstructed memory validation")
    return None, development, attempts


def plan_pairs(test_sets, replica_count):
    plan = []
    for dataset, cases in test_sets.items():
        seen = {"clarify": 0, "call": 0}
        for case in cases:
            cohort = case["expected"]["kind"]
            index = seen[cohort]
            round_index, replica_index = divmod(index, replica_count)
            seen[cohort] += 1
            first_is_b = (round_index + replica_index + (cohort == "call")) % 2
            plan.append({"dataset": dataset, "task_id": case["task_id"], "replica_index": replica_index,
                         "arms": ["B", "A"] if first_is_b else ["A", "B"]})
    return plan


def collect_test(test_sets, requests, counter, settings, plan, memory, groups, replica_count):
    cases = {case["task_id"]: case for dataset in test_sets.values() for case in dataset}
    count_lock = Lock()

    def count_tokens(payload):
        with count_lock:
            return counter(payload)

    def run_replica(replica_index):
        pairs = {}
        for index, entry in enumerate(plan):
            if entry["replica_index"] != replica_index:
                continue
            case = cases[entry["task_id"]]
            pair = []
            for arm in entry["arms"]:
                rules = memory["rules"] if arm == "B" else ()
                row = collect_decision(case, requests, count_tokens, settings,
                                       f"test:{case['task_id']}:{arm}", rules, replica_index)
                pair.append({**row, "arm": arm, "memory_version": groups[arm],
                             "dataset": entry["dataset"], "replica_index": replica_index})
            pairs[index] = pair
            print(f"Test pair {index + 1}/{len(plan)}: {case['task_id']} (replica {replica_index})", flush=True)
        return pairs

    with ThreadPoolExecutor(max_workers=replica_count) as executor:
        pairs = {}
        for result in executor.map(run_replica, range(replica_count)):
            pairs.update(result)
    return [row for index in range(len(plan)) for row in pairs[index]]


def freeze_protocol(args, cases, stress, settings, historical, endpoints, plan):
    model_files = ["config.json", "generation_config.json", "tokenizer.json",
                   "tokenizer_config.json", "model.safetensors.index.json"]
    code_files = ("run_memory_experiment.py", "memory_agent.py", "memory_experiment_data.py",
                  "report_memory_experiment.py", "apibank_env.py", "react_agent.py",
                  "collect_new_signals.py", "step_confidence.py", "step_monitor.py",
                  "minimal_validation_data.py")
    protocol = {
        "experiment": "missing-parameter-memory-v2",
        "dataset_path": str(args.cases.resolve()), "dataset_sha256": sha(args.cases),
        "stress_dataset_path": str(args.stress_cases.resolve()), "stress_dataset_sha256": sha(args.stress_cases),
        "schema_corpora_sha256": {name: sha(path) for name, path in SCHEMA_CORPORA.items()},
        "resolved_cases_sha256": digest(cases),
        "resolved_stress_cases_sha256": digest(stress),
        "splits": {name: [row["task_id"] for row in rows] for name, rows in cases.items()},
        "historical_ids": list(LEGACY_IDS), "historical_sha256": digest(historical),
        "model_path": str(args.model_path.resolve()), "settings": asdict(settings),
        "model_metadata_sha256": {name: sha(args.model_path / name) for name in model_files},
        "code_sha256": {name: sha(ROOT / "code" / name) for name in code_files},
        "groups": {"A": "M0", "B": "M1"}, "rule_limit": RULE_LIMIT,
        "synthesis_attempt_limit": SYNTHESIS_ATTEMPTS, "required_validation_passes": 4,
        "synthesis_total_token_cap": SYNTHESIS_TOKEN_CAP, "synthesis_completion_cap": SYNTHESIS_OUTPUT_CAP,
        "required_rule_coverage": "every stored rule is injected into at least one validation prompt",
        "rule_applicability": {"problem_type": PROBLEM_TYPE, "tools": [], "required_fields": []},
        "test_plan": plan, "replica_endpoints": endpoints,
    }
    require_frozen(args.output / "protocol.json", protocol)
    return protocol


def run_experiment(args):
    import httpx
    from openai import OpenAI
    from memory_experiment_data import load_stress_cases

    cases = load_cases(args.cases)
    stress = load_stress_cases(args.stress_cases, existing_cases=cases)
    test_sets = {"original": cases["test"], "stress": stress}
    endpoints = [args.base_url, *args.replica_base_url]
    if len(endpoints) > 2 or len(endpoints) != len(set(endpoints)):
        raise ValueError("Supply one or two distinct replica endpoints")
    plan = plan_pairs(test_sets, len(endpoints))
    historical = historical_evidence()
    settings = AgentSettings()
    protocol = freeze_protocol(args, cases, stress, settings, historical, endpoints, plan)
    if args.prepare_only:
        print(f"Frozen data: {args.output / 'protocol.json'}")
        return
    counter = QwenPromptCounter(args.model_path)
    with ExitStack() as stack:
        clients, serving = [], []
        for endpoint in endpoints:
            client = stack.enter_context(OpenAI(base_url=endpoint, api_key=os.getenv("OPENAI_API_KEY") or "EMPTY",
                                               http_client=httpx.Client(trust_env=False, timeout=300), max_retries=0))
            served = next((entry for entry in client.models.list().data if entry.id == settings.model), None)
            if served is None or not served.root or Path(served.root).resolve() != args.model_path.resolve():
                raise ValueError("Every replica must load the checkpoint used by the local token counter")
            context = (served.model_extra or {}).get("max_model_len")
            if context != SYNTHESIS_TOKEN_CAP:
                raise ValueError(f"Replica {endpoint} must use the frozen {SYNTHESIS_TOKEN_CAP}-token context")
            serving.append({"id": served.id, "checkpoint": str(Path(served.root).resolve()),
                            "max_model_len": context, "endpoint": endpoint})
            clients.append(client)
        require_frozen(args.output / "serving_model.json", {"replicas": serving})
        requests = LoggedRequests(clients, args.output)
        memory, development, attempts = build_memory(args.output, cases, requests, counter, settings, historical)
        results = []
        if memory is not None:
            results = collect_test(test_sets, requests, counter, settings, plan, memory,
                                   protocol["groups"], len(clients))
            write_json(args.output / "test_results.json", results)
        elif (args.output / "test_results.json").exists():
            raise ValueError("Stored test results require a validated M1")
        write_report(args.output, protocol, results, memory, development, attempts, requests.failed)


def main():
    from memory_experiment_data import DEFAULT_STRESS_CASES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--stress-cases", type=Path, default=DEFAULT_STRESS_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--replica-base-url", action="append", default=[], help="Optional second inference replica")
    parser.add_argument("--prepare-only", action="store_true", help="Freeze inputs without requesting the model")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"An experiment is already running in {args.output}")
        run_experiment(args)


if __name__ == "__main__":
    sys.exit(main())
