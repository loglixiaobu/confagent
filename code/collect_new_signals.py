#!/usr/bin/env python
"""Probe fixed legacy actions at reconstructed pre-call states; never execute tools.

Labels and current observations are deliberately absent from states.jsonl and
model requests. Raw requests/responses are checkpointed for restart and audit.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import time
from pathlib import Path

from apibank_env import TYPE_MAP, _norm_time, arithmetic_value, load_level3, typed_equal
# Frozen original prompt copied locally so CPU preparation does not import a model client.
SYSTEM_PROMPT = """You are an AI assistant that completes a user's requirement by calling APIs step by step.

Rules:
1. At each step, call EXACTLY ONE tool.
2. The available APIs are listed in your tool set. If you need an API that is not listed, \
call `ToolSearcher` with keywords describing the capability you need.
3. Before each call, verify every argument value comes from the requirement or from a previous \
API output. Never invent values.
4. If a call fails, READ the error message carefully and change the arguments before retrying. \
Do not repeat the same failing call.
5. Only when EVERY part of the requirement has been carried out, reply with a final answer \
and DO NOT call any tool. Before finishing, re-read the requirement clause by clause and \
confirm that each requested action has actually been executed by an API call.

Be concise in your final answer."""

ROOT = Path(__file__).resolve().parents[1]
VERSION = "legacy-reconstructed-signals-v1"
RETRY_PROMPT = ("The previous call failed with the same arguments. Read the error message again "
                "and change the arguments before retrying.")
SELF_PROMPT = ("On a scale from 0.0 to 1.0, how confident are you that this "
               "action is CORRECT (right API, right arguments)? Reply with ONLY a number.")
SEARCH_TOOL = {"type": "function", "function": {"name": "ToolSearcher", "description":
    "Search for a suitable API by keywords. Returns the API name, description and its input parameters.",
    "parameters": {"type": "object", "properties": {"keywords": {"type": "string", "description":
    "Keywords describing the capability you need."}}, "required": ["keywords"]}}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append(path, row):
    with Path(path).open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def legacy_tool(api):
    # Preserve the schema actually used in the original run, including its
    # list(str)->string defect; this experiment does not repair the old states.
    props = {name: {"type": TYPE_MAP.get(str(meta.get("type", "str")).lower(), "string"),
                    "description": meta.get("description", "")}
             for name, meta in api.get("input_parameters", {}).items()}
    return {"type": "function", "function": {"name": api["name"],
            "description": api.get("description", ""), "parameters": {
            "type": "object", "properties": props, "required": list(props)}}}


def reconstruct(trajectories, tasks):
    states = []
    for trajectory in trajectories:
        task_id = trajectory["task_id"]
        task = tasks[int(task_id.split("-")[1])]
        docs = {}
        for step in task["apis"]:
            if step.get("api_name") == "ToolSearcher":
                document = (step.get("output") or {}).get("output")
                if isinstance(document, dict) and document.get("name"):
                    docs[document["name"]] = document
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "Requirement: " + task["requirement"].strip()}]
        last_sig, repeat_count = None, 0
        for step in trajectory["steps"]:
            action = step["action_input"]
            states.append({"task_id": task_id, "step_id": step["step_id"],
                "candidate": copy.deepcopy(action), "messages": copy.deepcopy(messages),
                "tools": [copy.deepcopy(SEARCH_TOOL)] + [legacy_tool(api) for api in docs.values()],
                "api_documents": copy.deepcopy(list(docs.values()))})
            sig = action["api_name"] + "|" + json.dumps(action["arguments"], sort_keys=True, ensure_ascii=False)
            repeat_count = repeat_count + 1 if sig == last_sig else 0
            last_sig = sig
            call_id = f"reconstructed_{task_id}_{step['step_id']}"
            messages.extend([{"role": "assistant", "content": "", "tool_calls": [{"id": call_id,
                "type": "function", "function": {"name": action["api_name"],
                "arguments": json.dumps(action["arguments"], ensure_ascii=False)}}]},
                {"role": "tool", "tool_call_id": call_id,
                 "content": json.dumps(step["observation"], ensure_ascii=False, default=str)[:4000]}])
            if action["api_name"] == "ToolSearcher" and isinstance(step["observation"], dict) and step["observation"].get("name"):
                docs[step["observation"]["name"]] = step["observation"]
            if repeat_count == 1:
                messages.append({"role": "user", "content": RETRY_PROMPT})
    return states


def validate_states(states):
    """Validate prebuilt, label-free action states before any model request."""
    if not states:
        raise ValueError("State input must contain at least one state")
    seen = set()
    required = {"task_id", "step_id", "candidate", "messages", "tools", "api_documents"}
    for state in states:
        if not isinstance(state, dict) or not required.issubset(state):
            raise ValueError("Each state must contain task_id, step_id, candidate, messages, tools, api_documents")
        if (not isinstance(state["task_id"], str) or not state["task_id"]
                or type(state["step_id"]) is not int or state["step_id"] < 0):
            raise ValueError("State task_id must be nonempty and step_id must be a nonnegative integer")
        key = state["task_id"], state["step_id"]
        if key in seen:
            raise ValueError("Duplicate task_id/step_id in states")
        seen.add(key)
        candidate = state["candidate"]
        if (not isinstance(candidate, dict) or not isinstance(candidate.get("api_name"), str)
                or not candidate["api_name"] or not isinstance(candidate.get("arguments"), dict)):
            raise ValueError("State candidate must contain an api_name and arguments object")
        for field in ("messages", "tools", "api_documents"):
            if not isinstance(state[field], list) or not all(isinstance(item, dict) for item in state[field]):
                raise ValueError("State messages, tools and api_documents must be lists of objects")
        if not state["messages"] or state["messages"][0].get("role") != "system":
            raise ValueError("State messages must begin with a system prompt")
    return states


def prepare(out, k, consistency_thinking=True, source=None, data_dir=None, states_file=None):
    out.mkdir(parents=True, exist_ok=True)
    source = Path(source) if source is not None else ROOT / "results/traj_main_clean.jsonl"
    data_dir = Path(data_dir) if data_dir is not None else ROOT / "data/api-bank"
    data = data_dir / "test-data/level-3.json"
    if states_file is not None:
        states_file = Path(states_file)
        if not states_file.is_file():
            raise FileNotFoundError("Prepared states do not exist; supply an existing --states file")
        states = validate_states(read_jsonl(states_file))
        source_hash, data_hash = None, None
    else:
        if not source.is_file():
            raise FileNotFoundError("Trajectory source does not exist; supply --source")
        states = validate_states(reconstruct(read_jsonl(source), load_level3(str(data_dir))))
        source_hash, data_hash = sha(source), sha(data)
    protocol = {"version": VERSION, "source_sha256": source_hash, "data_sha256": data_hash,
        "collector_sha256": sha(__file__), "states_sha256": digest(states), "n_states": len(states),
        "model": "Qwen3-8B", "consistency_k": k, "consistency_temperature": 0.7,
        "consistency_top_p": 1.0, "consistency_thinking": consistency_thinking, "max_tokens": 4096,
        "self_report_temperature": 0, "self_report_thinking": False, "self_report_prompt": SELF_PROMPT,
        "seed": 20261015, "new_tool_executions": 0, "labels_used_for_requests": False,
        "context": "reconstructed pre-call; missing assistant prose and original call IDs/argument formatting",
        "tool_schema": "original legacy schema (including list(str) exposed as string)",
        "consistency_definition": "fraction of K independent samples agreeing with fixed original action; semantic argument normalization; EmailReminder.content ignored, structured fields retained",
        "strict_consistency_definition": "all arguments JSON value equality, ignoring object key order",
        "transport_failure_policy": "retry once with same seed; missing samples make primary consistency null",
        "no_tool_or_truncation_policy": "valid model responses without parsed call count as disagreement; separately report truncation",
        "calibrator_fitted": False}
    if states_file is not None:
        protocol["source_mode"] = "prebuilt_states"
        protocol["prepared_states_file_sha256"] = sha(states_file)
    target = out / "protocol.json"
    if target.exists() and json.loads(target.read_text()) != protocol:
        raise ValueError("Frozen protocol changed; use a new output directory")
    write_json(target, protocol)
    serialized = "".join(json.dumps(state, ensure_ascii=False) + "\n" for state in states)
    states_path = out / "states.jsonl"
    if states_path.exists() and states_path.read_text() != serialized:
        raise ValueError("Refusing to overwrite changed states")
    states_path.write_text(serialized)
    return states, protocol


def self_messages(state):
    messages = copy.deepcopy(state["messages"])
    # Tools are text here so the model must give a score rather than call an API.
    messages[0]["content"] += "\n\nAvailable API documentation:\n" + json.dumps(state["tools"], ensure_ascii=False)
    action = state["candidate"]
    messages.extend([{"role": "assistant", "content": "I am about to execute: %s(%s)" % (
        action["api_name"], json.dumps(action["arguments"], ensure_ascii=False))},
        {"role": "user", "content": SELF_PROMPT}])
    return messages


def parse_self(content):
    content = content.strip()
    if content.startswith("```") and content.endswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content[:-3]).strip()
    match = re.fullmatch(r"([01](?:\.\d+)?|\.\d+)", content)
    if match:
        value = float(match[1])
        if math.isfinite(value) and 0 <= value <= 1:
            return value
    return None


def equivalent(candidate, sampled, docs):
    if not sampled or sampled.get("api_name") != candidate["api_name"]:
        return False
    a, b = candidate.get("arguments", {}), sampled.get("arguments", {})
    if not isinstance(a, dict) or not isinstance(b, dict) or set(a) != set(b):
        return False
    api = next((doc for doc in docs if doc["name"] == candidate["api_name"]), {})
    metadata = api.get("input_parameters", {})
    for key in a:
        # For reminders, the event identity is recipient/time/location. Free-form
        # email wording is excluded explicitly; strict consistency keeps it.
        if candidate["api_name"] == "EmailReminder" and key == "content":
            if not isinstance(b[key], str) or not b[key].strip():
                return False
            continue
        if candidate["api_name"] == "Calculator" and key == "formula":
            try:
                if arithmetic_value(a[key]) == arithmetic_value(b[key]):
                    continue
            except (ValueError, SyntaxError, ArithmeticError):
                pass
        if candidate["api_name"] == "PersonalInfoUpdate" and key == "address" and isinstance(a[key], str) and isinstance(b[key], str):
            if a[key].strip().rstrip(".") == b[key].strip().rstrip("."):
                continue
        kind = metadata.get(key, {}).get("type", "str")
        if not typed_equal(_norm_time(a[key]), _norm_time(b[key]), kind):
            return False
    return True


def sample_action(response):
    calls = response["choices"][0]["message"].get("tool_calls") or []
    if not calls:
        return None
    function = calls[0]["function"]
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except (TypeError, ValueError):
        return None
    return {"api_name": function["name"], "arguments": arguments}


class Guard:
    """Check an optional external health gate before each sequential request.

    A health file must contain ``status: "healthy"``, a real boolean
    ``can_submit`` and a numeric Unix ``updated_at`` no older than 12 seconds.
    The inference endpoint operator owns resource management and this file.
    """
    def __init__(self, thermal_health_file=None, max_age=12):
        self.health_file = Path(thermal_health_file) if thermal_health_file else None
        self.max_age = max_age

    def wait(self):
        if self.health_file is None:
            return
        while True:
            try:
                state = json.loads(self.health_file.read_text())
            except (OSError, ValueError) as exc:
                raise RuntimeError("Health gate file is missing or invalid") from exc
            if not isinstance(state, dict):
                raise RuntimeError("Health gate must be a JSON object")
            updated = state.get("updated_at")
            if (type(updated) not in (int, float) or not math.isfinite(updated)
                    or not 0 <= time.time() - updated <= self.max_age):
                raise RuntimeError("Health gate is stale or has an invalid timestamp")
            if state.get("status") != "healthy" or type(state.get("can_submit")) is not bool:
                raise RuntimeError("Health gate must be healthy with boolean can_submit")
            if state["can_submit"]:
                return
            print("[waiting for health gate]", flush=True)
            time.sleep(1)


def collect(out, states, protocol, limit, cooldown, endpoints=None, thermal_health_file=None):
    # Requests are globally sequential, with a cooldown after every attempt.
    import httpx
    from openai import OpenAI

    endpoints = endpoints or ["http://localhost:8000/v1"]
    if not math.isfinite(cooldown) or cooldown < 1 or not endpoints or any(not endpoint.strip() for endpoint in endpoints):
        raise ValueError("At least one endpoint and cooldown>=1 are required")
    clients = [OpenAI(base_url=endpoint, api_key="EMPTY", max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=900)) for endpoint in endpoints]
    guard = Guard(thermal_health_file)
    cache = {row["request_id"]: row for row in read_jsonl(out / "requests.jsonl") if row.get("response")}
    selected = sorted(states, key=lambda x: (x["step_id"], digest([x["task_id"], x["step_id"]])))
    # Interleave trajectory depths; the frozen order is score/label-blind.
    if limit:
        selected = selected[:limit]
    num_calls = len(cache)
    started = time.time()
    results = []
    for i, state in enumerate(selected):
        row = {"task_id": state["task_id"], "step_id": state["step_id"],
            "consistency_k": protocol["consistency_k"], "self_report": None,
            "consistency": None, "strict_consistency": None, "n_samples": 0,
            "n_agree": 0, "n_strict_agree": 0, "n_no_tool": 0, "n_truncated": 0,
            "state_sha256": digest(state), "request_ids": []}
        for kind in ["self_report"] + [f"sample_{j}" for j in range(protocol["consistency_k"])]:
            request_id = f"{state['task_id']}:{state['step_id']}:{kind}"
            is_self = kind == "self_report"
            seed = 20261015 + int(digest(request_id)[:7], 16)
            kwargs = {"model": protocol["model"], "messages": self_messages(state) if is_self else state["messages"],
                "temperature": 0.0 if is_self else 0.7, "top_p": 1.0,
                "max_tokens": 64 if is_self else 4096, "seed": seed,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False if is_self else protocol["consistency_thinking"]}}}
            if not is_self:
                kwargs.update(tools=state["tools"], tool_choice="auto")
            row["request_ids"].append(request_id)
            record = cache.get(request_id)
            if record and record["payload_sha256"] != digest(kwargs):
                raise ValueError("Cached request payload changed")
            if not record:
                for attempt in range(2):
                    guard.wait()
                    t0 = time.time()
                    record = {"request_id": request_id, "kind": kind, "attempt": attempt,
                        "payload_sha256": digest(kwargs), "started_at": t0,
                        "endpoint_index": num_calls % len(clients), "request": kwargs}
                    try:
                        response = clients[num_calls % len(clients)].chat.completions.create(**kwargs)
                        record["response"] = response.model_dump()
                    except Exception as exc:
                        record["error"] = type(exc).__name__ + ": " + str(exc)[:600]
                    finally:
                        num_calls += 1
                        record["latency_s"] = time.time() - t0
                        append(out / "requests.jsonl", record)
                        time.sleep(cooldown)
                    if record.get("response"):
                        cache[request_id] = record
                        break
                if not record.get("response"):
                    print("[failed request]", request_id, record.get("error"), flush=True)
                    continue
            response = record["response"]
            if is_self:
                row["self_report"] = parse_self(response["choices"][0]["message"].get("content") or "")
            else:
                action = sample_action(response)
                row["n_samples"] += 1
                row["n_agree"] += int(equivalent(state["candidate"], action, state["api_documents"]))
                row["n_strict_agree"] += int(action == state["candidate"])
                row["n_no_tool"] += int(action is None)
                row["n_truncated"] += int(response["choices"][0]["finish_reason"] == "length")
        if row["n_samples"] == protocol["consistency_k"]:
            row["consistency"] = row["n_agree"] / row["n_samples"]
            row["strict_consistency"] = row["n_strict_agree"] / row["n_samples"]
        results.append(row)
        # Atomic upsert of scores; raw requests are the durable recovery source.
        tmp = out / "scores.jsonl.tmp"
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results))
        tmp.replace(out / "scores.jsonl")
        print(json.dumps({"progress": f"{i+1}/{len(selected)}", "task_id": row["task_id"],
            "step_id": row["step_id"], "self_report": row["self_report"], "consistency": row["consistency"],
            "n_truncated": row["n_truncated"], "elapsed_s": round(time.time()-started)}, ensure_ascii=False), flush=True)
    for client in clients:
        client.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/new_signals_fast_v1")
    ap.add_argument("--states", type=Path,
                    help="Reuse prepared states without the complete API-Bank dataset; use a new --out-dir")
    ap.add_argument("--source", type=Path, default=ROOT / "results/traj_main_clean.jsonl")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/api-bank")
    ap.add_argument("--endpoints", default="http://localhost:8000/v1",
                    help="Comma-separated OpenAI-compatible base URLs; requests remain sequential")
    ap.add_argument("--thermal-health-file", type=Path,
                    help="Optional external health gate; see Guard for its JSON format")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cooldown", type=float, default=3.0)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--no-consistency-thinking", action="store_true",
                    help="Fast screening: sample a no-thinking policy; keep separate from the original thinking-policy experiment")
    args = ap.parse_args()
    if args.k < 1 or not math.isfinite(args.cooldown) or args.cooldown < 1 or args.limit < 0:
        ap.error("k>=1, cooldown>=1, limit>=0 required")
    endpoints = [endpoint.strip() for endpoint in args.endpoints.split(",")]
    if any(not endpoint for endpoint in endpoints):
        ap.error("--endpoints must contain nonempty comma-separated base URLs")
    states, protocol = prepare(args.out_dir, args.k, not args.no_consistency_thinking,
                               source=args.source, data_dir=args.data_dir, states_file=args.states)
    print(json.dumps({"states": len(states), "k": args.k, "new_tool_executions": 0}), flush=True)
    if not args.prepare_only:
        collect(args.out_dir, states, protocol, args.limit, args.cooldown,
                endpoints=endpoints, thermal_health_file=args.thermal_health_file)


if __name__ == "__main__":
    main()
