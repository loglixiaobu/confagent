#!/usr/bin/env python
"""Estimate fixed-action confidence without executing any tool.

The sampler receives only the complete pre-call messages/tools supplied by the
caller. The proposed candidate is compared locally and is never appended to a
sampling request. Three no-thinking samples feed a frozen one-signal sigmoid.
Model/client imports are lazy; probability mapping and CPU tests need no model.
"""
from __future__ import annotations

import argparse
import copy
import json
import hashlib
import math
import threading
import time
from collections.abc import Mapping
from pathlib import Path

from apibank_env import _norm_time, arithmetic_value, typed_equal

MODEL = "Qwen3-8B"
K = 3
TEMPERATURE = 0.7
TOP_P = 1.0
MAX_TOKENS = 4096
REQUEST_LOCK = threading.Lock()

# These three helpers mirror the frozen collector exactly. Tests check AST
# identity so semantic agreement does not drift between offline and online use.
def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

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


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def load_calibrator(source):
    """Load and validate the frozen, unstandardized single-signal sigmoid."""
    value = (copy.deepcopy(dict(source)) if isinstance(source, Mapping)
             else json.loads(Path(source).read_text(encoding="utf-8")))
    if not isinstance(value, dict):
        raise ValueError("Calibrator must be a JSON object")
    expected = {"version": "consistency-platt-v1", "signal": "consistency",
                "link": "sigmoid", "score_direction": "higher_is_more_correct"}
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            raise ValueError("Unsupported calibrator " + field)
    for field in ("coefficient", "intercept"):
        if not _finite_number(value.get(field)):
            raise ValueError("Calibrator " + field + " must be finite")
    if value["coefficient"] < 0:
        raise ValueError("Calibrator coefficient cannot reverse score direction")
    input_range = value.get("input_range")
    if (not isinstance(input_range, list) or len(input_range) != 2
            or not all(_finite_number(number) for number in input_range)
            or input_range != [0, 1]):
        raise ValueError("Calibrator input_range must be [0, 1]")
    rule = value.get("decision_rule")
    if (not isinstance(rule, dict) or rule.get("signal") != "consistency"
            or rule.get("operator") != "<" or not _finite_number(rule.get("threshold"))
            or rule["threshold"] != 1):
        raise ValueError("Calibrator decision rule must be consistency < 1")
    protocol = value.get("collection_protocol")
    if (not isinstance(protocol, dict) or type(protocol.get("consistency_k")) is not int
            or protocol["consistency_k"] != K or protocol.get("consistency_thinking") is not False
            or not _finite_number(protocol.get("consistency_temperature"))
            or protocol["consistency_temperature"] != TEMPERATURE
            or not _finite_number(protocol.get("consistency_top_p"))
            or protocol["consistency_top_p"] != TOP_P
            or type(protocol.get("max_tokens")) is not int or protocol["max_tokens"] != MAX_TOKENS):
        raise ValueError("Calibrator must match the fixed K=3 no-thinking sampling protocol")
    return value


def calibrate_consistency(raw_consistency, calibrator):
    """Map raw agreement to a probability; missing scores never become zero."""
    calibration = load_calibrator(calibrator)
    if raw_consistency is None:
        return {"raw_consistency": None, "p_correct": None, "needs_verification": True,
                "reason": "score_missing", "calibrator_version": calibration["version"]}
    if not _finite_number(raw_consistency) or not 0 <= raw_consistency <= 1:
        raise ValueError("raw_consistency must be a finite number in [0, 1] or null")
    z = calibration["intercept"] + calibration["coefficient"] * raw_consistency
    # This form stays finite even for very large positive/negative logits.
    p_correct = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
    needs_verification = raw_consistency < calibration["decision_rule"]["threshold"]
    return {"raw_consistency": float(raw_consistency), "p_correct": p_correct,
            "needs_verification": needs_verification,
            "reason": "inconsistent_samples" if needs_verification else "all_samples_agree",
            "calibrator_version": calibration["version"]}


def _response_dict(response):
    if hasattr(response, "model_dump"):
        response = response.model_dump()
    if not isinstance(response, dict):
        raise ValueError("Model response must be a mapping")
    choices = response.get("choices")
    if (not isinstance(choices, list) or not choices or not isinstance(choices[0], dict)
            or not isinstance(choices[0].get("message"), dict)):
        raise ValueError("Model response is missing a first choice/message")
    calls = choices[0]["message"].get("tool_calls") or []
    if not isinstance(calls, list) or any(not isinstance(call, dict) for call in calls):
        raise ValueError("Model response tool_calls must be a list of mappings")
    if calls and (not isinstance(calls[0].get("function"), dict)
                  or not isinstance(calls[0]["function"].get("name"), str)):
        raise ValueError("First tool call is missing its function name")
    return response


def estimate_step_confidence(messages, tools, candidate, api_documents, calibrator, *,
                             client=None, request_fn=None, before_request=None,
                             after_request=None, request_namespace="", seed=20261019,
                             base_url="http://localhost:8000/v1", cooldown_s=1.0):
    """Return a step confidence from three independent pre-call samples.

    ``request_fn(**payload)`` may replace the OpenAI client and perform logging.
    ``before_request()`` and ``after_request(attempt_record)`` may block for a
    caller-owned resource gate. False returns or callback exceptions abort the
    remaining sampling. All requests in this process hold one shared lock and
    cool down for at least one second. External processes need their own gate.

    Transport failures retry once using exactly the same seed. Missing or
    invalid responses make confidence null. A valid response without a parsed
    tool call counts as disagreement. Multiple calls use the first call, as in
    the original collector, and quality flags retain the discarded-call count.
    This function never executes sampled actions or the supplied candidate.
    """
    calibration = load_calibrator(calibrator)
    if not _finite_number(cooldown_s) or cooldown_s < 1:
        raise ValueError("cooldown_s must be finite and at least 1 second")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(request_namespace, str):
        raise ValueError("request_namespace must be a string")
    if not isinstance(messages, list) or not messages or not all(isinstance(row, dict) for row in messages):
        raise ValueError("messages must be a nonempty list of message mappings")
    if not isinstance(tools, list) or not all(isinstance(row, dict) for row in tools):
        raise ValueError("tools must be a list of tool mappings")
    if (not isinstance(candidate, dict) or not isinstance(candidate.get("api_name"), str)
            or not isinstance(candidate.get("arguments"), dict)):
        raise ValueError("candidate must contain api_name and arguments")
    if not isinstance(api_documents, list) or not all(isinstance(row, dict) for row in api_documents):
        raise ValueError("api_documents must be a list of API document mappings")
    if client is not None and request_fn is not None:
        raise ValueError("Supply client or request_fn, not both")
    owned_client = None
    if request_fn is None:
        if client is None:
            import httpx
            from openai import OpenAI
            owned_client = OpenAI(base_url=base_url, api_key="EMPTY", max_retries=0,
                                  http_client=httpx.Client(trust_env=False, timeout=900))
            client = owned_client
        request_fn = client.chat.completions.create
    namespace = request_namespace or digest({"messages": messages, "tools": tools})[:24]
    started = time.monotonic()
    samples, abort = [], False
    try:
        for index in range(K):
            request_id = f"{namespace}:sample_{index}"
            sample_seed = seed + int(digest(request_id)[:7], 16)
            payload = {"model": MODEL, "messages": copy.deepcopy(messages),
                       "tools": copy.deepcopy(tools), "tool_choice": "auto",
                       "temperature": TEMPERATURE, "top_p": TOP_P,
                       "max_tokens": MAX_TOKENS, "seed": sample_seed,
                       "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
            sample = {"index": index, "request_id": request_id, "seed": sample_seed,
                      "request": payload, "payload_sha256": digest(payload), "attempts": [],
                      "response": None, "action": None, "agrees": None,
                      "strict_agrees": None, "finish_reason": None, "n_tool_calls": 0,
                      "truncated": False, "multiple_tool_calls": False,
                      "invalid_tool_arguments": False, "status": "not_attempted"}
            samples.append(sample)
            if abort:
                sample["status"] = "gate_blocked"
                continue
            for attempt in range(2):
                record = {"attempt": attempt, "response": None, "error": None,
                          "failure_kind": None, "request_sent": False, "latency_s": 0.0}
                sample["attempts"].append(record)
                with REQUEST_LOCK:
                    try:
                        if before_request is not None and before_request() is False:
                            raise RuntimeError("before_request resource gate declined request")
                    except Exception as exc:
                        record.update(error=type(exc).__name__ + ": " + str(exc), failure_kind="gate")
                        sample["status"], abort = "gate_blocked", True
                        break
                    request_start = time.monotonic()
                    try:
                        record["request_sent"] = True
                        response = request_fn(**copy.deepcopy(payload))
                        # Retain even malformed mapping responses for an audit.
                        if hasattr(response, "model_dump"):
                            response = response.model_dump()
                        record["response"] = copy.deepcopy(response) if isinstance(response, dict) else None
                        try:
                            response = _response_dict(response)
                        except (ValueError, TypeError) as exc:
                            record.update(error=str(exc), failure_kind="invalid_response")
                            sample["status"] = "invalid_response"
                        else:
                            sample["response"] = response
                            sample["status"] = "ok"
                    except Exception as exc:
                        record.update(error=type(exc).__name__ + ": " + str(exc), failure_kind="transport")
                        sample["status"] = "transport_missing"
                    finally:
                        record["latency_s"] = time.monotonic() - request_start
                        if after_request is not None:
                            try:
                                if after_request(record) is False:
                                    raise RuntimeError("after_request resource gate declined further requests")
                            except Exception as exc:
                                record["after_request_error"] = type(exc).__name__ + ": " + str(exc)
                                abort = True
                        time.sleep(cooldown_s)
                if sample["response"] is not None or abort:
                    break
            if sample["response"] is not None:
                response = sample["response"]
                choice = response["choices"][0]
                calls = choice["message"].get("tool_calls") or []
                action = sample_action(response)
                sample.update(action=action, agrees=bool(equivalent(candidate, action, api_documents)),
                              strict_agrees=bool(action == candidate), finish_reason=choice.get("finish_reason"),
                              n_tool_calls=len(calls), truncated=choice.get("finish_reason") == "length",
                              multiple_tool_calls=len(calls) > 1,
                              invalid_tool_arguments=bool(calls and (action is None or not isinstance(action.get("arguments"), dict))))
    finally:
        if owned_client is not None:
            owned_client.close()
    usable = [sample for sample in samples if sample["status"] == "ok"]
    raw = sum(sample["agrees"] for sample in usable) / K if len(usable) == K else None
    result = calibrate_consistency(raw, calibration)
    attempts = [attempt for sample in samples for attempt in sample["attempts"]]
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for attempt in attempts:
        token_usage = (attempt.get("response") or {}).get("usage") or {}
        for key in usage:
            number = token_usage.get(key)
            if type(number) is int and number >= 0:
                usage[key] += number
    result["samples"] = samples
    result["cost"] = {"n_samples_requested": K, "n_model_requests": sum(row["request_sent"] for row in attempts),
                      "usage": usage, "elapsed_s": time.monotonic() - started}
    result["quality"] = {
        "complete": len(usable) == K, "n_samples": len(usable),
        "n_agree": sum(sample["agrees"] for sample in usable),
        "n_no_tool": sum(sample["action"] is None for sample in usable),
        "n_truncated": sum(sample["truncated"] for sample in usable),
        "n_multiple_tool_calls": sum(sample["multiple_tool_calls"] for sample in usable),
        "n_invalid_tool_arguments": sum(sample["invalid_tool_arguments"] for sample in usable),
        "n_transport_failures": sum(row["failure_kind"] == "transport" for row in attempts),
        "n_invalid_responses": sum(row["failure_kind"] == "invalid_response" for row in attempts),
        "n_gate_blocks": sum(row["failure_kind"] == "gate" or "after_request_error" in row for row in attempts),
        "first_tool_call_only": True}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrator", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", type=Path, help="JSON object with messages/tools/candidate/api_documents")
    mode.add_argument("--raw-consistency", type=float, help="Offline sigmoid mapping; no model/client required")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--request-namespace", default="")
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--cooldown", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.raw_consistency is not None:
        result = calibrate_consistency(args.raw_consistency, args.calibrator)
    else:
        state = json.loads(args.input.read_text(encoding="utf-8"))
        result = estimate_step_confidence(
            messages=state["messages"], tools=state["tools"], candidate=state["candidate"],
            api_documents=state["api_documents"], calibrator=args.calibrator,
            base_url=args.base_url, request_namespace=args.request_namespace,
            seed=args.seed, cooldown_s=args.cooldown)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
