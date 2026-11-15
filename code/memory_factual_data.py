"""Fixed cross-session stored-fact tasks and stale-record controls."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

from apibank_env import api_schema_to_tool
from memory_experiment_data import _attach_documents, _load_documents, _read_json, evaluate_action
from minimal_validation_data import schema_valid

DEFAULT_FACTUAL_CASES = Path(__file__).resolve().parents[1] / "data/memory_factual_cases.json"
FACTUAL_CASES_VERSION = "cross-session-factual-memory-v1"
FACTUAL_CONTROL_COUNTS = {"current_update": 3, "absent": 3, "wrong_profile": 3, "wrong_slot": 3}
FACTUAL_TEST_COUNT = 24


def matching_factual_record(case):
    """Retrieve a user-confirmed record in the current profile and fact slot."""
    record = case["memory_record"]
    if (record is not None and record["profile_id"] == case["profile_id"]
            and record["slot"] == case["requested_slot"] and record["confirmed"] is True):
        return record
    return None


def _build_case(raw):
    known = raw["known_arguments"]
    field = raw["memory_field"]
    condition = raw["condition"]
    if field in known or (condition == "current_update") != ("current_value" in raw):
        raise ValueError(f"Invalid current argument declarations for {raw['task_id']}")
    sources = {name: {"source": "requirement", "value": value} for name, value in known.items()}
    target = raw["target_tool"]
    if condition == "current_update":
        value = raw["current_value"]
        sources[field] = {"source": "requirement", "value": value}
        expected = {"kind": "call", "api_name": target, "arguments": {**known, field: value}}
    elif condition == "saved":
        value = raw["memory_record"]["value"]
        sources[field] = {"source": "history", "value": value}
        expected = {"kind": "call", "api_name": target, "arguments": {**known, field: value}}
    else:
        sources[field] = {"source": "missing"}
        expected = {"kind": "clarify", "fields": [field]}
    return {
        **{name: raw[name] for name in ("task_id", "base_task_id", "requirement", "target_tool",
                                       "profile_id", "requested_slot", "memory_field", "condition",
                                       "memory_record")},
        "history": [], "api_names": [target], "parameter_sources": sources, "expected": expected,
        "clarification_aliases": {name: raw["aliases"] if name == field else [] for name in sources},
        "provenance": {
            "scenario": "controlled_cross_session_saved_facts",
            "condition": condition,
            "memory_source": "separate synthetic observation" if raw["memory_record"] else None,
        },
    }


def _validate_record(case):
    record = case["memory_record"]
    condition = case["condition"]
    if (condition == "absent") != (record is None):
        raise ValueError(f"Wrong memory availability for {case['task_id']}")
    if record is None:
        return
    observation = record["observation"]
    if (record["confirmed"] is not True or [msg["role"] for msg in observation] != ["user", "assistant"]
            or not all(isinstance(msg["content"], str) and msg["content"].strip() for msg in observation)):
        raise ValueError(f"Expected a user authorization and confirmation for {record['record_id']}")
    if (condition == "wrong_profile") != (case["profile_id"] != record["profile_id"]):
        raise ValueError(f"Wrong profile annotation for {record['record_id']}")
    if (condition == "wrong_slot") != (case["requested_slot"] != record["slot"]):
        raise ValueError(f"Wrong slot annotation for {record['record_id']}")
    if (matching_factual_record(case) is not None) != (condition in {"saved", "current_update"}):
        raise ValueError(f"Wrong retrieval annotation for {record['record_id']}")
    field = case["memory_field"]
    schema = api_schema_to_tool(case["api_documents"][0])["function"]["parameters"]["properties"][field]
    if not schema_valid(record["value"], schema):
        raise ValueError(f"Saved value violates the API schema for {record['record_id']}")
    if condition == "current_update":
        stale_action = {"kind": "call", "api_name": case["target_tool"], "arguments": {
            **case["expected"]["arguments"], field: record["value"],
        }}
        if evaluate_action(case, stale_action)["unsupported_fields"] != [field]:
            raise ValueError(f"The stored value must differ from the explicit current value for {record['record_id']}")


def load_factual_cases(path=DEFAULT_FACTUAL_CASES):
    """Return separate safety controls and independent factual recall cases."""
    raw = _read_json(path)
    if raw.get("version") != FACTUAL_CASES_VERSION:
        raise ValueError("Unsupported factual memory dataset version")
    if Counter(row["condition"] for row in raw["test"]) != {"saved": FACTUAL_TEST_COUNT}:
        raise ValueError("Expected a fixed set of 24 saved-fact test cases")
    if Counter(row["condition"] for row in raw["validation"]) != FACTUAL_CONTROL_COUNTS:
        raise ValueError("Expected twelve controls across the four retrieval conditions")
    rows = raw["validation"] + raw["test"]
    tools = [row["target_tool"] for row in rows]
    references = raw["schemas"]
    if set(tools) != set(references) or len(set(tools)) != len(rows):
        raise ValueError("Every factual scenario should use a separate API-Bank tool")
    if (len({row["task_id"] for row in rows}) != len(rows)
            or len({row["base_task_id"] for row in rows}) != len(rows)
            or not all(row["base_task_id"].startswith("factual-") for row in rows)):
        raise ValueError("Independent factual cases require unique factual base and task IDs")
    documents = _load_documents(references)
    splits = {"validation": [], "test": []}
    for split, cases in splits.items():
        for row in raw[split]:
            case = _build_case(row)
            # Validate history-backed labels using the actual stored observation.
            # The observation stays separate from the current conversation after validation.
            if case["condition"] == "saved":
                case["history"] = deepcopy(case["memory_record"]["observation"])
            _attach_documents(case, documents, references)
            case["history"] = []
            _validate_record(case)
            expected = case["expected"]
            action = ({**expected} if expected["kind"] == "call"
                      else {"kind": "clarify", "text": f"Please provide {case['memory_field'].replace('_', ' ')}."})
            if not evaluate_action(case, action)["correct"]:
                raise ValueError(f"Invalid reference action for {case['task_id']}")
            cases.append(case)
    return splits
