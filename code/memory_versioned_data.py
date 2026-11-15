"""Independent retained and revised cross-session facts with access controls."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

from apibank_env import api_schema_to_tool
from memory_experiment_data import _attach_documents, _load_documents, _read_json, evaluate_action
from minimal_validation_data import schema_valid

DEFAULT_VERSIONED_CASES = Path(__file__).resolve().parents[1] / "data/memory_versioned_cases.json"
VERSIONED_DATA_VERSION = "cross-session-versioned-memory-v1"
VERSIONED_COUNTS = {
    "validation": {"revoked": 2, "wrong_profile": 2, "wrong_slot": 2, "current_update": 2},
    "replication": {"saved": 12},
    "updated": {"updated": 20},
}


def _reference_selection(row):
    """Independently check the manifest's permitted record, before any inference."""
    events = row["memory_events"]
    condition = row["condition"]
    sequences = [event["sequence"] for event in events]
    if not sequences or any(type(seq) is not int or seq < 1 for seq in sequences) or len(set(sequences)) != len(events):
        raise ValueError(f"Invalid event chronology for {row['task_id']}")
    for event in events:
        observation = event["observation"]
        if ([turn["role"] for turn in observation] != ["user", "assistant"]
                or not all(isinstance(turn["content"], str) and turn["content"].strip() for turn in observation)
                or event["kind"] not in {"save", "revoke"}
                or (event["kind"] == "save") != ("value" in event)):
            raise ValueError(f"Invalid memory authorization event for {event['record_id']}")
    timeline = sorted(
        (event for event in events if event["profile_id"] == row["profile_id"]
         and event["slot"] == row["requested_slot"]),
        key=lambda event: event["sequence"],
    )
    expected_kinds = {
        "saved": ["save"], "updated": ["save", "save"], "current_update": ["save"],
        "revoked": ["save", "revoke"], "wrong_profile": [], "wrong_slot": [],
    }
    if [event["kind"] for event in timeline] != expected_kinds[condition]:
        raise ValueError(f"Wrong in-scope history for {row['task_id']}")
    if condition == "wrong_profile" and not all(event["slot"] == row["requested_slot"] for event in events):
        raise ValueError(f"Wrong-profile control must hold the slot constant for {row['task_id']}")
    if condition == "wrong_slot" and not all(event["profile_id"] == row["profile_id"] for event in events):
        raise ValueError(f"Wrong-slot control must hold the profile constant for {row['task_id']}")
    latest = timeline[-1] if timeline and timeline[-1]["kind"] == "save" else None
    if row["permitted_memory_ids"] != ([latest["record_id"]] if latest else []):
        raise ValueError(f"Wrong permitted memory record for {row['task_id']}")
    return latest, timeline


def _build_case(row, latest):
    known = row["known_arguments"]
    field = row["memory_field"]
    condition = row["condition"]
    target = row["target_tool"]
    if field in known or (condition == "current_update") != ("current_value" in row):
        raise ValueError(f"Invalid current-argument declarations for {row['task_id']}")
    sources = {key: {"source": "requirement", "value": value} for key, value in known.items()}
    if latest is not None:
        using_current = condition == "current_update"
        value = row["current_value"] if using_current else latest["value"]
        sources[field] = {"source": "requirement" if using_current else "history", "value": value}
        expected = {"kind": "call", "api_name": target, "arguments": {**known, field: value}}
    else:
        sources[field] = {"source": "missing"}
        expected = {"kind": "clarify", "fields": [field]}
    return {
        **{key: row[key] for key in ("task_id", "base_task_id", "requirement", "target_tool", "condition",
                                     "profile_id", "requested_slot", "memory_field", "memory_events",
                                     "permitted_memory_ids")},
        "api_names": [target], "history": [], "expected": expected, "parameter_sources": sources,
        "clarification_aliases": {key: row["aliases"] if key == field else [] for key in sources},
        "provenance": {
            "scenario": "controlled_versioned_cross_session_facts",
            "condition": condition,
            "stored_sequence_order": [event["sequence"] for event in row["memory_events"]],
            "wrong_scope_decoys": sum(
                event["profile_id"] != row["profile_id"] or event["slot"] != row["requested_slot"]
                for event in row["memory_events"]
            ),
        },
    }


def load_versioned_cases(path=DEFAULT_VERSIONED_CASES):
    """Expand fixed tasks while keeping every prior observation out of current history."""
    raw = _read_json(path)
    if raw.get("version") != VERSIONED_DATA_VERSION:
        raise ValueError("Unsupported versioned memory dataset version")
    for split, conditions in VERSIONED_COUNTS.items():
        if Counter(row["condition"] for row in raw[split]) != conditions:
            raise ValueError(f"Unexpected case counts in versioned {split}")
    rows = [row for split in VERSIONED_COUNTS for row in raw[split]]
    task_ids = {row["task_id"] for row in rows}
    base_ids = {row["base_task_id"] for row in rows}
    tools = {row["target_tool"] for row in rows}
    records = [event["record_id"] for row in rows for event in row["memory_events"]]
    if (len(task_ids) != len(rows) or len(base_ids) != len(rows) or len(tools) != len(rows)
            or len(set(records)) != len(records) or set(raw["schemas"]) != tools
            or not all(base.startswith("versioned-") for base in base_ids)):
        raise ValueError("Versioned cases require unique tasks, base IDs, tool names and record IDs")
    documents = _load_documents(raw["schemas"])
    cases = {split: [] for split in VERSIONED_COUNTS}
    for split in cases:
        for row in raw[split]:
            latest, timeline = _reference_selection(row)
            case = _build_case(row, latest)
            field = row["memory_field"]
            if case["parameter_sources"][field]["source"] == "history":
                case["history"] = deepcopy(latest["observation"])
            _attach_documents(case, documents, raw["schemas"])
            case["history"] = []
            schema = api_schema_to_tool(case["api_documents"][0])["function"]["parameters"]["properties"][field]
            if any(not schema_valid(event["value"], schema) for event in row["memory_events"] if event["kind"] == "save"):
                raise ValueError(f"Invalid typed memory value for {case['task_id']}")
            expected = case["expected"]
            oracle = (expected if expected["kind"] == "call"
                      else {"kind": "clarify", "text": f"Please provide {field.replace('_', ' ')}."})
            if not evaluate_action(case, oracle)["correct"]:
                raise ValueError(f"Invalid versioned reference action for {case['task_id']}")
            if case["condition"] in {"updated", "current_update"}:
                old_action = {**expected, "arguments": {**expected["arguments"], field: timeline[0]["value"]}}
                if evaluate_action(case, old_action)["unsupported_fields"] != [field]:
                    raise ValueError(f"Old value must conflict with the expected action for {case['task_id']}")
            cases[split].append(case)
    return cases
