"""Independent cross-session API-Bank cases requiring two saved details."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy

from apibank_env import api_schema_to_tool
from memory_experiment_data import _attach_documents, _read_json, evaluate_action
from memory_versioned_boundaries_data import DATA, _norm, _original_documents
from memory_versioned_provenance_data import (
    DEFAULT_PROVENANCE_CASES, PROVENANCE_COUNTS, PROVENANCE_PRIOR_CASE_FILES,
    PROVENANCE_SCHEMA_CORPORA, _literals, _prior_set_with_boundaries,
)
from minimal_validation_data import schema_valid


DEFAULT_MULTISLOT_CASES = DATA / "memory_multislot_cases.json"
MULTISLOT_VERSION = "cross-session-multislot-v1"
MULTISLOT_COUNTS = {
    "validation": {"revoked": 2, "missing": 2, "wrong_profile": 1, "wrong_slot": 1, "current_update": 2},
    "test": {"saved": 24},
}
MULTISLOT_SCHEMA_CORPORA = PROVENANCE_SCHEMA_CORPORA
MULTISLOT_PRIOR_CASE_FILES = (*PROVENANCE_PRIOR_CASE_FILES, DEFAULT_PROVENANCE_CASES)


def _prior_rows():
    ids, bases, prompts = _prior_set_with_boundaries()
    raw = _read_json(DEFAULT_PROVENANCE_CASES)
    for split in PROVENANCE_COUNTS:
        for row in raw[split]:
            ids.add(row["task_id"])
            bases.add(row["base_task_id"])
            prompts.add(_norm(row["requirement"]))
    return ids, bases, prompts


def _select(row, properties):
    fields = row["memory_fields"]
    details = row["natural_details"]
    static = row["known_arguments"]
    events = row["memory_events"]
    overrides = row.get("current_overrides", {})
    affected = row.get("affected_field")
    condition = row["condition"]
    prompt = _norm(row["requirement"])
    if (len(fields) != 2 or len(set(fields.values())) != 2 or set(fields) != set(details)
            or set(static) & set(fields) or set(static) | set(fields) != set(properties)
            or set(row["field_aliases"]) != set(properties)
            or (condition == "saved") != (affected is None)
            or (affected is not None and affected not in fields)
            or set(overrides) != ({affected} if condition == "current_update" else set())
            or any(_norm(detail) not in prompt for detail in details.values())):
        raise ValueError(f"Invalid multislot annotation for {row['task_id']}")
    sequences = [event["sequence"] for event in events]
    if (any(type(value) is not int or value < 1 for value in sequences)
            or len(sequences) != len(set(sequences))):
        raise ValueError(f"Invalid memory chronology for {row['task_id']}")
    all_scoped = []
    selected = {}
    missing_events = {"revoked": ["save", "revoke"], "missing": [],
                      "wrong_profile": [], "wrong_slot": [], "current_update": ["save"]}
    for field, slot in fields.items():
        timeline = sorted(
            (event for event in events if event["profile_id"] == row["profile_id"] and event["slot"] == slot),
            key=lambda event: event["sequence"],
        )
        expected_kinds = missing_events[condition] if field == affected else ["save"]
        if [event["kind"] for event in timeline] != expected_kinds:
            raise ValueError(f"Wrong scoped events for {row['task_id']} field {field}")
        for event in timeline:
            all_scoped.append(event)
            if _norm(details[field]) not in _norm(event["observation"][0]["content"]):
                raise ValueError(f"Authorization does not name its field for {event['record_id']}")
            if event["kind"] == "save" and not schema_valid(event["value"], properties[field]):
                raise ValueError(f"Wrong saved value type for {event['record_id']}")
        selected[field] = timeline[-1] if timeline and timeline[-1]["kind"] == "save" else None
    foreign = [event for event in events if event not in all_scoped]
    if condition == "wrong_profile":
        valid_foreign = (len(foreign) == 1 and foreign[0]["profile_id"] != row["profile_id"]
                         and foreign[0]["slot"] == fields[affected])
    elif condition == "wrong_slot":
        valid_foreign = (len(foreign) == 1 and foreign[0]["profile_id"] == row["profile_id"]
                         and foreign[0]["slot"] not in fields.values())
    else:
        valid_foreign = not foreign
    if not valid_foreign or any(event["kind"] != "save" or not schema_valid(event["value"], properties[affected])
                                for event in foreign):
        raise ValueError(f"Wrong nonmatching events for {row['task_id']}")
    for event in events:
        observation = event["observation"]
        if ([turn["role"] for turn in observation] != ["user", "assistant"]
                or event["kind"] not in {"save", "revoke"}
                or (event["kind"] == "save") != ("value" in event)):
            raise ValueError(f"Invalid memory authorization for {event['record_id']}")
        if event["kind"] == "save":
            authorization = _norm(observation[0]["content"])
            if any(not literal.search(authorization) or literal.search(prompt) for literal in _literals(event["value"])):
                raise ValueError(f"Stored value must be visible only in the prior record {event['record_id']}")
    if [event["record_id"] for event in selected.values() if event] != row["permitted_memory_ids"]:
        raise ValueError(f"Incorrect permitted IDs for {row['task_id']}")
    return selected


def _case(row, documents, references):
    tool = row["target_tool"]
    schema = api_schema_to_tool(documents[tool])["function"]["parameters"]
    selected = _select(row, schema["properties"])
    overrides = row.get("current_overrides", {})
    sources = {field: {"source": "requirement", "value": value}
               for field, value in row["known_arguments"].items()}
    for field, event in selected.items():
        if field in overrides:
            sources[field] = {"source": "requirement", "value": overrides[field]}
        elif event:
            sources[field] = {"source": "history", "value": event["value"]}
        else:
            sources[field] = {"source": "missing"}
    missing = [field for field, item in sources.items() if item["source"] == "missing"]
    expected = ({"kind": "clarify", "fields": missing}
                if missing else {"kind": "call", "api_name": tool,
                                 "arguments": {field: item["value"] for field, item in sources.items()}})
    aliases = row["field_aliases"]
    case = {
        **{key: row[key] for key in ("task_id", "base_task_id", "requirement", "target_tool", "condition",
                                     "profile_id", "memory_fields", "memory_events", "permitted_memory_ids")},
        "api_names": [tool], "expected": expected, "parameter_sources": sources,
        "field_aliases": aliases, "clarification_aliases": aliases,
        "history": [deepcopy(turn) for record in selected.values() if record for turn in record["observation"]],
        "provenance": {"scenario": "controlled_cross_session_two_saved_facts",
                       "condition": row["condition"], "natural_details": row["natural_details"]},
    }
    _attach_documents(case, documents, references)
    case["history"] = []
    assert len(missing) <= 1, "Only one target field may be missing from a multislot control"
    action = ({"kind": "clarify", "text": f"Please provide {missing[0].replace('_', ' ')}."}
              if missing else expected)
    if not evaluate_action(case, action)["correct"]:
        raise ValueError(f"Invalid reference action for {row['task_id']}")
    if overrides:
        field, = overrides
        stale = {**expected, "arguments": {**expected["arguments"], field: selected[field]["value"]}}
        if (not all(literal.search(_norm(row["requirement"])) for literal in _literals(overrides[field]))
                or evaluate_action(case, stale)["unsupported_fields"] != [field]):
            raise ValueError(f"Invalid explicit current-value override for {row['task_id']}")
    return case


def load_multislot_cases(path=DEFAULT_MULTISLOT_CASES):
    """Return checked A/B-ready rows without exposing stored observations to the current chat."""
    raw = _read_json(path)
    if raw.get("version") != MULTISLOT_VERSION:
        raise ValueError("Unsupported multislot dataset version")
    for split, counts in MULTISLOT_COUNTS.items():
        if Counter(row["condition"] for row in raw[split]) != counts:
            raise ValueError(f"Wrong condition counts in {split}")
    rows = [row for split in MULTISLOT_COUNTS for row in raw[split]]
    ids, bases, prompts = _prior_rows()
    references = raw["schemas"]
    tools = {row["target_tool"] for row in rows}
    records = [event["record_id"] for row in rows for event in row["memory_events"]]
    if set(references) != tools or len(tools) != len(rows) or len(set(records)) != len(records):
        raise ValueError("Expected unique tools and event IDs in this cohort")
    documents = _original_documents(references)
    splits = {split: [] for split in MULTISLOT_COUNTS}
    for split, cases in splits.items():
        for row in raw[split]:
            task, base, prompt = row["task_id"], row["base_task_id"], _norm(row["requirement"])
            if task in ids or base in bases or prompt in prompts or not base.startswith("multislot-"):
                raise ValueError(f"Duplicate or earlier use of multislot scenario {task}")
            ids.add(task)
            bases.add(base)
            prompts.add(prompt)
            cases.append(_case(row, documents, references))
    return splits
