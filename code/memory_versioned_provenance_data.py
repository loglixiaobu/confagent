"""Fresh cross-session facts with independently checked authorization links."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import re

from apibank_env import api_schema_to_tool
from memory_experiment_data import _attach_documents, _read_json, evaluate_action
from memory_versioned_boundaries_data import (
    BOUNDARIES_COUNTS, BOUNDARIES_SCHEMA_CORPORA, DATA, DEFAULT_BOUNDARIES_CASES,
    PRIOR_CASE_FILES, _norm, _original_documents, _prior_sets,
)
from memory_versioned_data import _build_case, _reference_selection
from minimal_validation_data import schema_valid


DEFAULT_PROVENANCE_CASES = DATA / "memory_versioned_provenance_cases.json"
PROVENANCE_VERSION = "cross-session-versioned-provenance-v1"
PROVENANCE_COUNTS = BOUNDARIES_COUNTS
PROVENANCE_SCHEMA_CORPORA = BOUNDARIES_SCHEMA_CORPORA
PROVENANCE_PRIOR_CASE_FILES = (*PRIOR_CASE_FILES, DEFAULT_BOUNDARIES_CASES)


def _prior_set_with_boundaries():
    tasks, bases, prompts, _ = _prior_sets()
    raw = _read_json(DEFAULT_BOUNDARIES_CASES)
    for split in PROVENANCE_COUNTS:
        for row in raw[split]:
            tasks.add(row["task_id"])
            bases.add(row["base_task_id"])
            prompts.add(_norm(row["requirement"]))
    return tasks, bases, prompts


def _literals(value):
    if isinstance(value, list):
        for item in value:
            yield from _literals(item)
    else:
        yield re.compile(r"(?<!\w)" + re.escape(_norm(str(value))) + r"(?!\w)")


def _check_observation_links(row, timeline):
    prompt, detail = _norm(row["requirement"]), _norm(row["natural_detail"])
    if detail not in prompt:
        raise ValueError(f"Missing natural detail description for {row['task_id']}")
    for event in row["memory_events"]:
        authorization = _norm(event["observation"][0]["content"])
        if event in timeline and detail not in authorization:
            raise ValueError(f"Authorization does not identify its natural slot for {event['record_id']}")
        if event["kind"] == "save":
            if any(not literal.search(authorization) or literal.search(prompt)
                   for literal in _literals(event["value"])):
                raise ValueError(f"Saved values must occur only in prior evidence for {event['record_id']}")


def load_provenance_cases(path=DEFAULT_PROVENANCE_CASES):
    """Load untouched cross-session prompts, authorized event transcripts, and answer keys."""
    raw = _read_json(path)
    if raw.get("version") != PROVENANCE_VERSION:
        raise ValueError("Unsupported provenance dataset version")
    for split, counts in PROVENANCE_COUNTS.items():
        if Counter(row["condition"] for row in raw[split]) != counts:
            raise ValueError(f"Incorrect condition counts in {split}")
    rows = [row for split in PROVENANCE_COUNTS for row in raw[split]]
    tools = {row["target_tool"] for row in rows}
    records = [event["record_id"] for row in rows for event in row["memory_events"]]
    references = raw["schemas"]
    if len(tools) != len(rows) or set(references) != tools or len(set(records)) != len(records):
        raise ValueError("Expected distinct tools within the cohort and independent event IDs")
    documents = _original_documents(references)
    tasks, bases, prompts = _prior_set_with_boundaries()
    cases = {split: [] for split in PROVENANCE_COUNTS}
    for split in cases:
        for row in raw[split]:
            task, base, prompt = (row["task_id"], row["base_task_id"], _norm(row["requirement"]))
            if task in tasks or base in bases or prompt in prompts or not base.startswith("provenance-"):
                raise ValueError(f"Duplicate or prior-set scenario: {task}")
            tasks.add(task)
            bases.add(base)
            prompts.add(prompt)
            latest, timeline = _reference_selection(row)
            _check_observation_links(row, timeline)
            case = _build_case(row, latest)
            case["provenance"].update(scenario="controlled_versioned_cross_session_provenance",
                                      natural_detail=row["natural_detail"])
            field = case["memory_field"]
            if case["parameter_sources"][field]["source"] == "history":
                case["history"] = deepcopy(latest["observation"])
            _attach_documents(case, documents, references)
            case["history"] = []
            field_schema = api_schema_to_tool(documents[case["target_tool"]])["function"]["parameters"]["properties"][field]
            if any(not schema_valid(event["value"], field_schema)
                   for event in row["memory_events"] if event["kind"] == "save"):
                raise ValueError(f"Invalid saved value for {task}")
            expected = case["expected"]
            action = (expected if expected["kind"] == "call"
                      else {"kind": "clarify", "text": f"Please provide {field.replace('_', ' ')}."})
            if not evaluate_action(case, action)["correct"]:
                raise ValueError(f"Invalid reference action for {task}")
            if case["condition"] in {"updated", "current_update"}:
                stale = {**expected, "arguments": {**expected["arguments"], field: timeline[0]["value"]}}
                if evaluate_action(case, stale)["unsupported_fields"] != [field]:
                    raise ValueError(f"Expected old value to conflict with the reference for {task}")
            cases[split].append(case)
    return cases
