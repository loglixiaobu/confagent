"""Independent versioned-memory cohort spanning API-Bank L1 and L3 tools."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from apibank_env import api_schema_to_tool
from memory_experiment_data import (_attach_documents, _load_documents, _read_json,
                                    SCHEMA_CORPORA, evaluate_action)
from memory_versioned_data import _build_case, _reference_selection
from minimal_validation_data import schema_valid


DATA = Path(__file__).resolve().parents[1] / "data"
DEFAULT_BOUNDARIES_CASES = DATA / "memory_versioned_boundaries_cases.json"
BOUNDARIES_VERSION = "cross-session-versioned-boundaries-v1"
BOUNDARIES_COUNTS = {
    "validation": {"revoked": 2, "wrong_profile": 2, "wrong_slot": 2, "current_update": 2},
    "replication": {"saved": 16},
    "updated": {"updated": 20},
}
BOUNDARIES_SCHEMA_CORPORA = {
    **SCHEMA_CORPORA, "level1_response": DATA / "api-bank" / "test-data" / "level-1-response.json",
}
PRIOR_CASE_FILES = tuple(DATA / name for name in (
    "memory_grounding_cases.json", "memory_grounding_stress_cases.json",
    "memory_grounding_confirmation_cases.json", "memory_factual_cases.json", "memory_versioned_cases.json",
))


def _original_documents(references):
    level1 = {name: ref for name, ref in references.items() if ref["corpus"] == "level1_response"}
    documents = _load_documents({name: ref for name, ref in references.items() if name not in level1})
    if level1:
        rows = _read_json(BOUNDARIES_SCHEMA_CORPORA["level1_response"])
        for name, ref in level1.items():
            text = rows[ref["record_index"]]["instruction"]
            parsed = [json.loads(line) for line in text.splitlines() if line.lstrip().startswith('{"name"')]
            matches = [doc for doc in parsed if doc["name"] == name]
            if len(matches) != 1:
                raise ValueError(f"Expected one original level-1 schema for {name}")
            documents[name] = matches[0]
    return documents


def _norm(prompt):
    return " ".join(prompt.casefold().split())


def _prior_sets():
    tasks, bases, prompts, tools_v6 = set(), set(), set(), set()
    for path in PRIOR_CASE_FILES:
        raw = _read_json(path)
        if path.name == "memory_versioned_cases.json":
            tools_v6.update(raw["schemas"])
        for split in ("development", "validation", "test", "base_cases", "replication", "updated"):
            for row in raw.get(split, []):
                if "task_id" in row:
                    tasks.add(row["task_id"])
                bases.add(row["base_task_id"])
                prompts.add(_norm(row["requirement"]))
                if "completion" in row:
                    prompts.add(_norm(row["requirement"] + " " + row["completion"]))
    return tasks, bases, prompts, tools_v6


def load_boundaries_cases(path=DEFAULT_BOUNDARIES_CASES):
    """Expand fixed V7 rows with independent labels and untouched prior observations."""
    raw = _read_json(path)
    if raw.get("version") != BOUNDARIES_VERSION:
        raise ValueError("Unsupported memory-boundaries dataset version")
    for split, counts in BOUNDARIES_COUNTS.items():
        if Counter(row["condition"] for row in raw[split]) != counts:
            raise ValueError(f"Wrong condition counts in {split}")
    rows = [row for split in BOUNDARIES_COUNTS for row in raw[split]]
    tasks, bases, prompts, previous_tools = _prior_sets()
    references = raw["schemas"]
    tools = {row["target_tool"] for row in rows}
    records = [event["record_id"] for row in rows for event in row["memory_events"]]
    if (set(references) != tools or len(tools) != len(rows)
            or previous_tools & tools or len(set(records)) != len(records)):
        raise ValueError("Boundaries cases require distinct records and new tool names relative to V6")
    documents = _original_documents(references)
    cases = {split: [] for split in BOUNDARIES_COUNTS}
    for split in cases:
        for row in raw[split]:
            task, base, prompt = (row["task_id"], row["base_task_id"], _norm(row["requirement"]))
            if task in tasks or base in bases or prompt in prompts or not base.startswith("boundary-"):
                raise ValueError(f"Duplicate or prior-set scenario: {task}")
            tasks.add(task)
            bases.add(base)
            prompts.add(prompt)
            latest, timeline = _reference_selection(row)
            case = _build_case(row, latest)
            case["provenance"]["scenario"] = "controlled_versioned_cross_session_boundaries"
            field = case["memory_field"]
            if case["parameter_sources"][field]["source"] == "history":
                case["history"] = deepcopy(latest["observation"])
            _attach_documents(case, documents, references)
            case["history"] = []
            field_schema = api_schema_to_tool(case["api_documents"][0])["function"]["parameters"]["properties"][field]
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
