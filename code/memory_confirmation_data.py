"""Independent held-out pairs for checking transferable argument grounding."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

from apibank_env import api_schema_to_tool
from memory_experiment_data import (_attach_documents, _load_documents, _read_json,
                                    evaluate_action, load_cases, load_stress_cases)
from minimal_validation_data import schema_valid

DEFAULT_CONFIRMATION_CASES = Path(__file__).resolve().parents[1] / "data/memory_grounding_confirmation_cases.json"
CONFIRMATION_VERSION = "memory-grounding-confirmation-cases-v1"
CONFIRMATION_COUNTS = {
    "identifier_or_entity": 6,
    "time_or_sequence_role": 6,
    "person_or_location_scope": 6,
    "value_or_collection_role": 6,
}


def _prior_rows(existing_cases):
    if existing_cases is None:
        original = load_cases()
        return [case for split in original.values() for case in split] + load_stress_cases(existing_cases=original)
    if isinstance(existing_cases, Mapping):
        return [case for split in existing_cases.values() for case in split]
    return list(existing_cases)


def _expand(base):
    field = base["missing_field"]
    known = base["known_arguments"]
    if field in known or not set(base.get("history_fields", ())).issubset(known):
        raise ValueError(f"Conflicting argument annotations for {base['base_task_id']}")
    if base.get("history_fields") and not base.get("history"):
        raise ValueError(f"Annotated history is absent for {base['base_task_id']}")
    arguments = {**known, field: base["complete_value"]}
    for variant in ("missing", "complete"):
        needs_clarification = variant == "missing"
        target = base["target_tool"]
        sources = {
            key: {"source": "history" if key in base.get("history_fields", ()) else "requirement", "value": value}
            for key, value in arguments.items()
        }
        expectation = {"kind": "call", "api_name": target, "arguments": arguments}
        requirement = base["requirement"]
        if needs_clarification:
            sources[field] = {"source": "missing"}
            expectation = {"kind": "clarify", "fields": [field]}
        else:
            requirement += " " + base["completion"]
        yield {
            "task_id": f"{base['base_task_id']}-{variant}",
            "base_task_id": base["base_task_id"],
            "requirement": requirement,
            "history": deepcopy(base.get("history", [])),
            "target_tool": target,
            "api_names": [target],
            "parameter_sources": sources,
            "expected": expectation,
            "clarification_aliases": {
                key: deepcopy(base.get("aliases", {}).get(key, []))
                for key in arguments
            },
            "provenance": {
                "scenario": "controlled_author_created_confirmation",
                "variant": variant,
                "challenge_type": base["category"],
                "evidence_boundary": base["evidence_boundary"],
                "tempting_args": {field: base["lure"]},
            },
        }


def load_confirmation_cases(path=DEFAULT_CONFIRMATION_CASES, *, existing_cases=None):
    """Materialize matched missing and complete cases with frozen API-Bank schemas."""
    raw = _read_json(path)
    if raw.get("version") != CONFIRMATION_VERSION:
        raise ValueError("Unsupported confirmation dataset version")
    bases = raw["base_cases"]
    if Counter(base["category"] for base in bases) != CONFIRMATION_COUNTS:
        raise ValueError("Confirmation set must contain six case pairs per challenge type")
    references = raw["schemas"]
    if set(references) != {base["target_tool"] for base in bases} or len(references) != len(bases):
        raise ValueError("Expected one distinct API-Bank schema for each confirmation case pair")
    documents = _load_documents(references)
    prior = _prior_rows(existing_cases)
    old_tools = {case["target_tool"] for case in prior}
    if old_tools & references.keys():
        raise ValueError("Confirmation tool names overlap the preceding experiments")
    base_ids = {case["base_task_id"] for case in prior}
    task_ids = {case["task_id"] for case in prior}
    prompts = {" ".join(case["requirement"].casefold().split()) for case in prior}
    cases = []
    for base in bases:
        base_id = base["base_task_id"]
        if not base_id.startswith("confirmation-") or base_id in base_ids:
            raise ValueError(f"Duplicate confirmation case pair: {base_id}")
        base_ids.add(base_id)
        missing, complete = _expand(base)
        for case in (missing, complete):
            task_id = case["task_id"]
            prompt = " ".join(case["requirement"].casefold().split())
            if task_id in task_ids or prompt in prompts:
                raise ValueError(f"Duplicate confirmation case: {task_id}")
            task_ids.add(task_id)
            prompts.add(prompt)
            _attach_documents(case, documents, references)
        field = base["missing_field"]
        lure_arguments = {**complete["expected"]["arguments"], field: base["lure"]}
        lure = {"kind": "call", "api_name": base["target_tool"], "arguments": lure_arguments}
        gold_call = {"kind": "call", "api_name": base["target_tool"],
                     "arguments": complete["expected"]["arguments"]}
        gold_question = {"kind": "clarify", "text": "Please provide " + field.replace("_", " ") + "."}
        schema = api_schema_to_tool(documents[base["target_tool"]])["function"]["parameters"]
        if (not schema_valid(lure_arguments, schema)
                or set(evaluate_action(missing, lure)["unsupported_fields"]) != {field}
                or set(evaluate_action(complete, lure)["unsupported_fields"]) != {field}
                or not evaluate_action(missing, gold_question)["correct"]
                or not evaluate_action(complete, gold_call)["correct"]):
            raise ValueError(f"Invalid confirmation reference answer or lure: {base_id}")
        cases.extend((missing, complete))
    return cases
