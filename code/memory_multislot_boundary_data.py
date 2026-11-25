"""Checked adapter for the V9 dual-field development and test extension.

The JSON deliberately separates raw ``case_input`` from ``answer_key`` and
``offline_candidates``. Candidates are development-only. Public helpers preserve that boundary: callers that
prepare verifier prompts should use :func:`load_boundary_model_inputs`, while
CPU-only evaluators may use :func:`load_boundary_candidate_replay`.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from apibank_env import api_schema_to_tool
from memory_experiment_data import _read_json, evaluate_action
from memory_multislot_data import MULTISLOT_SCHEMA_CORPORA, _case, _prior_rows
from memory_versioned_boundaries_data import DATA, _norm, _original_documents


DEFAULT_MULTISLOT_BOUNDARY_CASES = DATA / "memory_multislot_boundary_cases.json"
BOUNDARY_VERSION = "cross-session-multislot-boundary-v1"
BOUNDARY_COUNTS = {
    "both_grounded": 2,
    "one_missing": 2,
    "revoked_or_identity_mismatch": 2,
    "current_override": 2,
}
BOUNDARY_TEST_COUNTS = {
    "both_grounded": 3,
    "one_missing": 3,
    "revoked_or_identity_mismatch": 3,
    "current_override": 3,
}
BOUNDARY_SPLIT_COUNTS = {
    "development": BOUNDARY_COUNTS,
    "test": BOUNDARY_TEST_COUNTS,
}
ALLOWED_CONDITIONS = {"saved", "missing", "revoked", "wrong_profile", "current_update"}
EXPECTED_CONDITIONS = {
    "saved": 2, "missing": 2, "revoked": 1, "wrong_profile": 1, "current_update": 2,
}
EXPECTED_TEST_CONDITIONS = {
    "saved": 3, "missing": 3, "revoked": 2, "wrong_profile": 1, "current_update": 3,
}
EXPECTED_SPLIT_CONDITIONS = {
    "development": EXPECTED_CONDITIONS,
    "test": EXPECTED_TEST_CONDITIONS,
}
CATEGORY_CONDITIONS = {
    "both_grounded": {"saved"},
    "one_missing": {"missing"},
    "revoked_or_identity_mismatch": {"revoked", "wrong_profile"},
    "current_override": {"current_update"},
}
def _source_without_audit(source: Dict[str, Any]) -> Dict[str, Any]:
    return {key: source[key] for key in ("source", "value") if key in source}


def _visible_input(row: Dict[str, Any], resolved: Dict[str, Any]) -> Dict[str, Any]:
    """Build the only object that may be passed to a verifier or agent."""
    permitted = set(row["permitted_memory_ids"])
    records = [
        deepcopy(event) for event in row["memory_events"]
        if event["record_id"] in permitted and event["kind"] == "save"
    ]
    return {
        "task_id": row["task_id"],
        "profile_id": row["profile_id"],
        "requirement": row["requirement"],
        "target_tool": row["target_tool"],
        "api_documents": deepcopy(resolved["api_documents"]),
        "memory_fields": deepcopy(row["memory_fields"]),
        "permitted_memory_ids": list(row["permitted_memory_ids"]),
        "permitted_memory": records,
    }


def _check_against_api_bank(references: Dict[str, Any], snapshots: Dict[str, Any]) -> None:
    """Compare executable snapshots with the original corpora when available.

    The large API-Bank corpus is distributed separately and may be absent from
    a lightweight checkout.  Names, required fields and normalized JSON types
    are checked whenever all referenced corpus files exist.
    """
    needed = {reference["corpus"] for reference in references.values()}
    if not all(MULTISLOT_SCHEMA_CORPORA[name].exists() for name in needed):
        return
    originals = _original_documents(references)
    for tool in snapshots:
        snapshot = api_schema_to_tool(snapshots[tool])["function"]["parameters"]
        original = api_schema_to_tool(originals[tool])["function"]["parameters"]
        snapshot_types = {field: spec.get("type") for field, spec in snapshot["properties"].items()}
        original_types = {field: spec.get("type") for field, spec in original["properties"].items()}
        if snapshot["required"] != original["required"] or snapshot_types != original_types:
            raise ValueError(f"Tool snapshot differs from API-Bank schema for {tool}")


def _check_answer(case_id: str, answer: Dict[str, Any], resolved: Dict[str, Any],
                  row: Dict[str, Any]) -> None:
    if answer.get("correct_action") != resolved["expected"]:
        raise ValueError(f"Gold action differs from derived V9 action for {case_id}")
    annotated = answer.get("parameter_sources")
    if not isinstance(annotated, dict) or set(annotated) != set(resolved["parameter_sources"]):
        raise ValueError(f"Gold parameter fields differ from the tool schema for {case_id}")
    events = {event["record_id"]: event for event in row["memory_events"]}
    for field, derived in resolved["parameter_sources"].items():
        gold = annotated[field]
        if _source_without_audit(gold) != derived:
            raise ValueError(f"Gold source differs from derived V9 source for {case_id}:{field}")
        evidence_id = gold.get("evidence_id")
        if derived["source"] == "history":
            if evidence_id not in row["permitted_memory_ids"] or events[evidence_id]["value"] != derived["value"]:
                raise ValueError(f"History evidence is not permitted for {case_id}:{field}")
        elif derived["source"] == "requirement":
            if evidence_id != "current_request":
                raise ValueError(f"Requirement evidence is mislabeled for {case_id}:{field}")
        elif evidence_id is not None:
            raise ValueError(f"Missing field unexpectedly has evidence for {case_id}:{field}")
    expected_fields = resolved["expected"].get("fields", [])
    if answer.get("clarification_fields") != expected_fields:
        raise ValueError(f"Clarification labels differ from derived V9 action for {case_id}")


def _check_candidates(case_id: str, candidates: List[Dict[str, Any]],
                      resolved: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError(f"Expected correct and erroneous candidates for {case_id}")
    ids = [candidate.get("candidate_id") for candidate in candidates]
    if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Invalid candidate IDs for {case_id}")
    labels = Counter(candidate.get("expected_label") for candidate in candidates)
    if labels["correct"] < 1 or labels["incorrect"] < 1 or set(labels) - {"correct", "incorrect"}:
        raise ValueError(f"Candidates must include correct and incorrect labels for {case_id}")
    checked = []
    for candidate in candidates:
        action = candidate.get("action")
        if not isinstance(action, dict):
            raise ValueError(f"Candidate action is not an object for {case_id}")
        evaluation = evaluate_action(resolved, action)
        should_be_correct = candidate["expected_label"] == "correct"
        if evaluation["correct"] != should_be_correct:
            raise ValueError(f"Candidate label disagrees with V9 scorer for {case_id}:{candidate['candidate_id']}")
        if should_be_correct and candidate.get("error_type") is not None:
            raise ValueError(f"Correct candidate has an error type for {case_id}")
        if not should_be_correct and not candidate.get("error_type"):
            raise ValueError(f"Incorrect candidate lacks an error type for {case_id}")
        checked.append({**deepcopy(candidate), "evaluation": evaluation})
    return checked


def _load_checked(path=DEFAULT_MULTISLOT_BOUNDARY_CASES) -> List[Dict[str, Any]]:
    raw = _read_json(path)
    if raw.get("version") != BOUNDARY_VERSION:
        raise ValueError("Unsupported multislot boundary dataset version")
    split_bundles = {}
    for split, counts in BOUNDARY_SPLIT_COUNTS.items():
        bundles = raw.get(split)
        expected_size = sum(counts.values())
        if not isinstance(bundles, list) or len(bundles) != expected_size:
            raise ValueError(f"Expected exactly {expected_size} independent {split} boundary cases")
        if Counter(bundle.get("category") for bundle in bundles) != counts:
            raise ValueError(f"Wrong {split} boundary category allocation")
        conditions = Counter((bundle.get("case_input") or {}).get("condition") for bundle in bundles)
        if conditions != EXPECTED_SPLIT_CONDITIONS[split]:
            raise ValueError(f"Wrong underlying V9 condition allocation in {split}")
        split_bundles[split] = bundles

    references = raw.get("schemas")
    documents = raw.get("tool_documents")
    if not isinstance(references, dict) or not isinstance(documents, dict) or set(references) != set(documents):
        raise ValueError("Schema references and executable snapshots must name the same tools")
    for tool, document in documents.items():
        if document.get("name") != tool or not isinstance(document.get("input_parameters"), dict):
            raise ValueError(f"Invalid tool snapshot for {tool}")
    _check_against_api_bank(references, documents)

    prior_ids, prior_bases, prior_prompts = _prior_rows()
    seen_cases, seen_tasks, seen_bases, seen_prompts, seen_profiles = set(), set(), set(), set(), set()
    seen_records, seen_tools = set(), set()
    checked = []
    for split, bundle in (
        (split, bundle) for split in BOUNDARY_SPLIT_COUNTS for bundle in split_bundles[split]
    ):
        case_id = bundle.get("case_id")
        row = bundle.get("case_input")
        if not isinstance(case_id, str) or not isinstance(row, dict):
            raise ValueError("Every boundary bundle needs a case_id and case_input")
        task, base, profile = row.get("task_id"), row.get("base_task_id"), row.get("profile_id")
        prompt, tool = _norm(row.get("requirement", "")), row.get("target_tool")
        records = [event.get("record_id") for event in row.get("memory_events", [])]
        if (case_id in seen_cases or task in seen_tasks or base in seen_bases or prompt in seen_prompts
                or profile in seen_profiles or tool in seen_tools or any(record in seen_records for record in records)):
            raise ValueError(f"Boundary cases must be mutually independent: {case_id}")
        if task in prior_ids or base in prior_bases or prompt in prior_prompts:
            raise ValueError(f"Boundary case overlaps an earlier V9 scenario: {case_id}")
        if row.get("condition") not in ALLOWED_CONDITIONS or tool not in documents:
            raise ValueError(f"Unsupported condition or tool for {case_id}")
        if row["condition"] not in CATEGORY_CONDITIONS[bundle["category"]]:
            raise ValueError(f"Boundary category disagrees with V9 condition for {case_id}")
        seen_cases.add(case_id)
        seen_tasks.add(task)
        seen_bases.add(base)
        seen_prompts.add(prompt)
        seen_profiles.add(profile)
        seen_tools.add(tool)
        seen_records.update(records)

        resolved = _case(deepcopy(row), documents, references)
        _check_answer(case_id, bundle.get("answer_key") or {}, resolved, row)
        if split == "development":
            candidates = _check_candidates(case_id, bundle.get("offline_candidates"), resolved)
        else:
            if "offline_candidates" in bundle:
                raise ValueError(f"Independent test case must not contain candidates: {case_id}")
            candidates = []
        visible = _visible_input(row, resolved)
        if set(visible) & {"answer_key", "expected", "parameter_sources", "offline_candidates"}:
            raise ValueError(f"Gold fields leaked into model-visible input for {case_id}")
        checked.append({
            "case_id": case_id,
            "split": split,
            "category": bundle["category"],
            "model_input": visible,
            "evaluator_case": resolved,
            "answer_key": deepcopy(bundle["answer_key"]),
            "offline_candidates": candidates,
        })
    return checked


def _filter_split(bundles: List[Dict[str, Any]], split: str | None) -> List[Dict[str, Any]]:
    if split is not None and split not in BOUNDARY_SPLIT_COUNTS:
        raise ValueError(f"Unknown boundary split: {split}")
    return [bundle for bundle in bundles if split is None or bundle["split"] == split]


def load_boundary_model_inputs(path=DEFAULT_MULTISLOT_BOUNDARY_CASES,
                               split: str | None = None) -> List[Dict[str, Any]]:
    """Return gold-free inputs safe to serialize into verifier requests."""
    return [deepcopy(bundle["model_input"]) for bundle in _filter_split(_load_checked(path), split)]


def load_multislot_boundary_cases(path=DEFAULT_MULTISLOT_BOUNDARY_CASES,
                                  split: str | None = None) -> List[Dict[str, Any]]:
    """Return V9-compatible resolved cases for CPU evaluation only.

    The returned rows have the same ``expected``, ``parameter_sources``,
    ``api_documents``, ``memory_fields`` and ``permitted_memory_ids`` fields as
    :func:`memory_multislot_data.load_multislot_cases`.  Never pass these rows
    directly to a model because they intentionally contain evaluator labels.
    """
    rows = []
    for bundle in _filter_split(_load_checked(path), split):
        row = deepcopy(bundle["evaluator_case"])
        row["dataset_split"] = bundle["split"]
        row["boundary_category"] = bundle["category"]
        rows.append(row)
    return rows


def load_boundary_candidate_replay(path=DEFAULT_MULTISLOT_BOUNDARY_CASES) -> List[Dict[str, Any]]:
    """Return evaluator-only candidate rows with deterministic V9 scores."""
    rows = []
    for bundle in _filter_split(_load_checked(path), "development"):
        for candidate in bundle["offline_candidates"]:
            rows.append({
                "case_id": bundle["case_id"],
                "task_id": bundle["evaluator_case"]["task_id"],
                "category": bundle["category"],
                "split": bundle["split"],
                **deepcopy(candidate),
            })
    return rows

