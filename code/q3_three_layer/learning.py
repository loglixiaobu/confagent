"""Construction-only memory abstraction, reward fitting, and paired gates.

These functions receive public contracts and already collected observations.
Memory construction/reward fitting never read a split or an evaluator. Gates
only summarize the outcome pairs explicitly supplied by their caller.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


ACTIONS = ("keep", "memory", "verify")
KNOWN_UNITS = ("km", "m", "fraction", "percent", "MiB", "KiB", "other")
FEATURE_NAMES = (
    "intercept",
    *(f"source_unit:{unit}" for unit in KNOWN_UNITS),
    *(f"target_unit:{unit}" for unit in KNOWN_UNITS),
    "source_target_unit_mismatch",
    "draft_parseable",
    "draft_matches_input_value",
    "draft_unit_matches_target",
    "memory_available",
    "verification_budget",
    "adapter_active",
)
POLICY_SCHEMA_VERSION = 1

_UNIT_ALIASES = {
    "km": "km", "kilometer": "km", "kilometers": "km",
    "kilometre": "km", "kilometres": "km",
    "m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "fraction": "fraction", "ratio": "fraction",
    "percent": "percent", "percentage": "percent", "%": "percent",
    "mib": "MiB", "mebibyte": "MiB", "mebibytes": "MiB",
    "kib": "KiB", "kibibyte": "KiB", "kibibytes": "KiB",
}
_ORIGINS = frozenset({
    "model_first_pass_success", "model_self_repair", "tool_assisted_repair",
})


def _unit(value: Any) -> str:
    if not isinstance(value, str):
        return "other"
    return _UNIT_ALIASES.get(value.strip().lower(), "other")


def _number(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)) or value is None:
        return None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _same_number(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)


def _parsed_draft(draft: Any) -> tuple[float, str] | None:
    if isinstance(draft, str):
        text = draft.strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1])
        try:
            draft = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(draft, Mapping):
        return None
    if "value" not in draft and isinstance(draft.get("parsed"), Mapping):
        draft = draft["parsed"]
    value = _number(draft.get("value"))
    unit_text = draft.get("unit")
    if value is None or not isinstance(unit_text, str) or not unit_text.strip():
        return None
    return value, _unit(unit_text)


def features(
    contract: Mapping[str, Any],
    draft: Any,
    memory_available: bool,
    budget: float = 1,
    adapter_active: bool = False,
) -> list[float]:
    """Encode only named public observations; ignore IDs, labels and extra keys.

    Matching the input value is an observable feature, not a correctness label:
    the same value is right for an identity task and may be wrong for conversion.
    """
    source = _unit(contract.get("source_unit"))
    target = _unit(contract.get("target_unit"))
    parsed = _parsed_draft(draft)
    input_value = _number(contract.get("source_value"))
    remaining = _number(budget)
    if remaining is None or remaining < 0:
        raise ValueError("verification budget must be finite and nonnegative")
    return [
        1.0,
        *(float(source == unit) for unit in KNOWN_UNITS),
        *(float(target == unit) for unit in KNOWN_UNITS),
        float(source != target),
        float(parsed is not None),
        float(parsed is not None and input_value is not None
              and _same_number(parsed[0], input_value)),
        float(parsed is not None and target != "other" and parsed[1] == target),
        float(memory_available is True),
        remaining,
        float(adapter_active is True),
    ]


def _trace_record(experience: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate one declared computational event without inventing a trace.

    The caller supplies the independent semantic certificate via certified=True.
    The arithmetic checks here only bind that trace to its public source inputs;
    they cannot replace the independent certificate.
    """
    if experience.get("certified") is not True:
        return None
    origin = experience.get("origin")
    if origin not in _ORIGINS:
        return None
    contract, trace = experience.get("contract"), experience.get("trace")
    if not isinstance(contract, Mapping) or not isinstance(trace, Mapping):
        return None
    if trace.get("executed") is not True or trace.get("success") is not True:
        return None
    tool = trace.get("tool")
    task_id, entity = experience.get("task_id"), contract.get("entity")
    if not all(isinstance(value, str) and value.strip() for value in (tool, task_id, entity)):
        return None
    source, target = _unit(contract.get("source_unit")), _unit(contract.get("target_unit"))
    if "other" in (source, target):
        return None
    if (_unit(trace.get("source_unit")), _unit(trace.get("target_unit"))) != (source, target):
        return None
    value = _number(contract.get("source_value"))
    traced_value = _number(trace.get("source_value"))
    multiplier, result = _number(trace.get("multiplier")), _number(trace.get("result"))
    if any(number is None for number in (value, traced_value, multiplier, result)):
        return None
    if multiplier <= 0 or not _same_number(value, traced_value):
        return None
    if not math.isfinite(traced_value * multiplier) or not _same_number(result, traced_value * multiplier):
        return None
    if source == target and not _same_number(multiplier, 1.0):
        return None
    return {
        "source_unit": source, "target_unit": target, "multiplier": multiplier,
        "task_id": task_id, "entity": entity, "origin": origin, "tool": tool,
    }


def build_memory(experiences: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Abstract certified actual traces, requiring two entities per unit pair.

    Source IDs/entities remain audit metadata. Instructions and executable fields
    contain no source value, instance answer, raw prompt, draft or oracle label.
    Conflicting certified factors for one pair quarantine that entire pair.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for experience in experiences:
        if not isinstance(experience, Mapping):
            continue
        record = _trace_record(experience)
        if record is not None:
            groups[(record["source_unit"], record["target_unit"])].append(record)
    bank = []
    for (source, target), records in sorted(groups.items()):
        # A duplicated ID cannot manufacture an independent second entity.
        source_entities: dict[str, str] = {}
        inconsistent_ids = False
        for record in records:
            prior = source_entities.setdefault(record["task_id"], record["entity"])
            inconsistent_ids |= prior != record["entity"]
        multiplier = records[0]["multiplier"]
        if (inconsistent_ids or len(set(source_entities.values())) < 2
                or any(not _same_number(record["multiplier"], multiplier) for record in records)):
            continue
        instruction = (
            f"When the current visible source unit is {source} and the required target unit is "
            f"{target}, multiply the CURRENT visible source value by {multiplier:.12g}. "
            f"Return that computed value in {target}. Use only the current task's quantity; "
            "do not reuse example quantities. Apply this rule only to this exact unit pair."
        )
        evidence = sorted({(r["task_id"], r["entity"], r["origin"], r["tool"]) for r in records})
        bank.append({
            "source_unit": source, "target_unit": target, "multiplier": multiplier,
            "instruction": instruction,
            "evidence_ids": sorted(source_entities),
            "entities": sorted(set(source_entities.values())),
            "origins": sorted({record["origin"] for record in records}),
            "evidence": [dict(zip(("task_id", "entity", "origin", "tool"), values)) for values in evidence],
        })
    return bank


def retrieve(bank: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Retrieve only an unambiguous exact public unit-pair match."""
    source, target = _unit(contract.get("source_unit")), _unit(contract.get("target_unit"))
    if "other" in (source, target):
        return None
    matches = [entry for entry in bank
               if (_unit(entry.get("source_unit")), _unit(entry.get("target_unit"))) == (source, target)]
    return matches[0] if len(matches) == 1 else None


def memory_instruction(entry: Mapping[str, Any]) -> str:
    instruction = entry.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("memory entry has no instruction")
    return instruction


def _feature_array(values: Sequence[float]) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("features must be a finite numeric vector") from exc
    if vector.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(vector)):
        raise ValueError(f"features must contain {len(FEATURE_NAMES)} finite numbers")
    return vector


class RidgePolicy:
    """One genuinely fitted ridge reward regressor per supported action."""

    def __init__(self, coefficients: Mapping[str, Any], counts: Mapping[str, int], regularization: float = 1.0):
        if _number(regularization) != 1.0:
            raise ValueError("the frozen protocol requires ridge regularization 1.0")
        self.regularization = 1.0
        self.coefficients: dict[str, np.ndarray | None] = {}
        self.counts: dict[str, int] = {}
        for action in ACTIONS:
            count = counts.get(action, 0)
            if type(count) is not int or count < 0:
                raise ValueError("action counts must be nonnegative integers")
            coefficient = coefficients.get(action)
            if (coefficient is None) != (count == 0):
                raise ValueError("coefficient availability must match observed action coverage")
            self.coefficients[action] = None if coefficient is None else _feature_array(coefficient).copy()
            self.counts[action] = count

    @classmethod
    def fit(cls, rows: Iterable[Mapping[str, Any]]) -> "RidgePolicy":
        grouped: dict[str, list[tuple[np.ndarray, float]]] = {action: [] for action in ACTIONS}
        for row in rows:
            action = row.get("action")
            if action not in ACTIONS:
                raise ValueError(f"unsupported policy action: {action!r}")
            reward = _number(row.get("reward"))
            if reward is None:
                raise ValueError("observed reward must be a finite number")
            grouped[action].append((_feature_array(row["features"]), reward))
        if not any(grouped.values()):
            raise ValueError("policy fitting needs observed action rewards")
        coefficients, counts = {}, {}
        for action in ACTIONS:
            observations = grouped[action]
            counts[action] = len(observations)
            if not observations:
                coefficients[action] = None
                continue
            design = np.stack([vector for vector, _ in observations])
            reward = np.array([value for _, value in observations], dtype=np.float64)
            # The intercept is included in the fixed public feature schema and
            # regularized consistently with every other coefficient.
            coefficients[action] = np.linalg.solve(
                design.T @ design + np.eye(len(FEATURE_NAMES)), design.T @ reward,
            )
        return cls(coefficients, counts)

    def predict_rewards(self, feature_values: Sequence[float]) -> dict[str, float]:
        vector = _feature_array(feature_values)
        return {action: float(vector @ coefficient) for action, coefficient in self.coefficients.items()
                if coefficient is not None}

    def choose(self, feature_values: Sequence[float], available_actions: Iterable[str]) -> str:
        available = set(available_actions)
        if not available.issubset(ACTIONS):
            raise ValueError("availability mask contains an unsupported action")
        scores = self.predict_rewards(feature_values)
        eligible = [action for action in ACTIONS if action in available and action in scores]
        if not eligible:
            raise ValueError("no available action has observed training coverage")
        # ACTIONS fixes deterministic ties in favor of keep, then memory, verify.
        return max(eligible, key=lambda action: scores[action])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": POLICY_SCHEMA_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "regularization": self.regularization,
            "coefficients": {action: None if coefficient is None else coefficient.tolist()
                             for action, coefficient in self.coefficients.items()},
            "counts": dict(self.counts),
            "coverage": {action: self.counts[action] > 0 for action in ACTIONS},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RidgePolicy":
        if payload.get("schema_version") != POLICY_SCHEMA_VERSION or payload.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("policy feature schema/version does not match")
        policy = cls(payload["coefficients"], payload["counts"], payload["regularization"])
        if payload.get("coverage") != {action: policy.counts[action] > 0 for action in ACTIONS}:
            raise ValueError("serialized coverage disagrees with observed counts")
        return policy


def paired_gate(
    base_bools: Iterable[bool | None],
    candidate_bools: Iterable[bool | None],
    min_pairs: int = 4,
    require_fix: bool = True,
) -> dict[str, Any]:
    """Count paired fixes/harms; unknowns and incomplete pairs fail closed."""
    if type(min_pairs) is not int or min_pairs < 1 or type(require_fix) is not bool:
        raise ValueError("gate configuration requires a positive pair count and boolean require_fix")
    base, candidate = list(base_bools), list(candidate_bools)
    known = [(before, after) for before, after in zip(base, candidate)
             if type(before) is bool and type(after) is bool]
    fixes = sum(not before and after for before, after in known)
    harms = sum(before and not after for before, after in known)
    unknown = min(len(base), len(candidate)) - len(known)
    if len(base) != len(candidate):
        reason = "length_mismatch"
    elif unknown:
        reason = "unknown_outcomes"
    elif len(base) < min_pairs:
        reason = "insufficient_pairs"
    elif harms:
        reason = "observed_harm"
    elif require_fix and not fixes:
        reason = "no_fix"
    else:
        reason = "passed"
    return {
        "F": fixes, "H": harms, "n": min(len(base), len(candidate)),
        "known_pairs": len(known), "unknown_pairs": unknown, "required_pairs": min_pairs,
        "pass": reason == "passed", "reason": reason,
    }
