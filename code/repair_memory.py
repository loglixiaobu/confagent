"""Q3: source certificates, transfer validation and bounded guidance retrieval.

References/labels stay in the evaluator. Actor-visible memory contains guidance,
API scope and cues only; stored actions are never executed as a policy.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def json_object(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text).removesuffix("```").strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except (ValueError, TypeError):
        return None


@dataclass
class RepairMemory:
    memory_id: str
    instruction: str
    api_names: list[str]
    cues: list[str]
    source_task_ids: list[str]
    source_round: int
    source_certificate: dict[str, Any]
    validation: dict[str, Any] = field(default_factory=dict)
    status: str = "proposed"

    def visible(self):
        return {"instruction": self.instruction, "api_names": list(self.api_names),
                "cues": list(self.cues)}

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, row):
        return cls(**row)


def source_certificate(task_id, before, after, evaluator):
    """The supplied independent evaluator judges both, never a critic's verdict."""
    before_ok, after_ok = evaluator(before), evaluator(after)
    return {"task_id": task_id, "candidate_correct": before_ok,
            "executed_correct": after_ok,
            "is_repair": before_ok is False and after_ok is True,
            "eligible": after_ok is True, "evaluator": "independent-action-oracle-v1"}


def candidate_from_proposal(proposal, source_task_id, source_round, certificate,
                            exposed_apis, forbidden_literals=()):
    if not certificate.get("eligible") or not certificate.get("is_repair"):
        raise ValueError("A memory requires an independently correct source repair")
    if not isinstance(proposal, dict):
        raise ValueError("Memory proposal must be an object")
    instruction, apis, cues = (proposal.get("instruction"), proposal.get("api_names"),
                              proposal.get("cues", []))
    if not isinstance(instruction, str) or not 15 <= len(instruction.strip()) <= 500:
        raise ValueError("Instruction must contain 15..500 characters")
    if (not isinstance(apis, list) or not apis or not all(isinstance(a, str) for a in apis)
            or not set(apis).issubset(exposed_apis)):
        raise ValueError("Memory scope must use exposed APIs")
    if not isinstance(cues, list) or len(cues) > 5 or any(
            not isinstance(cue, str) or not 2 <= len(cue) <= 40 for cue in cues):
        raise ValueError("Invalid visible cue list")
    text = instruction + " " + " ".join(cues)
    for literal in forbidden_literals:
        literal = str(literal).strip()
        if literal and re.search(r"(?<!\w)" + re.escape(literal) + r"(?!\w)", text, re.I):
            raise ValueError("Source instance value copied into reusable guidance")
    body = {"instruction": instruction.strip(), "api_names": sorted(set(apis)),
            "cues": sorted(set(cues))}
    return RepairMemory("mem-" + digest(body)[:16], **body,
                        source_task_ids=[source_task_id], source_round=source_round,
                        source_certificate=certificate)


class MemoryBank:
    def __init__(self, memories=(), version=0):
        self.memories = list(memories)
        self.version = version
        ids = [m.memory_id for m in self.memories]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate memory ID")

    @property
    def sha256(self):
        return digest({"version": self.version,
                       "memories": [m.to_dict() for m in self.memories]})

    def retrieve(self, messages, tools, mode="validated", top_k=2, max_chars=900):
        if mode not in {"validated", "unfiltered", "unscoped"}:
            raise ValueError("Unknown memory retrieval mode")
        exposed = {(t.get("function") or {}).get("name") for t in tools}
        evidence = " ".join(str(m.get("content") or "") for m in messages).lower()
        selected, used = [], 0
        for memory in sorted(self.memories, key=lambda item: item.memory_id):
            if mode != "unfiltered" and memory.status != "accepted":
                continue
            if mode != "unscoped":
                if not exposed.intersection(memory.api_names):
                    continue
                if memory.cues and not any(cue.lower() in evidence for cue in memory.cues):
                    continue
            visible = json.dumps(memory.visible(), ensure_ascii=False)
            if len(selected) < top_k and used + len(visible) <= max_chars:
                selected.append(memory)
                used += len(visible)
        return selected

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {"schema_version": 1, "version": self.version,
                 "memories": [m.to_dict() for m in self.memories], "sha256": self.sha256}
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("schema_version") != 1:
            raise ValueError("Unsupported memory bank schema")
        bank = cls([RepairMemory.from_dict(row) for row in value["memories"]], value["version"])
        if value.get("sha256") != bank.sha256:
            raise ValueError("Memory bank contents changed")
        return bank


def memory_messages(messages, selected):
    if not selected:
        return list(messages)
    guidance = [memory.visible() for memory in selected]
    return list(messages) + [{"role": "user", "content": (
        "Reusable guidance from earlier tasks (apply only when its conditions fit; "
        "all current arguments must come from this task or its observations):\n"
        + json.dumps(guidance, ensure_ascii=False))}]


def assert_disjoint_splits(groups, fingerprint):
    all_ids, seen_states = set(), {}
    for split, tasks in groups.items():
        for task in tasks:
            meta = task["_q3"]
            if meta["split"] != split or meta["id"] in all_ids:
                raise ValueError("Duplicate or mislabeled task split")
            all_ids.add(meta["id"])
            state_hash = fingerprint(task)
            if state_hash in seen_states:
                raise ValueError("Duplicate public state across split/task IDs")
            seen_states[state_hash] = split


def validate_transfer(memory, probes, generate: Callable, judge: Callable,
                      minimum_pairs=4, harm_weight=2.0):
    """Promote from independent paired development decisions; never inspect test."""
    if any(probe["split"] != "validation" for probe in probes):
        raise ValueError("Memory promotion may only use validation probes")
    if any(probe["task_id"] in memory.source_task_ids for probe in probes):
        raise ValueError("Source and validation overlap")
    scoped = [probe for probe in probes if set(probe["api_names"]).intersection(memory.api_names)]
    rows = []
    bank = MemoryBank([memory])
    for probe in scoped:
        selected = bank.retrieve(probe["messages"], probe["tools"], mode="unfiltered")
        baseline = generate(probe, [])
        guided = generate(probe, selected) if selected else baseline
        rows.append({"task_id": probe["task_id"], "variant": probe["variant"],
                     "memory_applied": bool(selected), "baseline_action": baseline,
                     "memory_action": guided, "baseline_correct": bool(judge(probe, baseline)),
                     "memory_correct": bool(judge(probe, guided))})
    fixes = sum(not row["baseline_correct"] and row["memory_correct"] for row in rows)
    harms = sum(row["baseline_correct"] and not row["memory_correct"] for row in rows)
    variants = {row["variant"] for row in rows}
    utility = fixes - harm_weight * harms
    enough = len(rows) >= minimum_pairs and {"conversion", "already_target"}.issubset(variants)
    accepted = enough and fixes > 0 and harms == 0 and utility > 0
    memory.validation = {"protocol": "independent-transfer-v1", "pairs": len(rows),
                         "fixes": fixes, "harms": harms, "net_utility": utility,
                         "harm_weight": harm_weight, "variants": sorted(variants),
                         "evidence": rows, "state": "complete",
                         "reason": "positive_net_gain_no_observed_harm" if accepted else
                         "insufficient_coverage" if not enough else "no_positive_safe_transfer"}
    memory.status = "accepted" if accepted else "rejected"
    return memory.validation


def validate_bank(bank, probes, generate: Callable, judge: Callable):
    """Check the actual retrieved combinations; single-item safety is insufficient."""
    if any(probe["split"] != "validation" for probe in probes):
        raise ValueError("Bank validation may only use development probes")
    source_ids = {tid for memory in bank.memories for tid in memory.source_task_ids}
    if any(probe["task_id"] in source_ids for probe in probes):
        raise ValueError("Bank source/validation overlap")
    rows = []
    for probe in probes:
        selected = bank.retrieve(probe["messages"], probe["tools"])
        baseline = generate(probe, [])
        guided = generate(probe, selected) if selected else baseline
        rows.append({"task_id": probe["task_id"], "memory_ids": [m.memory_id for m in selected],
                     "baseline_correct": bool(judge(probe, baseline)),
                     "memory_correct": bool(judge(probe, guided)),
                     "baseline_action": baseline, "memory_action": guided})
    fixes = sum(not row["baseline_correct"] and row["memory_correct"] for row in rows)
    harms = sum(row["baseline_correct"] and not row["memory_correct"] for row in rows)
    return {"protocol": "joint-bank-transfer-v1", "bank_sha256": bank.sha256,
            "pairs": len(rows), "fixes": fixes, "harms": harms,
            "accepted": bool(rows) and harms == 0 and (fixes > 0 or not bank.memories),
            "evidence": rows}
