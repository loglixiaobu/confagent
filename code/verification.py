#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""研究内容二：执行前验证、预算策略与纠错动作。

本模块不依赖 API-Bank 的 gold 标签。传给这里的内容只有用户需求、历史观测、
工具 schema 和模型生成的候选动作，避免验证器偷看 evaluator 信息。
"""

from __future__ import annotations

import json
import random
import re
import time
from decimal import Decimal, InvalidOperation
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from action_semantics import semantic_action_equal, strict_action_equal


@dataclass
class CandidateAction:
    api_name: str
    arguments: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"api_name": self.api_name, "arguments": self.arguments}

    def signature(self) -> str:
        return "%s|%s" % (
            self.api_name,
            json.dumps(self.arguments, sort_keys=True, ensure_ascii=False, default=str),
        )


@dataclass
class UsageCost:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_calls: int = 0
    latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: "UsageCost") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.model_calls += other.model_calls
        self.latency_ms += other.latency_ms

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class ProvenanceResult:
    risk: float
    issues: List[Dict[str, Any]] = field(default_factory=list)
    supported_fields: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.issues

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    verifier: str
    valid: Optional[bool]
    risk: float
    issues: List[Dict[str, Any]] = field(default_factory=list)
    revised_action: Optional[CandidateAction] = None
    usage: UsageCost = field(default_factory=UsageCost)
    raw: Optional[Dict[str, Any]] = None
    available: bool = True

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "verifier": self.verifier,
            "valid": self.valid,
            "risk": self.risk,
            "issues": self.issues,
            "revised_action": self.revised_action.to_dict() if self.revised_action else None,
            "usage": self.usage.to_dict(),
            "available": self.available,
        }
        if self.raw is not None:
            out["raw"] = self.raw
        return out


@dataclass
class BudgetState:
    """预算单位统一为额外模型调用次数。确定性检查不消耗预算。"""

    total: int
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.used)

    def can_spend(self, calls: int = 1) -> bool:
        return calls >= 0 and self.remaining >= calls

    def spend(self, calls: int) -> bool:
        if calls < 0:
            raise ValueError("calls 不能为负")
        if not self.can_spend(calls):
            return False
        self.used += calls
        return True


def usage_from_response(response: Any, latency_ms: int = 0) -> UsageCost:
    usage = getattr(response, "usage", None)
    return UsageCost(
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        model_calls=1,
        latency_ms=latency_ms,
    )


def _flatten_values(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for v in value.values():
            yield from _flatten_values(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _flatten_values(v)
    elif value is not None:
        yield value


def _normalise(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _tokens(value: Any) -> set:
    return {t for t in re.findall(r"[a-z0-9@._:+-]+", _normalise(value)) if len(t) > 2}


def _number(value: Any) -> Optional[Decimal]:
    if isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value))
        return result if result.is_finite() else None
    except (InvalidOperation, ValueError, TypeError):
        return None


def _numbers_in_evidence(values: Iterable[Any]) -> set:
    """Collect scalar numbers and numeric literals embedded in visible text."""
    numbers = set()
    for value in values:
        direct = _number(value)
        if direct is not None:
            numbers.add(direct)
        if isinstance(value, str):
            for literal in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", value):
                parsed = _number(literal)
                if parsed is not None:
                    numbers.add(parsed)
    return numbers


def _structured_time_grounded(value: Any, evidence_text: str) -> bool:
    """Accept a visible date/time even when the model changes separators.

    For example, ``2026-11-03T14:00`` is grounded by the requirement text
    ``2026-11-03 at 14:00``. Every detected date/time component must be visible;
    this does not infer or manufacture a missing component.
    """
    if not isinstance(value, str):
        return False
    # Digit boundaries (rather than ``\b``) also cover the ISO ``dateTtime`` form.
    dates = re.findall(r"(?<!\d)\d{4}-\d{1,2}-\d{1,2}(?!\d)", value)
    times = re.findall(r"(?<!\d)\d{1,2}:\d{2}(?::\d{2})?(?!\d)", value)
    parts = dates + times
    return bool(parts) and all(part.lower() in evidence_text for part in parts)


def _tool_schema(tools: List[Dict[str, Any]], api_name: str) -> Optional[Dict[str, Any]]:
    for tool in tools:
        fn = tool.get("function") or {}
        if fn.get("name") == api_name:
            return fn
    return None


def check_argument_provenance(requirement: str, observations: List[Any],
                              candidate: CandidateAction,
                              tools: List[Dict[str, Any]]) -> ProvenanceResult:
    """确定性参数来源检查；只检查可观察证据，不使用 gold 参数。"""
    schema = _tool_schema(tools, candidate.api_name)
    issues: List[Dict[str, Any]] = []
    supported: List[str] = []
    if schema is None:
        return ProvenanceResult(1.0, [{"kind": "unknown_api", "field": None,
                                      "message": "candidate API is not in the exposed tool set"}], [])

    params = schema.get("parameters") or {}
    required = params.get("required") or []
    properties = params.get("properties") or {}
    args = candidate.arguments
    for name in required:
        if name not in args:
            issues.append({"kind": "missing_argument", "field": name,
                           "message": "required argument is missing"})
    for name in args:
        if name not in properties:
            issues.append({"kind": "unknown_argument", "field": name,
                           "message": "argument is not declared by the tool schema"})

    evidence_values = list(_flatten_values([requirement, observations]))
    evidence_norm = [_normalise(v) for v in evidence_values]
    evidence_numbers = _numbers_in_evidence(evidence_values)
    evidence_tokens = set().union(*(_tokens(v) for v in evidence_values)) if evidence_values else set()
    evidence_text = " ".join(evidence_norm)
    composition_fields = {"content", "message", "body", "description", "query", "keywords", "text"}

    for field_name, value in args.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            issues.append({"kind": "empty_value", "field": field_name,
                           "message": "argument value is empty"})
            continue
        value_norm = _normalise(value)
        exact = any(value_norm == ev or (len(value_norm) >= 4 and value_norm in ev)
                    for ev in evidence_norm)
        numeric = _number(value)
        if numeric is not None and numeric in evidence_numbers:
            exact = True
        overlap = _tokens(value) & evidence_tokens
        compositional = field_name.lower() in composition_fields and bool(overlap)
        if _structured_time_grounded(value, evidence_text):
            compositional = True
        if isinstance(value, (list, tuple)):
            leaves = list(_flatten_values(value))
            compositional = bool(leaves) and all(
                _normalise(leaf) in evidence_norm
                or (_number(leaf) is not None and _number(leaf) in evidence_numbers)
                for leaf in leaves)
        # Safe arithmetic is grounded only when every numeric literal was
        # already visible.  This accepts sums of observed counts without
        # legitimising a fabricated constant or an unsupported /12 conversion.
        if candidate.api_name == "Calculator" and field_name == "formula":
            try:
                from apibank_env import arithmetic_value
                arithmetic_value(value)
                literals = [_number(v) for v in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", str(value))]
                compositional = bool(literals) and all(v in evidence_numbers for v in literals)
            except (ValueError, SyntaxError, ArithmeticError):
                compositional = False
        description = str((properties.get(field_name) or {}).get("description", "")).lower()
        if numeric is not None and "meter" in description:
            km_values = [Decimal(v) * 1000 for v in re.findall(
                r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:km|kilometers?)\b", requirement.lower())]
            if numeric in km_values:
                compositional = True
        # 布尔值和很小的枚举通常由任务语义决定，单纯字符串检索会产生大量假阳性。
        primitive_ok = isinstance(value, bool) or value_norm in {"true", "false", "yes", "no"}
        if exact or compositional or primitive_ok:
            supported.append(field_name)
        else:
            issues.append({
                "kind": "unsupported_value",
                "field": field_name,
                "value": value,
                "message": "value cannot be traced to the requirement or previous observations",
            })

    if not issues:
        risk = 0.0
    else:
        severities = {
            "unknown_api": 1.0, "missing_argument": 0.95, "empty_value": 0.9,
            "unsupported_value": 0.75, "unknown_argument": 0.45,
        }
        risk = max(severities.get(i["kind"], 0.5) for i in issues)
    return ProvenanceResult(risk=risk, issues=issues, supported_fields=supported)


def fuse_risk(token_confidence: Optional[float], self_report: Optional[float],
              consistency: Optional[float], provenance_risk: Optional[float]) -> float:
    """融合多个可用信号。权重只用于初始实验，正式阈值应在 train/dev 上拟合。"""
    parts: List[Tuple[float, float]] = []
    if token_confidence is not None:
        parts.append((0.10, 1.0 - token_confidence))
    if self_report is not None:
        parts.append((0.25, 1.0 - self_report))
    if consistency is not None:
        parts.append((0.25, 1.0 - consistency))
    if provenance_risk is not None:
        parts.append((0.40, provenance_risk))
    if not parts:
        return 0.5
    denom = sum(w for w, _ in parts)
    return min(1.0, max(0.0, sum(w * v for w, v in parts) / denom))


def should_verify(policy: str, risk: float, provenance: ProvenanceResult,
                  budget: BudgetState, threshold: float,
                  rng: random.Random, required_calls: int = 1) -> Tuple[bool, str]:
    if policy == "vanilla":
        return False, "policy=vanilla"
    if policy == "exception":
        return False, "exception policy waits for execution feedback"
    if not budget.can_spend(required_calls):
        return False, "verification budget exhausted"
    if policy == "always":
        return True, "policy=always"
    if policy == "random":
        hit = rng.random() < threshold
        return hit, "random draw < %.3f" % threshold if hit else "random draw rejected"
    if policy == "provenance":
        return (not provenance.valid,
                "argument provenance issue" if not provenance.valid else "all arguments grounded")
    if policy in ("threshold", "fused"):
        return (risk >= threshold,
                "risk %.3f >= threshold %.3f" % (risk, threshold)
                if risk >= threshold else "risk below threshold")
    raise ValueError("未知验证策略: %s" % policy)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def run_critic(client: Any, model: str, messages: List[Dict[str, Any]],
               tools: List[Dict[str, Any]], candidate: CandidateAction,
               provenance: ProvenanceResult, max_tokens: int = 512) -> VerificationResult:
    """用独立结构化 critic 验证并在必要时给出修订动作。"""
    schema = _tool_schema(tools, candidate.api_name)
    evidence = {
        "candidate_action": candidate.to_dict(),
        "candidate_tool_schema": schema,
        "available_tool_schemas": [t.get("function") for t in tools if t.get("function")],
        "deterministic_provenance_check": provenance.to_dict(),
    }
    prompt = (
        "You are the action verifier. Verify the candidate using only the requirement and "
        "tool observations already shown. Never invent a value.\n"
        "Return exactly ONE JSON object and no other text. It must have these three keys:\n"
        '{"valid": true_or_false, "issues": [], "revised_action": null}\n'
        "When a justified correction exists, revised_action must instead be "
        '{"api_name": "...", "arguments": {...}}. Otherwise keep it null.\n'
        "Do not repeat the evidence or tool schema in your answer.\n"
        "EVIDENCE:\n" + json.dumps(evidence, ensure_ascii=False)
    )
    probe = list(messages) + [{"role": "user", "content": prompt}]
    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model, messages=probe, temperature=0.0, max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except Exception as exc:
        latency = int((time.time() - t0) * 1000)
        return VerificationResult(
            verifier="critic", valid=None, risk=provenance.risk,
            issues=[{"kind": "critic_error", "message": "%s: %s" % (type(exc).__name__, exc)}],
            usage=UsageCost(model_calls=1, latency_ms=latency), available=False,
        )
    latency = int((time.time() - t0) * 1000)
    usage = usage_from_response(response, latency)
    text = getattr(response.choices[0].message, "content", "") or ""
    obj = _extract_json(text)
    if obj is None:
        return VerificationResult(
            verifier="critic", valid=None, risk=provenance.risk,
            issues=[{"kind": "invalid_critic_output", "message": "critic did not return valid JSON"}],
            usage=usage, raw={"text": text[:1000]}, available=False,
        )
    revised = None
    ra = obj.get("revised_action")
    if isinstance(ra, dict) and isinstance(ra.get("api_name"), str) and isinstance(ra.get("arguments"), dict):
        revised = CandidateAction(ra["api_name"], ra["arguments"])
    issues = obj.get("issues") if isinstance(obj.get("issues"), list) else []
    valid = obj.get("valid") if isinstance(obj.get("valid"), bool) else None
    if valid is None:
        return VerificationResult(
            verifier="critic", valid=None, risk=provenance.risk,
            issues=[{"kind": "invalid_critic_output",
                     "message": "critic JSON is missing a boolean valid field"}],
            usage=usage, raw=obj, available=False,
        )
    risk = 0.0 if valid is True else max(provenance.risk, 0.75 if issues else 0.5)
    return VerificationResult(
        verifier="critic", valid=valid, risk=risk, issues=issues,
        revised_action=revised, usage=usage, raw=obj,
    )


def run_consistency_vote(client: Any, model: str, messages: List[Dict[str, Any]],
                         tools: List[Dict[str, Any]], candidate: CandidateAction,
                         k: int, temperature: float, max_tokens: int,
                         enable_thinking: bool,
                         seed: Optional[int] = None) -> VerificationResult:
    """在同一状态采样 k 次，以多数动作作为可选修订。"""
    # One slot per requested sample.  A successful response without a parsed
    # action is a genuine disagreement; a transport failure makes the whole
    # primary score unavailable rather than silently shrinking K.
    samples: List[Optional[CandidateAction]] = []
    usage = UsageCost()
    missing_samples = 0
    for sample_idx in range(max(0, k)):
        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, tools=tools, tool_choice="auto",
                temperature=temperature, max_tokens=max_tokens,
                seed=(seed + sample_idx if seed is not None else None),
                extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
            )
        except Exception:
            usage.add(UsageCost(model_calls=1,
                                latency_ms=int((time.time() - t0) * 1000)))
            missing_samples += 1
            continue
        usage.add(usage_from_response(response, int((time.time() - t0) * 1000)))
        calls = getattr(response.choices[0].message, "tool_calls", None) or []
        if not calls:
            samples.append(None)
            continue
        tc = calls[0]
        try:
            args = json.loads(tc.function.arguments or "{}")
        except Exception:
            samples.append(None)
            continue
        if isinstance(args, dict):
            samples.append(CandidateAction(tc.function.name, args))
        else:
            samples.append(None)

    if missing_samples or len(samples) != max(0, k):
        return VerificationResult(
            verifier="consistency", valid=None, risk=1.0,
            issues=[{"kind": "incomplete_samples",
                     "message": "one or more requested samples are missing"}],
            usage=usage, available=False,
            raw={"agreement": None, "strict_agreement": None,
                 "requested_samples": k, "missing_samples": missing_samples},
        )

    parsed = [sample for sample in samples if sample is not None]
    chosen_n = sum(semantic_action_equal(candidate, sample, tools) for sample in parsed)
    strict_n = sum(strict_action_equal(candidate, sample) for sample in parsed)
    agreement = chosen_n / k if k else 0.0
    strict_agreement = strict_n / k if k else 0.0

    clusters: List[Tuple[CandidateAction, int]] = []
    for sample in parsed:
        for idx, (representative, count) in enumerate(clusters):
            if semantic_action_equal(representative, sample, tools):
                clusters[idx] = (representative, count + 1)
                break
        else:
            clusters.append((sample, 1))
    majority, majority_n = max(clusters, key=lambda item: item[1]) if clusters else (None, 0)
    # Do not revise from a plurality produced alongside no-action responses.
    revised = (majority if majority is not None and majority_n > k / 2
               and not semantic_action_equal(candidate, majority, tools) else None)
    return VerificationResult(
        verifier="consistency", valid=agreement >= 0.5, risk=1.0 - agreement,
        issues=[] if agreement >= 0.5 else [{"kind": "low_consistency", "message": "sampled actions disagree"}],
        revised_action=revised, usage=usage,
        raw={"agreement": agreement, "strict_agreement": strict_agreement,
             "n_samples": k, "n_parsed_actions": len(parsed), "n_agree": chosen_n,
             "majority_count": majority_n,
             "sample_actions": [s.to_dict() if s is not None else None for s in samples]},
    )


def correction_outcome(before_correct: Optional[bool], after_correct: Optional[bool],
                       changed: bool) -> str:
    if before_correct is None or after_correct is None:
        return "not_evaluated"
    if before_correct and after_correct:
        return "unchanged_correct"
    if before_correct and not after_correct:
        return "correct_to_error"
    if not before_correct and after_correct:
        return "error_to_correct"
    return "changed_but_still_error" if changed else "unchanged_error"


def grounded_label(provenance: ProvenanceResult,
                   replay_correct: Optional[bool]) -> Optional[bool]:
    """Conservative evidence-grounded label for online Q2 attribution.

    A provenance violation is an observable error even when a fabricated value
    happens to equal hidden replay data.  Conversely, a grounded action rejected
    by the deterministic replay is left unknown instead of being forced wrong.
    This label is evaluator-only and must never enter model messages.
    """
    if not provenance.valid:
        return False
    if replay_correct is True:
        return True
    return None
