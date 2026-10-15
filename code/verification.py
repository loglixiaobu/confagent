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
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


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
    evidence_tokens = set().union(*(_tokens(v) for v in evidence_values)) if evidence_values else set()
    composition_fields = {"content", "message", "body", "description", "query", "keywords", "text"}

    for field_name, value in args.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            issues.append({"kind": "empty_value", "field": field_name,
                           "message": "argument value is empty"})
            continue
        value_norm = _normalise(value)
        exact = any(value_norm == ev or (len(value_norm) >= 4 and value_norm in ev)
                    for ev in evidence_norm)
        overlap = _tokens(value) & evidence_tokens
        compositional = field_name.lower() in composition_fields and bool(overlap)
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
                  rng: random.Random) -> Tuple[bool, str]:
    if policy == "vanilla":
        return False, "policy=vanilla"
    if policy == "exception":
        return False, "exception policy waits for execution feedback"
    if not budget.can_spend(1):
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
    prompt = {
        "candidate_action": candidate.to_dict(),
        "candidate_tool_schema": schema,
        "available_tool_schemas": [t.get("function") for t in tools if t.get("function")],
        "deterministic_provenance_check": provenance.to_dict(),
        "instruction": (
            "Verify the candidate using only the requirement and tool observations above. "
            "Never invent a value. Return JSON only with keys valid, issues, and revised_action. "
            "revised_action must be null when no justified correction is available, otherwise "
            "it must contain api_name and arguments."
        ),
    }
    probe = list(messages) + [{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}]
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
    samples: List[CandidateAction] = []
    usage = UsageCost()
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
            continue
        usage.add(usage_from_response(response, int((time.time() - t0) * 1000)))
        calls = getattr(response.choices[0].message, "tool_calls", None) or []
        if not calls:
            continue
        tc = calls[0]
        try:
            args = json.loads(tc.function.arguments or "{}")
        except Exception:
            continue
        if isinstance(args, dict):
            samples.append(CandidateAction(tc.function.name, args))

    if not samples:
        return VerificationResult(
            verifier="consistency", valid=None, risk=1.0,
            issues=[{"kind": "no_valid_samples", "message": "no sampled tool action was parseable"}],
            usage=usage, available=False,
        )
    counts = Counter(s.signature() for s in samples)
    majority_sig, majority_n = counts.most_common(1)[0]
    majority = next(s for s in samples if s.signature() == majority_sig)
    chosen_n = counts.get(candidate.signature(), 0)
    agreement = chosen_n / len(samples)
    revised = majority if majority_sig != candidate.signature() and majority_n > chosen_n else None
    return VerificationResult(
        verifier="consistency", valid=agreement >= 0.5, risk=1.0 - agreement,
        issues=[] if agreement >= 0.5 else [{"kind": "low_consistency", "message": "sampled actions disagree"}],
        revised_action=revised, usage=usage,
        raw={"agreement": agreement, "n_samples": len(samples), "n_agree": chosen_n,
             "majority_count": majority_n,
             "sample_actions": [s.to_dict() for s in samples]},
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
