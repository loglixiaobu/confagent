"""Run one API-Bank decision with an optional, frozen experience memory."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math

from collect_new_signals import parse_self, self_messages
from memory_experiment_data import case_tools, clarification_spans
from react_agent import SYSTEM_PROMPT, _THINK_RE
from step_confidence import literal_grounding, schema_valid
from step_monitor import ReviewBudget


NEXT_STEP_PROMPT = SYSTEM_PROMPT + """

For this evaluation, take only the immediate next step. A clarification
question to the user is also allowed as the next step. If asking a question,
reply to the user without calling a tool. A clarification does not complete the
user's task. Otherwise, follow the rules above and call at most one tool."""

PROBLEM_TYPE = "missing_required_parameter"


@dataclass(frozen=True)
class AgentSettings:
    model: str = "Qwen3-8B"
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 20261115
    enable_thinking: bool = True
    max_new_tokens: int = 768
    total_token_cap: int = 4096
    max_checks: int = 1

    def __post_init__(self):
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not 0 <= self.temperature <= 2 or not 0 < self.top_p <= 1:
            raise ValueError("Invalid sampling settings")
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if type(self.total_token_cap) is not int or self.total_token_cap <= self.max_new_tokens:
            raise ValueError("total_token_cap must exceed max_new_tokens")
        if type(self.max_checks) is not int or self.max_checks not in (0, 1):
            raise ValueError("The one-step agent supports zero or one check pass")


class QwenPromptCounter:
    """Render the same Qwen chat template used by the local inference service."""

    def __init__(self, model_path):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)

    def __call__(self, payload):
        template_kwargs = payload.get("extra_body", {}).get("chat_template_kwargs", {})
        token_ids = self.tokenizer.apply_chat_template(
            payload["messages"],
            tools=payload.get("tools"),
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        return len(token_ids)


class RequestBudget:
    """Enforce a cap over all prompt and output tokens for a decision."""

    def __init__(self, total_token_cap, send, count_prompt_tokens):
        if type(total_token_cap) is not int or total_token_cap < 1:
            raise ValueError("total_token_cap must be a positive integer")
        self.total_token_cap = total_token_cap
        self.send = send
        self.count_prompt_tokens = count_prompt_tokens
        self.requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def cost(self):
        return {
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }

    def complete(self, request_id, kind, payload):
        prompt_tokens = self.count_prompt_tokens(payload)
        if type(prompt_tokens) is not int or prompt_tokens < 1:
            raise ValueError("The prompt counter must return a positive integer")
        remaining = self.total_token_cap - self.cost["total_tokens"] - prompt_tokens
        if remaining < 1:
            raise ValueError(f"Token budget exhausted before request {request_id}")
        requested_tokens = payload.get("max_tokens")
        if type(requested_tokens) is not int or requested_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        request = {**payload, "max_tokens": min(requested_tokens, remaining)}

        self.requests += 1
        response = self.send(request_id, kind, request)
        if hasattr(response, "model_dump"):
            response = response.model_dump()
        if not isinstance(response, dict) or not isinstance(response.get("usage"), dict):
            raise ValueError(f"Model returned no token usage for {request_id}")
        usage = response["usage"]
        actual_prompt = usage.get("prompt_tokens")
        actual_completion = usage.get("completion_tokens")
        if (type(actual_prompt) is not int or actual_prompt < 1
                or type(actual_completion) is not int or actual_completion < 0):
            raise ValueError(f"Model returned invalid token usage for {request_id}")
        self.prompt_tokens += actual_prompt
        self.completion_tokens += actual_completion
        if usage.get("total_tokens", actual_prompt + actual_completion) != actual_prompt + actual_completion:
            raise ValueError(f"Model returned inconsistent token usage for {request_id}")
        if actual_prompt != prompt_tokens:
            raise ValueError(
                f"Prompt token count mismatch for {request_id}: local {prompt_tokens}, server {actual_prompt}"
            )
        if actual_completion > request["max_tokens"] or self.cost["total_tokens"] > self.total_token_cap:
            raise ValueError(f"Model exceeded the token limit for {request_id}")
        return response


def _model_inputs(case):
    tools = case_tools(case)
    messages = [
        {"role": "system", "content": NEXT_STEP_PROMPT},
        *deepcopy(case["history"]),
        {"role": "user", "content": "Requirement: " + case["requirement"].strip()},
    ]
    return messages, tools


def matching_rules(rules, tools):
    selected = []
    for rule in rules:
        scope = rule["applicability"]
        if scope["problem_type"] != PROBLEM_TYPE:
            continue
        api_names = set(scope.get("tools", []))
        fields = set(scope.get("required_fields", []))
        for tool in tools:
            function = tool["function"]
            if function["name"] == "ToolSearcher":
                continue
            required = function["parameters"].get("required", [])
            if required and (not api_names or function["name"] in api_names) and fields.issubset(required):
                selected.append(rule)
                break
    return selected


def _memory_text(rules):
    lines = ["\n\nVerified experience for handling required tool parameters:"]
    for rule in rules:
        scope = rule["applicability"]
        selectors = []
        if scope.get("tools"):
            selectors.append("tools " + ", ".join(scope["tools"]))
        if scope.get("required_fields"):
            selectors.append("fields " + ", ".join(scope["required_fields"]))
        scope_text = f" ({'; '.join(selectors)})" if selectors else ""
        lines.append(f"- {rule['handling']}{scope_text}")
    return "\n".join(lines)


def _response_choice(response):
    choices = response.get("choices") or []
    if (not isinstance(choices, list) or len(choices) != 1
            or not isinstance(choices[0], dict) or not isinstance(choices[0].get("message"), dict)):
        raise ValueError("Expected exactly one model choice with a message")
    return choices[0]


def _assistant_text(message):
    text = message.get("content") or ""
    if not isinstance(text, str):
        return ""
    return _THINK_RE.sub("", text).split("</think>")[-1].split("<think>")[0].strip()


def _reject_non_json_constant(value):
    raise ValueError("Invalid JSON constant " + value)


def _parse_finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Tool argument is outside the finite number range")
    return parsed


def _parse_call(call):
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        return {"kind": "other", "error": "invalid_tool_call"}
    api_name = function["name"]
    raw_arguments = function.get("arguments")
    try:
        arguments = json.loads(raw_arguments, parse_constant=_reject_non_json_constant,
                               parse_float=_parse_finite_float)
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object")
    except (TypeError, ValueError):
        return {"kind": "other", "api_name": api_name, "error": "invalid_tool_arguments"}
    return {"kind": "call", "api_name": api_name, "arguments": arguments}


def parse_action(choice):
    message = choice["message"]
    text = _assistant_text(message)
    calls = message.get("tool_calls") or []
    if len(calls) == 1:
        action = _parse_call(calls[0])
        if text:
            action["text"] = text
        return action
    if calls:
        return {
            "kind": "other", "text": text, "error": "multiple_tool_calls",
            "actions": [_parse_call(call) for call in calls],
        }
    if not text and choice.get("finish_reason") == "length":
        return {"kind": "other", "text": "", "error": "truncated"}
    return {"kind": "clarify" if clarification_spans(text) else "other", "text": text}


def _precheck(case, action, tools, limit):
    budget = ReviewBudget(max_checks=limit)
    checks = []
    if budget.claim({"needs_verification": True}):
        schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
        calls = action.get("actions", [action])
        for call in calls:
            if call["kind"] != "call":
                continue
            schema = schemas.get(call["api_name"])
            checks.append({
                "api_name": call["api_name"],
                "schema_valid": schema is not None and schema_valid(call["arguments"], schema),
                "literal_grounding": literal_grounding(call, {
                    "requirement": case["requirement"], "history": case["history"],
                }),
            })
    return {"checks_used": budget.used, "max_checks": budget.max_checks, "calls": checks}


def _seed(task_id, config):
    return config.seed + int(hashlib.sha256(task_id.encode()).hexdigest()[:7], 16)


def run_next_step(case, *, send, count_prompt_tokens, config=AgentSettings(),
                  rules=(), request_id=None):
    """One model decision and a fixed audit pass, using model-visible inputs only."""
    messages, tools = _model_inputs(case)
    rules = matching_rules(rules, tools)
    if rules:
        messages[0]["content"] += _memory_text(rules)
    budget = RequestBudget(config.total_token_cap, send, count_prompt_tokens)
    payload = {
        "model": config.model, "messages": messages, "tools": tools, "tool_choice": "auto",
        "temperature": config.temperature, "top_p": config.top_p,
        "max_tokens": config.max_new_tokens, "seed": _seed(case["task_id"], config),
        "extra_body": {"chat_template_kwargs": {"enable_thinking": config.enable_thinking}},
    }
    response = budget.complete(request_id or case["task_id"], "decision", payload)
    choice = _response_choice(response)
    action = parse_action(choice)
    return {
        "action": action,
        "finish_reason": choice.get("finish_reason"),
        "memory_rule_ids": [rule["rule_id"] for rule in rules],
        "precheck": _precheck(case, action, tools, config.max_checks),
        "cost": budget.cost,
    }


def self_report_for_action(case, action, send, count_prompt_tokens,
                           config=AgentSettings(), request_id=None):
    """Collect the existing self-reported correctness signal for a dev call."""
    budget = RequestBudget(config.total_token_cap, send, count_prompt_tokens)
    if action["kind"] != "call":
        return {"self_report": None, "cost": budget.cost}
    messages, tools = _model_inputs(case)
    response = budget.complete(request_id or case["task_id"] + ":confidence", "self_report", {
        "model": config.model,
        "messages": self_messages({"messages": messages, "tools": tools, "candidate": action}),
        "temperature": 0.0, "top_p": 1.0, "max_tokens": 64,
        "seed": _seed(case["task_id"] + ":confidence", config),
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    })
    score = parse_self(_assistant_text(_response_choice(response)["message"]))
    return {"self_report": score, "cost": budget.cost}
