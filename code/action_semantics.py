#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared semantic equality for Q2 candidate actions.

The frozen signal-collection artifacts keep their original implementation and
hashes.  This module mirrors that published protocol for new online runs so the
consistency verifier does not fall back to raw JSON string equality.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

from apibank_env import _norm_time, arithmetic_value, typed_equal


def _parts(action: Any) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    if isinstance(action, dict):
        name = action.get("api_name")
        args = action.get("arguments")
    else:
        name = getattr(action, "api_name", None)
        args = getattr(action, "arguments", None)
    return (name if isinstance(name, str) else None,
            args if isinstance(args, dict) else None)


def _parameter_metadata(tools: Iterable[Dict[str, Any]], api_name: str) -> Dict[str, Any]:
    """Accept either raw API-Bank documents or OpenAI function schemas."""
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(fn, dict) and fn.get("name") == api_name:
            return ((fn.get("parameters") or {}).get("properties") or {})
        if isinstance(tool, dict) and tool.get("name") == api_name:
            return tool.get("input_parameters") or {}
    return {}


def strict_action_equal(left: Any, right: Any) -> bool:
    """Exact action equality, ignoring only object key order."""
    left_name, left_args = _parts(left)
    right_name, right_args = _parts(right)
    return left_name is not None and left_name == right_name and left_args == right_args


def semantic_action_equal(left: Any, right: Any,
                          tools: Iterable[Dict[str, Any]]) -> bool:
    """Compare actions using the frozen new-signal semantic protocol."""
    left_name, left_args = _parts(left)
    right_name, right_args = _parts(right)
    if left_name is None or left_name != right_name or left_args is None or right_args is None:
        return False
    if set(left_args) != set(right_args):
        return False

    metadata = _parameter_metadata(tools, left_name)
    for key, left_value in left_args.items():
        right_value = right_args[key]
        if left_name == "EmailReminder" and key == "content":
            if not isinstance(right_value, str) or not right_value.strip():
                return False
            continue
        if left_name == "Calculator" and key == "formula":
            try:
                if arithmetic_value(left_value) == arithmetic_value(right_value):
                    continue
            except (ValueError, SyntaxError, ArithmeticError):
                pass
        if (left_name == "PersonalInfoUpdate" and key == "address"
                and isinstance(left_value, str) and isinstance(right_value, str)
                and left_value.strip().rstrip(".") == right_value.strip().rstrip(".")):
            continue
        kind = metadata.get(key, {}).get("type", "str")
        if not typed_equal(_norm_time(left_value), _norm_time(right_value), kind):
            return False
    return True
