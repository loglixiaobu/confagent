"""Small OpenAI-compatible facade backed by a local Transformers model.

This module implements only the response surface used by code/react_agent.py.
It intentionally does not emulate an HTTP server or vLLM.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def _install_torch_autocast_compatibility() -> None:
    """Bridge the Torch 2.2 / Transformers 5 autocast signature mismatch.

    Transformers 5 calls ``torch.is_autocast_enabled(device_type)`` while the
    CPU-only Torch 2.2 build available on this smoke-test machine exposes the
    older zero-argument form. Keep this compatibility local to this process;
    the production vLLM path and project code remain untouched.
    """
    original = torch.is_autocast_enabled
    try:
        original("cpu")
        return
    except TypeError:
        pass

    def compatible(device_type: Optional[str] = None) -> bool:
        if device_type == "cpu" and hasattr(torch, "is_autocast_cpu_enabled"):
            return bool(torch.is_autocast_cpu_enabled())
        return bool(original())

    torch.is_autocast_enabled = compatible  # type: ignore[assignment]


_install_torch_autocast_compatibility()


def _namespace(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def _tool_calls(text: str, tools: Optional[List[Dict[str, Any]]]) -> List[Any]:
    if not tools:
        return []
    allowed = {
        str((tool.get("function") or {}).get("name"))
        for tool in tools
        if (tool.get("function") or {}).get("name")
    }
    payloads: List[Dict[str, Any]] = []
    for match in _TOOL_CALL_RE.finditer(text or ""):
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            payloads.append(value)

    # Tiny models occasionally omit the XML wrapper but still return the exact
    # Qwen JSON object. Accept that narrow form without treating arbitrary prose
    # as a function call.
    if not payloads:
        cleaned = _THINK_RE.sub("", text or "").strip()
        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict) and value.get("name") in allowed:
            payloads.append(value)

    result = []
    for item in payloads[:1]:
        name = item.get("name")
        arguments = item.get("arguments", {})
        if name not in allowed or not isinstance(arguments, dict):
            continue
        result.append(_namespace(
            id="call_" + uuid.uuid4().hex[:12],
            type="function",
            function=_namespace(
                name=name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            ),
        ))
    return result


class _Completions:
    def __init__(self, owner: "LocalTransformersClient") -> None:
        self.owner = owner

    def create(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        seed: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Any:
        if seed is not None:
            torch.manual_seed(seed)
        template_kwargs = ((extra_body or {}).get("chat_template_kwargs") or {})
        enable_thinking = bool(template_kwargs.get("enable_thinking", False))

        template_args: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        if tools:
            template_args["tools"] = tools
        prompt = self.owner.tokenizer.apply_chat_template(messages, **template_args)
        encoded = self.owner.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.owner.max_input_tokens,
        )
        input_tokens = int(encoded["input_ids"].shape[-1])
        generated_limit = max(8, min(int(max_tokens), self.owner.max_new_tokens))
        generation: Dict[str, Any] = {
            "max_new_tokens": generated_limit,
            "pad_token_id": self.owner.tokenizer.eos_token_id,
        }
        if temperature and temperature > 0:
            generation.update(do_sample=True, temperature=max(float(temperature), 1e-5), top_p=0.9)
        else:
            generation["do_sample"] = False

        started = time.perf_counter()
        with torch.inference_mode():
            output = self.owner.model.generate(**encoded, **generation)
        elapsed = time.perf_counter() - started
        new_ids = output[0, input_tokens:]
        text = self.owner.tokenizer.decode(new_ids, skip_special_tokens=False)
        for token in (self.owner.tokenizer.eos_token or "", "<|im_end|>", "<|endoftext|>"):
            if token:
                text = text.replace(token, "")
        text = text.strip()
        calls = _tool_calls(text, tools)
        content = _TOOL_CALL_RE.sub("", text).strip() if calls else text
        completion_tokens = int(new_ids.shape[-1])
        print(
            "    [local-model] input=%d output=%d time=%.1fs parsed_tool=%s"
            % (input_tokens, completion_tokens, elapsed,
               calls[0].function.name if calls else "none"),
            flush=True,
        )
        message = _namespace(content=content, tool_calls=calls or None)
        choice = _namespace(message=message, logprobs=None, finish_reason="tool_calls" if calls else "stop")
        return _namespace(
            id="local-" + uuid.uuid4().hex,
            choices=[choice],
            usage=_namespace(
                prompt_tokens=input_tokens,
                completion_tokens=completion_tokens,
                total_tokens=input_tokens + completion_tokens,
            ),
        )


class LocalTransformersClient:
    """Drop-in client for the subset of ``client.chat.completions`` we use."""

    def __init__(
        self,
        model_dir: str,
        *,
        max_input_tokens: int = 4096,
        max_new_tokens: int = 192,
        cpu_threads: int = 8,
        dtype: str = "float32",
    ) -> None:
        path = Path(model_dir).resolve()
        if not path.exists():
            raise FileNotFoundError("Local model directory does not exist: %s" % path)
        torch.set_num_threads(max(1, cpu_threads))
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        if dtype not in {"float32", "auto"}:
            raise ValueError("dtype must be 'float32' or 'auto'")
        load_dtype: Any = torch.float32 if dtype == "float32" else "auto"
        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.tokenizer.truncation_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path),
            local_files_only=True,
            dtype=load_dtype,
        )
        self.model.eval()
        self.chat = _namespace(completions=_Completions(self))
