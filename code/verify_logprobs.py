#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D1 关键风险验证：vLLM 能否同时返回 tool_call 解析结果 与 token 级 logprob。

这是整个项目的第一道技术关卡：
  - 如果两者可同时获得 -> 置信度信号源可以用 token logprob（最细粒度）
  - 如果冲突           -> 降级为「多采样自一致性 + 自陈置信度」，不依赖 logprob

用法：
    python verify_logprobs.py                      # 用默认地址 localhost:8000
    python verify_logprobs.py --base-url http://localhost:8000/v1 --model Qwen3-8B
"""

import argparse
import json
import math
import sys
from typing import Any, Dict, List

from openai import OpenAI


TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "QueryMeeting",
            "description": "The API for retrieving the meeting details from the user's calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_name": {"type": "string", "description": "Name of the user."}
                },
                "required": ["user_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "EmailReminder",
            "description": "This API sends an email reminder to the user with the meeting details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The content of the email."},
                    "time": {"type": "string", "description": "The time of the meeting."},
                    "location": {"type": "string", "description": "The location of the meeting."},
                    "recipient": {"type": "string", "description": "The email address of the recipient."},
                },
                "required": ["content", "time", "location", "recipient"],
            },
        },
    },
]

USER_MSG = (
    "You are an agent that solves tasks by calling APIs.\n"
    "Task: Query meeting of John and send email reminder to john@example.com.\n"
    "Call the first API you need now."
)


def extract_token_stats(choice) -> Dict[str, Any]:
    """从 logprobs 中提取可用于置信度的统计量。"""
    lp = getattr(choice, "logprobs", None)
    if lp is None:
        return {"available": False}

    content = getattr(lp, "content", None) or []
    tokens, lps, entropies = [], [], []
    for t in content:
        tokens.append(t.token)
        lps.append(t.logprob)
        # 用 top_logprobs 计算该位置的分布熵（语义不确定性的一个代理量）
        top = getattr(t, "top_logprobs", None) or []
        if top:
            ps = [math.exp(x.logprob) for x in top]
            s = sum(ps)
            if s > 0:
                ps = [p / s for p in ps]
                entropies.append(-sum(p * math.log(p + 1e-12) for p in ps))

    if not lps:
        return {"available": False, "reason": "logprobs.content 为空"}

    mean_lp = sum(lps) / len(lps)
    return {
        "available": True,
        "n_tokens": len(lps),
        "mean_logprob": mean_lp,
        "min_logprob": min(lps),
        "geo_mean_prob": math.exp(mean_lp),
        "mean_entropy_topk": (sum(entropies) / len(entropies)) if entropies else None,
        "tokens": tokens,
        "logprobs": lps,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--top-logprobs", type=int, default=5)
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    print("=" * 72)
    print("1) 服务与模型可用性")
    print("=" * 72)
    try:
        models = [m.id for m in client.models.list().data]
        print("   可用模型:", models)
    except Exception as e:
        print("   [FAIL] 无法连接服务:", e)
        print("   提示：先启动 vLLM，见 run_vllm.sh")
        return 2

    print()
    print("=" * 72)
    print("2) 单次请求同时开启 tools + logprobs")
    print("=" * 72)
    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": USER_MSG}],
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=512,
            logprobs=True,
            top_logprobs=args.top_logprobs,
        )
    except Exception as e:
        print("   [FAIL] 请求失败:", type(e).__name__, e)
        print("   -> 若报错与 logprobs/tools 互斥有关，走降级方案（多采样一致性）")
        return 3

    choice = resp.choices[0]
    msg = choice.message

    tool_calls = getattr(msg, "tool_calls", None)
    print("   tool_calls 数量:", len(tool_calls) if tool_calls else 0)
    if tool_calls:
        for tc in tool_calls:
            print("     - name:", tc.function.name)
            print("       arguments:", tc.function.arguments)
    else:
        print("     (未产生工具调用)")
        print("     content:", (msg.content or "")[:200])

    print()
    stats = extract_token_stats(choice)
    print("   logprobs 统计:")
    for k, v in stats.items():
        if k in ("tokens", "logprobs"):
            continue
        print("     %-20s %s" % (k, v))
    if stats.get("available"):
        print("     前 20 个 token:", stats["tokens"][:20])

    print()
    print("=" * 72)
    print("3) 结论")
    print("=" * 72)
    ok_tools = bool(tool_calls)
    ok_lp = bool(stats.get("available"))
    print("   tool_call 解析 :", "OK" if ok_tools else "未触发（可能是模型选择直接回答）")
    print("   token logprob  :", "OK" if ok_lp else "FAIL")
    if ok_lp:
        print()
        print("   ==> 两者可同时获得。置信度信号源可直接使用 token logprob。")
        print("       建议提取：mean_logprob / min_logprob / geo_mean_prob / mean_entropy_topk")
        return 0
    print()
    print("   ==> 未拿到 logprob，需降级为「多采样自一致性 + 自陈置信度」方案。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
