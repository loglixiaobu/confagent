#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ReAct 循环 + 步骤级置信度提取（研究内容一的基础设施）。

【本文件最重要的设计决策】
--------------------------------------------------------------------
实测确认：vLLM 开启 tool calling 时，**工具调用段是约束解码生成的**，
`<tool_call>{"name": ..., "arguments": ...}</tool_call>` 整段的每个 token
logprob 恒为 0.0000（概率 ≈ 1），**不含任何不确定性信息**。

    => 不能用工具调用参数的 logprob 作为步骤置信度。
    => 步骤级置信度必须从 **推理段（<think> ... </think>）** 提取。

（验证脚本：probes/probe_logprob_structure.py、probes/probe_two_tools.py）

【另外两个已修复的坑】
1. 历史回灌时若保留 `<think>` 段，模型会认为"已经想过了"，直接吐 tool_call，
   导致推理段只剩 1–2 个 token，logprob ≈ 0 → 置信度假性变成 1.000。
   => 回灌历史前**剥离 `<think>` 段**，强制模型每步重新思考。
   => 并对推理段过短的步骤标记 `degenerate`，置信度置为 None（校准时应排除）。

2. 模型在同一个错误调用上无限重试直到 max_steps。
   => 加入重复调用熔断：连续重复 2 次给出提示，连续重复 3 次直接终止。

【推理段提取的候选置信度特征】（后续用校准实验比较哪个最可校准）
    geo_mean_prob       几何平均概率 = exp(mean_logprob)，天然落在 [0,1]（默认主置信度）
    p10_prob            对数概率 10 分位对应的概率（比 min 更稳健）
    min_prob            最差 token 的概率
    tail_geo_mean_prob  最后 20 个 token 的几何平均概率（"收尾信心"）
    frac_highprob       高概率 token 占比
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apibank_env import APIBankMockEnv, load_level3  # noqa: E402
from trajectory import Trajectory, TrajectoryStep, TrajectoryWriter  # noqa: E402

# 推理段少于这么多 token 视为"没有真正思考"，置信度不可用
MIN_THINK_TOKENS = 5

# 连续重复同一调用多少次后终止
MAX_REPEAT = 3

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


# ---------------------------------------------------------------------------
@dataclass
class AgentConfig:
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "Qwen3-8B"

    max_steps: int = 12
    max_tokens: int = 4096
    temperature: float = 0.0
    top_logprobs: int = 5

    enable_thinking: bool = True
    self_report: bool = False
    # 多采样一致性：在同一对话状态下采样 k 次，统计与贪心动作一致的比例。
    # k=0 表示关闭（每步会额外产生 k 次模型调用，成本显著）。
    consistency_k: int = 0
    consistency_temp: float = 0.7
    verbose: bool = False

    seed: int = 0
    round_idx: int = 0
    method: str = "vanilla_react"


SYSTEM_PROMPT = """You are an AI assistant that completes a user's requirement by calling APIs step by step.

Rules:
1. At each step, call EXACTLY ONE tool.
2. The available APIs are listed in your tool set. If you need an API that is not listed, \
call `ToolSearcher` with keywords describing the capability you need.
3. Before each call, verify every argument value comes from the requirement or from a previous \
API output. Never invent values.
4. If a call fails, READ the error message carefully and change the arguments before retrying. \
Do not repeat the same failing call.
5. Only when EVERY part of the requirement has been carried out, reply with a final answer \
and DO NOT call any tool. Before finishing, re-read the requirement clause by clause and \
confirm that each requested action has actually been executed by an API call.

Be concise in your final answer."""


# ---------------------------------------------------------------------------
# 置信度特征提取
# ---------------------------------------------------------------------------
def extract_confidence_features(choice) -> Dict[str, Any]:
    """只统计推理段，排除被约束解码强制的工具调用段。"""
    lp = getattr(choice, "logprobs", None)
    content = getattr(lp, "content", None) if lp is not None else None
    if not content:
        return {"available": False, "reason": "no logprobs"}

    think_start = think_end = tool_start = None
    for i, t in enumerate(content):
        tok = t.token
        if think_start is None and "<think>" in tok:
            think_start = i + 1
        if think_end is None and "</think>" in tok:
            think_end = i
        if tool_start is None and "<tool_call>" in tok:
            tool_start = i
            break

    if think_start is not None and think_end is not None and think_end > think_start:
        span, span_name = content[think_start:think_end], "think"
    elif tool_start is not None:
        span, span_name = content[:tool_start], "pre_tool"
    else:
        span, span_name = content, "all"

    lps = [t.logprob for t in span if getattr(t, "logprob", None) is not None]
    if not lps:
        return {"available": False, "reason": "empty span (%s)" % span_name}

    entropies: List[float] = []
    for t in span:
        top = getattr(t, "top_logprobs", None) or []
        if len(top) >= 2:
            ps = [math.exp(x.logprob) for x in top]
            s = sum(ps)
            if s > 0:
                ps = [p / s for p in ps]
                entropies.append(-sum(p * math.log(p + 1e-12) for p in ps))

    mean_lp = sum(lps) / len(lps)
    srt = sorted(lps)
    p10 = srt[max(0, int(len(srt) * 0.10) - 1)]
    tail = lps[-20:] if len(lps) >= 20 else lps

    feats: Dict[str, Any] = {
        "available": True,
        "span": span_name,
        "n_tokens": len(lps),
        # 候选置信度（均在 [0,1]）
        "geo_mean_prob": math.exp(mean_lp),
        "p10_prob": math.exp(p10),
        "min_prob": math.exp(min(lps)),
        "tail_geo_mean_prob": math.exp(sum(tail) / len(tail)),
        "frac_highprob": sum(1 for x in lps if x > -0.1) / len(lps),
        # 原始统计
        "mean_logprob": mean_lp,
        "min_logprob": min(lps),
        "p10_logprob": p10,
        "frac_lowprob": sum(1 for x in lps if x < -1.0) / len(lps),
        "n_lowprob_tokens": sum(1 for x in lps if x < -1.0),
        "mean_entropy_topk": (sum(entropies) / len(entropies)) if entropies else None,
    }

    if len(lps) < MIN_THINK_TOKENS:
        # 推理段过短 = 模型没有真正思考，此时 logprob 无意义，不能当作"高置信度"
        feats["degenerate"] = True
        feats["confidence"] = None
        feats["confidence_source"] = None
    else:
        feats["degenerate"] = False
        feats["confidence"] = feats["geo_mean_prob"]
        feats["confidence_source"] = "token_logprob"
    return feats


def _action_sig(name: str, args_raw: str) -> str:
    """把一次动作规范化为可比较的签名（解析 JSON 后按 key 排序）。"""
    try:
        obj = json.loads(args_raw or "{}")
        return "%s|%s" % (name, json.dumps(obj, sort_keys=True, ensure_ascii=False))
    except Exception:
        return "%s|%s" % (name, (args_raw or "").strip())


def measure_consistency(client: OpenAI, cfg: AgentConfig,
                        messages: List[Dict[str, Any]], tools: List[Dict[str, Any]],
                        chosen_sig: str, k: int) -> Dict[str, Any]:
    """多采样一致性信号（消融 A1 的第三种信号源）。

    做法：在**同一对话状态**下，以 temperature>0 重新采样 k 次，
    统计与贪心动作（temperature=0 得到的那次）一致的比例。

    直觉：如果模型对这一步真有把握，不同采样路径会收敛到同一个动作；
    如果它在猜，采样结果会发散。这是一个**不依赖 token logprob** 的信号，
    因此对"约束解码导致 logprob 无信息"的情形特别有价值。
    """
    if k <= 0:
        return {"available": False}
    agree = 0
    sigs: List[str] = []
    for _ in range(k):
        try:
            r = client.chat.completions.create(
                model=cfg.model, messages=messages, tools=tools, tool_choice="auto",
                temperature=cfg.consistency_temp, max_tokens=cfg.max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": cfg.enable_thinking}},
            )
        except Exception:
            continue
        tcs = getattr(r.choices[0].message, "tool_calls", None) or []
        if tcs:
            s = _action_sig(tcs[0].function.name, tcs[0].function.arguments)
        else:
            s = "__no_tool__|"
        sigs.append(s)
        if s == chosen_sig:
            agree += 1
    if not sigs:
        return {"available": False}
    return {
        "available": True,
        "consistency": agree / len(sigs),
        "n_samples": len(sigs),
        "n_agree": agree,
        "sample_sigs": sigs,
    }


def ask_self_report(client: OpenAI, cfg: AgentConfig, messages: List[Dict[str, Any]],
                    action_desc: str) -> Optional[float]:
    """自陈置信度（消融 A1 的第二种信号源）。"""
    probe = messages + [
        {"role": "assistant", "content": "I am about to execute: %s" % action_desc},
        {"role": "user", "content": ("On a scale from 0.0 to 1.0, how confident are you that this "
                                     "action is CORRECT (right API, right arguments)? "
                                     "Reply with ONLY a number.")},
    ]
    try:
        r = client.chat.completions.create(
            model=cfg.model, messages=probe, temperature=0.0, max_tokens=64,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        for tok in (r.choices[0].message.content or "").replace(",", " ").split():
            try:
                v = float(tok)
                if 0.0 <= v <= 1.0:
                    return v
                if 1.0 < v <= 100.0:
                    return v / 100.0
            except ValueError:
                continue
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# 单任务执行
# ---------------------------------------------------------------------------
def run_task(task: Dict[str, Any], cfg: AgentConfig, client: OpenAI,
             task_index: int = 0,
             distractors: Optional[List[Dict[str, Any]]] = None) -> Trajectory:
    env = APIBankMockEnv(task, distractor_apis=distractors)
    task_id = "l3-%03d" % task_index

    traj = Trajectory(
        task_id=task_id, model=cfg.model, seed=cfg.seed,
        round_idx=cfg.round_idx, method=cfg.method,
        config={"max_steps": cfg.max_steps, "temperature": cfg.temperature,
                "enable_thinking": cfg.enable_thinking, "self_report": cfg.self_report},
    )

    tool_searcher = {
        "type": "function",
        "function": {
            "name": "ToolSearcher",
            "description": ("Search for a suitable API by keywords. Returns the API name, "
                            "description and its input parameters."),
            "parameters": {
                "type": "object",
                "properties": {"keywords": {"type": "string",
                                            "description": "Keywords describing the capability you need."}},
                "required": ["keywords"],
            },
        },
    }
    from apibank_env import api_schema_to_tool

    # 预解锁本任务涉及的全部 API：ToolSearcher 作为**可选**检索动作保留。
    # 理由：若强制先检索才能解锁，模型常常查完第一个 API 就认为任务完成，
    #       导致任务几乎必然失败、拿不到有意义的置信度样本。
    unlocked: Dict[str, Dict[str, Any]] = dict(env.tool_registry)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Requirement: %s" % task.get("requirement", "").strip()},
    ]

    last_sig: Optional[str] = None
    repeat_count = 0
    stop_reason = "max_steps"

    for _ in range(cfg.max_steps):
        tools = [tool_searcher] + [api_schema_to_tool(a) for a in unlocked.values()]
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=cfg.model, messages=messages, tools=tools, tool_choice="auto",
                temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                logprobs=True, top_logprobs=cfg.top_logprobs, seed=cfg.seed,
                extra_body={"chat_template_kwargs": {"enable_thinking": cfg.enable_thinking}},
            )
        except Exception as e:
            stop_reason = "model_error:%s" % type(e).__name__
            if cfg.verbose:
                print("   [model error] %s: %s" % (type(e).__name__, e))
            break

        latency_ms = int((time.time() - t0) * 1000)
        choice = resp.choices[0]
        msg = choice.message
        usage = getattr(resp, "usage", None)
        step_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        feats = extract_confidence_features(choice)
        tool_calls = getattr(msg, "tool_calls", None) or []

        if not tool_calls:
            traj.final_answer = msg.content or ""
            stop_reason = "final_answer"
            break

        tc = tool_calls[0]
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except Exception:
            args = {"_raw": tc.function.arguments}

        action_type = "tool_search" if name == "ToolSearcher" else "tool_call"

        # --- 重复调用熔断 ---
        sig = "%s|%s" % (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
        if sig == last_sig:
            repeat_count += 1
        else:
            repeat_count = 0
            last_sig = sig

        res = env.call(name, args)

        if name == "ToolSearcher" and isinstance(res.observation, dict) and res.observation.get("name"):
            unlocked[res.observation["name"]] = res.observation

        self_rep = ask_self_report(client, cfg, messages, "%s(%s)" % (
            name, json.dumps(args, ensure_ascii=False))) if cfg.self_report else None

        # 多采样一致性：在同一状态下重采样 k 次
        cons = (measure_consistency(client, cfg, messages, tools,
                                    _action_sig(name, tc.function.arguments),
                                    cfg.consistency_k)
                if cfg.consistency_k > 0 else {"available": False})

        conf = feats.get("confidence") if feats.get("available") else None
        raw = {k: v for k, v in feats.items()} if feats.get("available") else None
        if raw is None:
            raw = {}
        if self_rep is not None:
            raw["self_report"] = self_rep
        if cons.get("available"):
            raw["consistency"] = cons["consistency"]
            raw["consistency_n"] = cons["n_samples"]
            raw["consistency_agree"] = cons["n_agree"]
        if not raw:
            raw = None

        traj.add(TrajectoryStep(
            task_id=task_id, step_id=0, action_type=action_type,
            action_input={"api_name": name, "arguments": args},
            observation=res.observation if res.observation is not None else {"error": res.exception},
            confidence=conf,
            confidence_source=(feats.get("confidence_source") if conf is not None else None),
            verified=False, correction=None,
            cost_tokens=step_tokens, cost_calls=1,
            matched_gold=res.matched_gold,
            strict_matched_gold=res.strict_matched_gold,
            failure_type=res.failure_type,
            latency_ms=latency_ms,
            raw_logprobs=raw,
        ))

        # --- 回灌历史：**剥离 <think> 段**，强制每步重新思考 ---
        clean_content = _THINK_RE.sub("", msg.content or "").strip()
        messages.append({
            "role": "assistant", "content": clean_content,
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": name, "arguments": tc.function.arguments}}],
        })
        messages.append({
            "role": "tool", "tool_call_id": tc.id,
            "content": json.dumps(
                res.observation if res.observation is not None else {"error": res.exception},
                ensure_ascii=False, default=str)[:4000],
        })

        if cfg.verbose:
            print("   step%-2d %-14s matched=%-5s conf=%-6s n_tok=%s%s" % (
                len(traj.steps), name, res.matched_gold,
                ("%.3f" % conf) if conf is not None else "None",
                (feats.get("n_tokens") if feats.get("available") else "-"),
                "  [DEGENERATE]" if feats.get("degenerate") else ""))

        if env.success():
            traj.final_answer = clean_content
            stop_reason = "success"
            break

        if repeat_count >= MAX_REPEAT - 1:
            stop_reason = "repeated_call"
            break

        if repeat_count == 1:
            messages.append({"role": "user", "content": (
                "The previous call failed with the same arguments. Read the error message again "
                "and change the arguments before retrying.")})

    traj.final_success = env.success()
    traj.config["stop_reason"] = stop_reason
    return traj


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="ReAct 循环 + 步骤级置信度提取")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / 'data/api-bank'))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / 'outputs/traj_vanilla.jsonl'))
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--self-report", action="store_true")
    ap.add_argument("--consistency-k", type=int, default=0,
                    help="多采样一致性的采样次数（0=关闭；每步额外 k 次模型调用）")
    ap.add_argument("--consistency-temp", type=float, default=0.7)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = AgentConfig(
        base_url=args.base_url, model=args.model, max_steps=args.max_steps,
        temperature=args.temperature, enable_thinking=not args.no_thinking,
        self_report=args.self_report, consistency_k=args.consistency_k,
        consistency_temp=args.consistency_temp,
        seed=args.seed, verbose=args.verbose,
    )
    client = OpenAI(base_url=cfg.base_url, api_key="EMPTY")
    tasks = load_level3(args.data_dir, "test")[args.offset: args.offset + args.limit]

    print("=" * 72)
    print("ReAct | model=%s | thinking=%s | tasks=%d | max_steps=%d"
          % (cfg.model, cfg.enable_thinking, len(tasks), cfg.max_steps))
    print("=" * 72)

    writer = TrajectoryWriter(args.out)
    t_start = time.time()
    n_ok = 0
    for i, task in enumerate(tasks):
        print("\n[%d/%d] %s" % (i + 1, len(tasks), task.get("requirement", "").strip()[:70]))
        traj = run_task(task, cfg, client, task_index=args.offset + i)
        writer.write(traj)
        n_ok += bool(traj.final_success)
        n_err = sum(1 for s in traj.steps if s.matched_gold is False)
        n_deg = sum(1 for s in traj.steps
                    if (s.raw_logprobs or {}).get("degenerate"))
        confs = [s.confidence for s in traj.steps if s.confidence is not None]
        print("   -> steps=%d success=%s wrong=%d degenerate=%d stop=%s"
              % (traj.n_steps, traj.final_success, n_err, n_deg,
                 traj.config.get("stop_reason")))
        print("      conf: %s" % (", ".join("%.3f" % c for c in confs) or "-"))

    dt = time.time() - t_start
    print()
    print("=" * 72)
    print("完成 %d 任务，成功 %d，用时 %.1f 秒（%.1f 秒/任务）"
          % (len(tasks), n_ok, dt, dt / max(len(tasks), 1)))
    print("轨迹:", args.out)


if __name__ == "__main__":
    main()
