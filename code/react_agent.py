#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ReAct 循环 + 步骤级置信度提取 + 预算约束验证纠错闭环。

研究内容二的执行顺序是：候选动作 -> 风险/预算决策 -> 执行前验证与修订 ->
工具执行 -> 环境反馈局部重试/重规划。gold 标签只用于轨迹评估，不进入模型上下文。

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
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apibank_env import APIBankMockEnv, load_level3  # noqa: E402
from trajectory import Trajectory, TrajectoryStep, TrajectoryWriter  # noqa: E402
from verification import (BudgetState, CandidateAction, UsageCost, VerificationResult,  # noqa: E402
                          check_argument_provenance, correction_outcome,
                          fuse_risk, run_consistency_vote, run_critic,
                          should_verify, usage_from_response)

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

    # 研究内容二：验证预算以“额外模型调用次数”为统一单位。
    verify_policy: str = "vanilla"  # vanilla/always/random/exception/provenance/threshold/fused
    verifier: str = "critic"        # critic/consistency/hybrid
    verify_budget: int = 0
    risk_threshold: float = 0.60
    replan_after: int = 2
    max_retries: int = 3
    feedback_mode: str = "realistic"  # realistic/diagnostic
    verbose: bool = False

    seed: int = 0
    round_idx: int = 0
    method: str = "vanilla_react"

    def __post_init__(self) -> None:
        if self.verify_policy not in {
            "vanilla", "always", "random", "exception", "provenance", "threshold", "fused"
        }:
            raise ValueError("未知 verify_policy: %s" % self.verify_policy)
        if self.verifier not in {"critic", "consistency", "hybrid"}:
            raise ValueError("未知 verifier: %s" % self.verifier)
        if self.verify_budget < 0:
            raise ValueError("verify_budget 不能为负")
        if not 0.0 <= self.risk_threshold <= 1.0:
            raise ValueError("risk_threshold 必须位于 [0,1]")
        if self.feedback_mode not in {"realistic", "diagnostic"}:
            raise ValueError("feedback_mode 必须是 realistic 或 diagnostic")


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


def ask_self_report(client: OpenAI, cfg: AgentConfig, messages: List[Dict[str, Any]],
                    action_desc: str) -> tuple[Optional[float], UsageCost]:
    """自陈置信度及其真实调用成本（消融 A1 的第二种信号源）。"""
    probe = messages + [
        {"role": "assistant", "content": "I am about to execute: %s" % action_desc},
        {"role": "user", "content": ("On a scale from 0.0 to 1.0, how confident are you that this "
                                     "action is CORRECT (right API, right arguments)? "
                                     "Reply with ONLY a number.")},
    ]
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=cfg.model, messages=probe, temperature=0.0, max_tokens=64,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        cost = usage_from_response(r, int((time.time() - t0) * 1000))
        for tok in (r.choices[0].message.content or "").replace(",", " ").split():
            try:
                v = float(tok)
                if 0.0 <= v <= 1.0:
                    return v, cost
                if 1.0 < v <= 100.0:
                    return v / 100.0, cost
            except ValueError:
                continue
    except Exception:
        return None, UsageCost(model_calls=1, latency_ms=int((time.time() - t0) * 1000))
    return None, cost


# ---------------------------------------------------------------------------
# 单任务执行
# ---------------------------------------------------------------------------
def run_task(task: Dict[str, Any], cfg: AgentConfig, client: OpenAI,
             task_index: int = 0,
             distractors: Optional[List[Dict[str, Any]]] = None) -> Trajectory:
    env = APIBankMockEnv(task, distractor_apis=distractors,
                         feedback_mode=cfg.feedback_mode)
    task_id = "l3-%03d" % task_index

    traj = Trajectory(
        task_id=task_id, model=cfg.model, seed=cfg.seed,
        round_idx=cfg.round_idx, method=cfg.method,
        config={"max_steps": cfg.max_steps, "temperature": cfg.temperature,
                "enable_thinking": cfg.enable_thinking, "self_report": cfg.self_report,
                "consistency_k": cfg.consistency_k,
                "consistency_temp": cfg.consistency_temp,
                "verify_policy": cfg.verify_policy, "verifier": cfg.verifier,
                "verify_budget": cfg.verify_budget,
                "risk_threshold": cfg.risk_threshold,
                "replan_after": cfg.replan_after, "max_retries": cfg.max_retries,
                "feedback_mode": cfg.feedback_mode,
                "distractors": len(distractors or [])},
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
    budget = BudgetState(max(0, cfg.verify_budget))
    rng = random.Random(cfg.seed * 1000003 + task_index)
    observations: List[Any] = []
    previous_failed = False
    consecutive_failures = 0
    replan_pending = False
    last_failed_step_id: Optional[int] = None

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
        base_cost = usage_from_response(resp, latency_ms)

        feats = extract_confidence_features(choice)
        tool_calls = getattr(msg, "tool_calls", None) or []

        if not tool_calls:
            traj.final_answer = msg.content or ""
            traj.add(TrajectoryStep(
                task_id=task_id, step_id=0, action_type="answer",
                action_input={"content": traj.final_answer}, observation=None,
                cost_tokens=base_cost.total_tokens, cost_calls=0,
                prompt_tokens=base_cost.prompt_tokens,
                completion_tokens=base_cost.completion_tokens,
                model_calls=1, tool_calls=0, latency_ms=latency_ms,
            ))
            stop_reason = "final_answer"
            break

        tc = tool_calls[0]
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except Exception:
            args = {"_raw": tc.function.arguments}

        original = CandidateAction(name, args)
        final = original
        original_eval = env.evaluate_candidate(original.api_name, original.arguments)
        provenance = check_argument_provenance(
            task.get("requirement", ""), observations, original, tools)
        budget_before = budget.remaining
        extra_cost = UsageCost()

        # 自陈属于额外模型调用，必须占用同一验证预算。
        self_rep: Optional[float] = None
        if cfg.self_report and (cfg.verify_policy == "vanilla" or budget.can_spend(1)):
            if cfg.verify_policy != "vanilla":
                budget.spend(1)
            self_rep, sr_cost = ask_self_report(
                client, cfg, messages,
                "%s(%s)" % (name, json.dumps(args, ensure_ascii=False)))
            extra_cost.add(sr_cost)

        token_conf = feats.get("confidence") if feats.get("available") else None
        signal_vote = None
        if cfg.consistency_k > 0 and cfg.verify_policy in ("vanilla", "fused"):
            if cfg.verify_policy == "vanilla" or budget.can_spend(cfg.consistency_k):
                if cfg.verify_policy != "vanilla":
                    budget.spend(cfg.consistency_k)
                signal_vote = run_consistency_vote(
                    client, cfg.model, messages, tools, original, cfg.consistency_k,
                    cfg.consistency_temp, cfg.max_tokens, cfg.enable_thinking,
                    seed=cfg.seed * 1000003 + task_index * 101 + len(traj.steps) * 17)
                extra_cost.add(signal_vote.usage)
        consistency_signal = None
        if signal_vote is not None and signal_vote.raw:
            consistency_signal = signal_vote.raw.get("agreement")
        fused_risk = fuse_risk(token_conf, self_rep, consistency_signal, provenance.risk)
        if cfg.verify_policy == "threshold":
            # 单信号阈值基线：只使用原始步骤置信度，不借用 provenance。
            risk = 1.0 - token_conf if token_conf is not None else 0.5
        elif cfg.verify_policy == "provenance":
            risk = provenance.risk
        else:
            risk = fused_risk
        do_verify, trigger_reason = should_verify(
            cfg.verify_policy, risk, provenance, budget,
            cfg.risk_threshold, rng)

        verification = None
        verification_usage_precounted = False
        if do_verify:
            if cfg.verifier == "critic" and budget.can_spend(1):
                budget.spend(1)
                verification = run_critic(
                    client, cfg.model, messages, tools, original, provenance,
                    max_tokens=min(768, cfg.max_tokens))
            elif cfg.verifier == "consistency":
                if signal_vote is not None:
                    verification = signal_vote
                    verification_usage_precounted = True
                else:
                    k = min(cfg.consistency_k if cfg.consistency_k > 0 else 3,
                            budget.remaining)
                    if k >= 2:
                        budget.spend(k)
                        verification = run_consistency_vote(
                            client, cfg.model, messages, tools, original, k,
                            cfg.consistency_temp, cfg.max_tokens, cfg.enable_thinking,
                            seed=cfg.seed * 1000003 + task_index * 101 + len(traj.steps) * 17)
            elif cfg.verifier == "hybrid" and budget.can_spend(1):
                budget.spend(1)
                critic_result = run_critic(
                    client, cfg.model, messages, tools, original, provenance,
                    max_tokens=min(768, cfg.max_tokens))
                vote_result = signal_vote
                if vote_result is None:
                    k = min(cfg.consistency_k if cfg.consistency_k > 0 else 3,
                            budget.remaining)
                    if k >= 2:
                        budget.spend(k)
                        vote_result = run_consistency_vote(
                            client, cfg.model, messages, tools, original, k,
                            cfg.consistency_temp, cfg.max_tokens, cfg.enable_thinking,
                            seed=cfg.seed * 1000003 + task_index * 101 + len(traj.steps) * 17)
                hybrid_usage = UsageCost()
                hybrid_usage.add(critic_result.usage)
                if vote_result is not None:
                    hybrid_usage.add(vote_result.usage)
                revised = critic_result.revised_action
                if revised is None and vote_result is not None:
                    revised = vote_result.revised_action
                risks = [critic_result.risk]
                if vote_result is not None:
                    risks.append(vote_result.risk)
                verification = VerificationResult(
                    verifier="hybrid",
                    valid=(critic_result.valid if vote_result is None
                           else bool(critic_result.valid and vote_result.valid)),
                    risk=sum(risks) / len(risks),
                    issues=critic_result.issues + (vote_result.issues if vote_result else []),
                    revised_action=revised, usage=hybrid_usage,
                    raw={"critic": critic_result.to_dict(),
                         "consistency": vote_result.to_dict() if vote_result else None},
                    available=critic_result.available or bool(vote_result and vote_result.available),
                )
                if signal_vote is not None:
                    # signal_vote 已计入 extra_cost，只需补 critic 成本。
                    extra_cost.add(critic_result.usage)
                    verification_usage_precounted = True
            if verification is not None:
                if not verification_usage_precounted:
                    extra_cost.add(verification.usage)
                if verification.revised_action is not None:
                    # 修订动作必须仍属于当前暴露工具集；避免 critic 虚构 API。
                    exposed = {((t.get("function") or {}).get("name")) for t in tools}
                    if verification.revised_action.api_name in exposed:
                        final = verification.revised_action

        changed = final.signature() != original.signature()
        final_eval = env.evaluate_candidate(final.api_name, final.arguments)
        outcome = correction_outcome(
            original_eval.matched_gold, final_eval.matched_gold, changed)
        if previous_failed:
            outcome = ("error_to_correct" if final_eval.matched_gold
                       else "changed_but_still_error")

        # --- 重复调用熔断：按真正即将执行的动作判断 ---
        sig = final.signature()
        if sig == last_sig:
            repeat_count += 1
        else:
            repeat_count = 0
            last_sig = sig

        res = env.call(final.api_name, final.arguments)

        if final.api_name == "ToolSearcher" and isinstance(res.observation, dict) and res.observation.get("name"):
            unlocked[res.observation["name"]] = res.observation

        conf = token_conf
        conf_source = feats.get("confidence_source") if conf is not None else None
        if cfg.verify_policy == "fused":
            conf = 1.0 - fused_risk
            conf_source = "fused"
        raw = {k: v for k, v in feats.items()} if feats.get("available") else None
        if raw is None:
            raw = {}
        if self_rep is not None:
            raw["self_report"] = self_rep
        raw["provenance_risk"] = provenance.risk
        raw["fused_risk"] = fused_risk
        vote_for_raw = verification if (verification is not None and verification.verifier == "consistency") else signal_vote
        if vote_for_raw is not None and vote_for_raw.raw:
            raw["consistency"] = vote_for_raw.raw.get("agreement")
            raw["consistency_n"] = vote_for_raw.raw.get("n_samples")
            raw["consistency_agree"] = vote_for_raw.raw.get("n_agree")
        elif verification is not None and verification.verifier == "hybrid" and verification.raw:
            vote = verification.raw.get("consistency") or {}
            vote_raw = vote.get("raw") or {}
            if vote_raw.get("agreement") is not None:
                raw["consistency"] = vote_raw.get("agreement")
                raw["consistency_n"] = vote_raw.get("n_samples")
                raw["consistency_agree"] = vote_raw.get("n_agree")
        if not raw:
            raw = None

        if replan_pending:
            action_type = "replan"
        elif previous_failed:
            action_type = "retry"
        else:
            action_type = "tool_search" if final.api_name == "ToolSearcher" else "tool_call"
        correction_type = None
        correction_text = None
        if changed:
            if final.api_name != original.api_name:
                correction_type = "tool_revision"
            elif any(i.get("kind") == "unsupported_value" for i in provenance.issues):
                correction_type = "information_supplement"
            else:
                correction_type = "argument_revision"
            correction_text = "%s revised candidate before execution" % (
                verification.verifier if verification else "verifier")
        elif verification is not None:
            correction_type = "none"
        if replan_pending:
            correction_type = "replan"
            correction_text = "replanned after repeated execution failures"
        elif previous_failed and correction_type in (None, "none"):
            correction_type = "local_retry"
            correction_text = "local retry after environment feedback"

        step = traj.add(TrajectoryStep(
            task_id=task_id, step_id=0, action_type=action_type,
            action_input=final.to_dict(),
            observation=res.observation if res.observation is not None else {"error": res.exception},
            confidence=conf,
            confidence_source=conf_source,
            verified=verification is not None,
            correction=correction_text,
            candidate_action=original.to_dict(),
            verification_type=(cfg.verifier if verification is not None else None),
            trigger_reason=trigger_reason, risk_score=risk,
            verification_result=(verification.to_dict() if verification else
                                 {"provenance": provenance.to_dict()}),
            correction_type=correction_type, correction_changed=changed,
            correction_outcome=outcome,
            candidate_matched_gold=original_eval.matched_gold,
            parent_step_id=last_failed_step_id if previous_failed else None,
            budget_before=budget_before, budget_after=budget.remaining,
            cost_tokens=base_cost.total_tokens + extra_cost.total_tokens,
            cost_calls=1,
            prompt_tokens=base_cost.prompt_tokens + extra_cost.prompt_tokens,
            completion_tokens=base_cost.completion_tokens + extra_cost.completion_tokens,
            model_calls=base_cost.model_calls + extra_cost.model_calls,
            tool_calls=1,
            verification_prompt_tokens=extra_cost.prompt_tokens,
            verification_completion_tokens=extra_cost.completion_tokens,
            verification_model_calls=extra_cost.model_calls,
            verification_latency_ms=extra_cost.latency_ms,
            matched_gold=res.matched_gold,
            strict_matched_gold=res.strict_matched_gold,
            failure_type=res.failure_type,
            latency_ms=latency_ms + extra_cost.latency_ms,
            raw_logprobs=raw,
        ))

        # --- 回灌历史：**剥离 <think> 段**，强制每步重新思考 ---
        clean_content = _THINK_RE.sub("", msg.content or "").strip()
        final_args_json = json.dumps(final.arguments, ensure_ascii=False)
        messages.append({
            "role": "assistant", "content": clean_content,
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": final.api_name, "arguments": final_args_json}}],
        })
        messages.append({
            "role": "tool", "tool_call_id": tc.id,
            "content": json.dumps(
                res.observation if res.observation is not None else {"error": res.exception},
                ensure_ascii=False, default=str)[:4000],
        })
        observations.append(res.observation if res.observation is not None else {"error": res.exception})

        if cfg.verbose:
            print("   step%-2d %-14s matched=%-5s conf=%-6s n_tok=%s%s" % (
                len(traj.steps), final.api_name, res.matched_gold,
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

        previous_failed = res.matched_gold is False
        if previous_failed:
            last_failed_step_id = step.step_id
        replan_pending = False
        if previous_failed:
            consecutive_failures += 1
            if cfg.verify_policy != "vanilla" and consecutive_failures >= cfg.max_retries:
                stop_reason = "max_retries"
                break
            if cfg.verify_policy != "vanilla" and consecutive_failures >= cfg.replan_after:
                replan_pending = True
                messages.append({"role": "user", "content": (
                    "Multiple local attempts failed. Re-plan the remaining task from the original "
                    "requirement and successful observations. Choose a materially different next action.")})
            elif cfg.verify_policy != "vanilla":
                messages.append({"role": "user", "content": (
                    "The last API call failed. Correct only the unsupported or invalid part, use values "
                    "grounded in the requirement or prior tool outputs, and retry locally.")})
        else:
            consecutive_failures = 0

    traj.final_success = env.success()
    traj.clean_success = env.clean_success()
    traj.recovered_success = env.recovered_success()
    traj.config["stop_reason"] = stop_reason
    traj.config["verification_budget_used"] = budget.used
    traj.config["verification_budget_remaining"] = budget.remaining
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
    ap.add_argument("--verify-policy", default="vanilla",
                    choices=["vanilla", "always", "random", "exception",
                             "provenance", "threshold", "fused"])
    ap.add_argument("--verifier", default="critic",
                    choices=["critic", "consistency", "hybrid"])
    ap.add_argument("--verify-budget", type=int, default=0,
                    help="每任务允许的额外模型调用次数")
    ap.add_argument("--risk-threshold", type=float, default=0.60)
    ap.add_argument("--replan-after", type=int, default=2)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--feedback-mode", default="realistic",
                    choices=["realistic", "diagnostic"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = AgentConfig(
        base_url=args.base_url, model=args.model, max_steps=args.max_steps,
        temperature=args.temperature, enable_thinking=not args.no_thinking,
        self_report=args.self_report, consistency_k=args.consistency_k,
        consistency_temp=args.consistency_temp,
        verify_policy=args.verify_policy, verifier=args.verifier,
        verify_budget=args.verify_budget, risk_threshold=args.risk_threshold,
        replan_after=args.replan_after, max_retries=args.max_retries,
        feedback_mode=args.feedback_mode,
        seed=args.seed, verbose=args.verbose,
        method=("vanilla_react" if args.verify_policy == "vanilla"
                else "conf_verify_%s_%s" % (args.verify_policy, args.verifier)),
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
