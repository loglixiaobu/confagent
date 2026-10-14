#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一轨迹格式（全组接口约定，请勿随意改动字段名）。

为什么要有这个模块
--------------------------------------------------------------------
5 个人分头写代码，最容易翻车的地方就是"各自记录各自的轨迹"。
一旦格式不一致，后面的置信度模块、验证策略模块、评测模块全部要返工。

因此本文件是**唯一的轨迹定义来源**：所有模块只通过它读写 trajectory.jsonl。

字段约定（与项目方案文档一致）
--------------------------------------------------------------------
    task_id            任务标识
    step_id            步骤序号（从 1 开始）
    action_type        动作类型：tool_call / tool_search / retry / replan / answer
    action_input       动作输入（工具名 + 参数）
    observation        环境返回的观测
    confidence         步骤级置信度 ∈ [0,1]
    confidence_source  置信度来源：token_logprob / self_report / consistency / fused
    verified           是否对该步骤执行了验证动作
    correction         纠错动作描述（无则 None）
    cost_tokens        该步骤消耗的 token
    cost_calls         该步骤消耗的工具调用次数
    final_success      整条轨迹是否成功（写回时填充）
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# 动作类型的受控词表——新增类型必须在这里登记，避免各写各的
ACTION_TYPES = ("tool_call", "tool_search", "retry", "replan", "answer")

# 置信度来源的受控词表
CONFIDENCE_SOURCES = ("token_logprob", "self_report", "consistency", "fused")


@dataclass
class TrajectoryStep:
    """单步轨迹记录。"""

    task_id: str
    step_id: int
    action_type: str
    action_input: Dict[str, Any]
    observation: Any = None

    confidence: Optional[float] = None
    confidence_source: Optional[str] = None

    verified: bool = False
    correction: Optional[str] = None

    cost_tokens: int = 0
    cost_calls: int = 0

    # --- 以下为分析用辅助字段（可选，便于事后归因） ---
    matched_gold: Optional[bool] = None          # 宽松口径标签（主标签）
    strict_matched_gold: Optional[bool] = None   # 严格口径标签（敏感性分析用）
    failure_type: Optional[str] = None           # 失败类型分类
    latency_ms: Optional[int] = None             # 该步耗时
    raw_logprobs: Optional[Dict[str, Any]] = None  # token logprob 原始统计

    def __post_init__(self) -> None:
        if self.action_type not in ACTION_TYPES:
            raise ValueError(
                "action_type 必须是 %s 之一，收到 %r" % (ACTION_TYPES, self.action_type)
            )
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence 必须落在 [0,1]，收到 %r" % self.confidence)
        if self.confidence_source is not None and self.confidence_source not in CONFIDENCE_SOURCES:
            raise ValueError(
                "confidence_source 必须是 %s 之一，收到 %r"
                % (CONFIDENCE_SOURCES, self.confidence_source)
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Trajectory:
    """一条完整轨迹。"""

    task_id: str
    steps: List[TrajectoryStep] = field(default_factory=list)

    # 运行配置，便于复现
    model: Optional[str] = None
    seed: Optional[int] = None
    config: Dict[str, Any] = field(default_factory=dict)

    # 终局结果
    final_success: Optional[bool] = None
    final_answer: Optional[str] = None

    # 递归改进相关
    round_idx: int = 0          # 第几轮（递归改进实验用）
    method: Optional[str] = None  # 方法名：vanilla_react / conf_verify / ...

    def add(self, step: TrajectoryStep) -> TrajectoryStep:
        step.step_id = len(self.steps) + 1
        self.steps.append(step)
        return step

    # -- 成本汇总 -----------------------------------------------------------
    @property
    def total_tokens(self) -> int:
        return sum(s.cost_tokens for s in self.steps)

    @property
    def total_calls(self) -> int:
        return sum(s.cost_calls for s in self.steps)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    # -- 分析用 -------------------------------------------------------------
    def error_propagation_rate(self) -> Optional[float]:
        """错误传播率：首个错误步骤之后，仍被继续执行的步骤占比。

        这是研究内容二的核心指标之一——衡量"一个错误会污染多远"。
        """
        first_err = next((i for i, s in enumerate(self.steps) if s.matched_gold is False), None)
        if first_err is None:
            return 0.0
        downstream = len(self.steps) - first_err - 1
        return downstream / max(len(self.steps) - 1, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model": self.model,
            "seed": self.seed,
            "round_idx": self.round_idx,
            "method": self.method,
            "config": self.config,
            "final_success": self.final_success,
            "final_answer": self.final_answer,
            "n_steps": self.n_steps,
            "total_tokens": self.total_tokens,
            "total_calls": self.total_calls,
            "steps": [s.to_dict() for s in self.steps],
        }


class TrajectoryWriter:
    """把轨迹以 JSONL 追加写入，保证多人并发写不会互相覆盖。

    每行 = 一条完整轨迹（含 steps 数组）。这样单文件即可支撑后续所有分析，
    也便于用 pandas 直接 read_json(lines=True) 展开。
    """

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def write(self, traj: Trajectory) -> None:
        rec = traj.to_dict()
        rec["_written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def write_many(self, trajs: List[Trajectory]) -> None:
        for t in trajs:
            self.write(t)


def demo() -> None:
    """自检：构造一条带错误的轨迹，验证字段与分析函数。"""
    t = Trajectory(task_id="demo-0", model="Qwen3-8B", seed=0, method="vanilla_react", round_idx=0)
    t.add(TrajectoryStep(
        task_id=t.task_id, step_id=0, action_type="tool_search",
        action_input={"keywords": "QueryMeeting"},
        observation={"name": "QueryMeeting"},
        confidence=0.91, confidence_source="token_logprob",
        cost_tokens=120, cost_calls=1, matched_gold=True,
    ))
    t.add(TrajectoryStep(
        task_id=t.task_id, step_id=0, action_type="tool_call",
        action_input={"api_name": "QueryMeeting", "user_name": "John"},
        observation={"meetings": []},
        confidence=0.55, confidence_source="token_logprob",
        cost_tokens=180, cost_calls=1, matched_gold=True,
    ))
    t.add(TrajectoryStep(
        task_id=t.task_id, step_id=0, action_type="tool_call",
        action_input={"api_name": "EmailReminder", "recipient": "wrong@x.com"},
        observation=None,
        confidence=0.42, confidence_source="token_logprob",
        cost_tokens=210, cost_calls=1, matched_gold=False, failure_type="wrong_value",
    ))
    t.final_success = False

    print(json.dumps(t.to_dict(), ensure_ascii=False, indent=2)[:900])
    print()
    print("n_steps            :", t.n_steps)
    print("total_tokens       :", t.total_tokens)
    print("total_calls        :", t.total_calls)
    print("error_propagation  :", t.error_propagation_rate())


if __name__ == "__main__":
    demo()
