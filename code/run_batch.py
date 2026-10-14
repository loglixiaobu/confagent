#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多卡并行批处理运行器。

为什么要多卡
--------------------------------------------------------------------
单卡跑 level-3 全量约需 30+ 分钟；8 张 3090 若只用 1 张，7/8 的算力被浪费。
本脚本把任务分片到多个 vLLM 端点（每个端点占一张卡），并发执行。

用法
--------------------------------------------------------------------
    # 1) 先在多张卡上起多个 vLLM 副本
    #    GPU=0 PORT=8000 bash run_vllm.sh
    #    GPU=1 PORT=8001 bash run_vllm.sh   ...
    # 2) 并行跑
    python run_batch.py --endpoints 8000,8001,8002,8003 --limit 50 --concurrency 4

难度控制
--------------------------------------------------------------------
    --distractors N   给每个任务额外注入 N 个无关 API（增加选错工具的概率）
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apibank_env import (APIBankMockEnv, collect_distractor_apis,  # noqa: E402
                         load_level3)
from react_agent import AgentConfig, run_task  # noqa: E402
from trajectory import Trajectory, TrajectoryWriter  # noqa: E402


# ---------------------------------------------------------------------------
# 温度保护
# ---------------------------------------------------------------------------
def max_gpu_temp(visible_only: bool = True) -> Optional[float]:
    """返回当前所有（或可见）GPU 的最高核心温度（摄氏度）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return None
    temps = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[1].isdigit():
            temps.append(int(parts[1]))
    return max(temps) if temps else None


class ThermalGuard:
    """在提交任务前检查 GPU 温度；过热则暂停，避免持续满载烧卡。

    背景：3090 的 GDDR6X 显存结温在持续满载下可达 100°C 以上，
    而该传感器不通过 nvidia-smi 暴露。GPU 从总线上掉下来（掉卡）通常
    就是长期高温导致。因此宁可牺牲吞吐，也要把温度压在阈值以下。
    """

    def __init__(self, max_temp: float = 75.0, cooldown: float = 0.0,
                 poll: float = 20.0) -> None:
        self.max_temp = max_temp
        self.cooldown = cooldown
        self.poll = poll
        self._lock = threading.Lock()
        self.pause_events = 0

    def wait_if_hot(self) -> None:
        with self._lock:
            t = max_gpu_temp()
            while t is not None and t >= self.max_temp:
                self.pause_events += 1
                print("  [thermal] 最高温 %.0f°C >= %.0f°C，暂停 %.0fs 降温（第 %d 次）"
                      % (t, self.max_temp, self.poll, self.pause_events), flush=True)
                time.sleep(self.poll)
                t = max_gpu_temp()
        if self.cooldown > 0:
            time.sleep(self.cooldown)


def task_apis(task: Dict[str, Any]) -> set:
    """任务 gold 链涉及的全部 API 名称（含 ToolSearcher 检索出的）。"""
    out = set()
    for st in task.get("apis", []):
        name = st.get("api_name")
        if name and name != "ToolSearcher":
            out.add(name)
        o = (st.get("output") or {}).get("output")
        if isinstance(o, dict) and o.get("name"):
            out.add(o["name"])
    return out


def build_task_payload(task: Dict[str, Any], distractor_pool: List[Dict[str, Any]],
                       n_distractors: int, rng: random.Random) -> Dict[str, Any]:
    """把干扰 API 挂到任务上（run_task 内部会用 APIBankMockEnv 的构造参数）。"""
    t = dict(task)
    if n_distractors > 0 and distractor_pool:
        own = {st["output"]["output"]["name"]
               for st in task.get("apis", [])
               if st.get("api_name") == "ToolSearcher"
               and isinstance((st.get("output") or {}).get("output"), dict)
               and (st.get("output") or {}).get("output", {}).get("name")}
        pool = [a for a in distractor_pool if a.get("name") not in own]
        k = min(n_distractors, len(pool))
        t["_distractors"] = rng.sample(pool, k) if k else []
    else:
        t["_distractors"] = []
    return t


def run_one(task: Dict[str, Any], cfg: AgentConfig, endpoint: str, task_index: int,
            distractors: List[Dict[str, Any]]) -> Trajectory:
    """在指定端点上跑一个任务（带干扰 API）。"""
    client = OpenAI(base_url="http://%s/v1" % endpoint, api_key="EMPTY")
    local_cfg = AgentConfig(**{**cfg.__dict__, "base_url": "http://%s/v1" % endpoint})
    return run_task(task, local_cfg, client, task_index=task_index, distractors=distractors)


def main() -> None:
    ap = argparse.ArgumentParser(description="多卡并行批处理")
    ap.add_argument("--endpoints", default="localhost:8000",
                    help="逗号分隔的 host:port 列表")
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--data-dir", default="/home1/lh/confagent/data/api-bank")
    ap.add_argument("--out", default="/home1/lh/confagent/outputs/traj_vanilla.jsonl")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="每个端点上的并发任务数")
    ap.add_argument("--distractors", type=int, default=0,
                    help="每任务注入的干扰 API 数量（难度控制）")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--self-report", action="store_true")
    ap.add_argument("--tag", default="", help="写入文件名后缀，便于区分不同配置")
    ap.add_argument("--max-temp", type=float, default=75.0,
                    help="GPU 最高温度阈值（摄氏度），超过则暂停降温")
    ap.add_argument("--cooldown", type=float, default=0.0,
                    help="每个任务开始前的固定冷却秒数（占空比降温）")
    ap.add_argument("--exclude-apis", default="",
                    help="逗号分隔的 API 名；gold 链涉及这些 API 的任务将被剔除")
    ap.add_argument("--consistency-k", type=int, default=0,
                    help="多采样一致性的采样次数（0=关闭）")
    ap.add_argument("--consistency-temp", type=float, default=0.7)
    args = ap.parse_args()

    endpoints = [e.strip() for e in args.endpoints.split(",") if e.strip()]
    all_tasks = load_level3(args.data_dir, "test")

    # 剔除 gold 链涉及指定 API 的任务（例如上一批正确率为 0% 的子任务）
    excluded = {a.strip() for a in args.exclude_apis.split(",") if a.strip()}
    if excluded:
        before = len(all_tasks)
        all_tasks = [t for t in all_tasks if not (task_apis(t) & excluded)]
        print("剔除 API %s：任务 %d -> %d" % (sorted(excluded), before, len(all_tasks)))

    tasks = all_tasks[args.offset: args.offset + args.limit]
    pool = collect_distractor_apis(all_tasks)
    rng = random.Random(args.seed)

    out = args.out
    if args.tag:
        root, ext = os.path.splitext(out)
        out = "%s_%s%s" % (root, args.tag, ext)

    cfg = AgentConfig(
        model=args.model, max_steps=args.max_steps, temperature=args.temperature,
        enable_thinking=not args.no_thinking, self_report=args.self_report,
        consistency_k=args.consistency_k, consistency_temp=args.consistency_temp,
        seed=args.seed, method="vanilla_react",
    )

    print("=" * 72)
    print("多卡批处理 | 模型=%s | 任务=%d | 端点=%d | 并发/端点=%d | 干扰API=%d"
          % (args.model, len(tasks), len(endpoints), args.concurrency, args.distractors))
    print("信号采集 | self_report=%s | consistency_k=%d (temp=%.2f)"
          % (args.self_report, args.consistency_k, args.consistency_temp))
    print("端点:", ", ".join(endpoints))
    print("输出:", out)
    print("=" * 72)

    writer = TrajectoryWriter(out)
    lock = threading.Lock()
    results: List[Trajectory] = []
    guard = ThermalGuard(max_temp=args.max_temp, cooldown=args.cooldown)

    t_start_temp = max_gpu_temp()
    print("起始最高温: %s°C | 温度阈值: %.0f°C | 任务间冷却: %.0fs"
          % (t_start_temp if t_start_temp is not None else "?", args.max_temp, args.cooldown))

    def job(idx: int, task: Dict[str, Any]) -> Trajectory:
        guard.wait_if_hot()
        ep = endpoints[idx % len(endpoints)]
        payload = build_task_payload(task, pool, args.distractors, rng)
        traj = run_one(payload, cfg, ep, args.offset + idx, payload["_distractors"])
        with lock:
            writer.write(traj)
            results.append(traj)
            n = len(results)
            if n % 5 == 0 or n == len(tasks):
                ok = sum(1 for t in results if t.final_success)
                print("  进度 %d/%d  成功 %d (%.0f%%)  最高温 %s°C"
                      % (n, len(tasks), ok, 100.0 * ok / n,
                         max_gpu_temp() if max_gpu_temp() is not None else "?"))
        return traj

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(endpoints) * max(args.concurrency, 1)) as ex:
        futs = [ex.submit(job, i, t) for i, t in enumerate(tasks)]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print("  [task error]", type(e).__name__, e)

    dt = time.time() - t0
    ok = sum(1 for t in results if t.final_success)
    steps = [s for t in results for s in t.steps]
    n_ok_steps = sum(1 for s in steps if s.matched_gold)
    n_deg = sum(1 for s in steps if (s.raw_logprobs or {}).get("degenerate"))

    print()
    print("=" * 72)
    print("完成 %d 任务 | 任务成功率 %.1f%% | 用时 %.1f 分钟 (%.1f 秒/任务)"
          % (len(results), 100.0 * ok / max(len(results), 1), dt / 60, dt / max(len(results), 1)))
    print("步骤总数 %d | 步骤正确率 %.1f%% | 负样本 %d | 退化置信度 %d"
          % (len(steps), 100.0 * n_ok_steps / max(len(steps), 1),
             len(steps) - n_ok_steps, n_deg))
    print("温度保护触发 %d 次 | 结束时最高温 %s°C"
          % (guard.pause_events, max_gpu_temp() if max_gpu_temp() is not None else "?"))
    print("输出:", out)


if __name__ == "__main__":
    main()
