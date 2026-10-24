#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""API-Bank level-3 多端点批处理与 Q2 可复现实验入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apibank_env import collect_distractor_apis, load_level3  # noqa: E402
from confidence_runtime import (K as FROZEN_CONFIDENCE_K,  # noqa: E402
                                MODEL as FROZEN_CONFIDENCE_MODEL,
                                load_calibrator)
from react_agent import AgentConfig, run_task  # noqa: E402
from trajectory import Trajectory, TrajectoryWriter  # noqa: E402


MANIFEST_VERSION = "q2-task-manifest-v1"
PROTOCOL_VERSION = "q2-server-experiment-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=str)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def task_fingerprint(task: Dict[str, Any]) -> str:
    return fingerprint({k: v for k, v in task.items() if not str(k).startswith("_")})


def dataset_fingerprint(tasks: List[Dict[str, Any]]) -> str:
    return fingerprint([task_fingerprint(task) for task in tasks])


def task_apis(task: Dict[str, Any]) -> set:
    """Return API names in a gold chain; used only for pre-run cohort filtering."""
    names = set()
    for step in task.get("apis", []):
        name = step.get("api_name")
        if name and name != "ToolSearcher":
            names.add(name)
        output = (step.get("output") or {}).get("output")
        if isinstance(output, dict) and output.get("name"):
            names.add(output["name"])
    return names


def create_task_manifest(tasks: List[Dict[str, Any]], selected_indices: List[int],
                         split: str, role: str) -> Dict[str, Any]:
    rows = []
    for index in selected_indices:
        rows.append({
            "task_id": "%s-%05d" % (split, index),
            "dataset_index": index,
            "task_fingerprint": task_fingerprint(tasks[index]),
        })
    body = {
        "version": MANIFEST_VERSION,
        "role": role,
        "dataset_split": split,
        "dataset_fingerprint": dataset_fingerprint(tasks),
        "tasks": rows,
    }
    body["manifest_id"] = fingerprint(body)
    return body


def load_task_manifest(path: str, tasks: List[Dict[str, Any]],
                       split: str) -> Tuple[Dict[str, Any], List[int]]:
    with open(path, encoding="utf-8") as source:
        manifest = json.load(source)
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError("不支持的任务清单版本: %r" % manifest.get("version"))
    if manifest.get("dataset_split") != split:
        raise ValueError("任务清单 split=%r 与 --split=%r 不一致"
                         % (manifest.get("dataset_split"), split))
    current_dataset = dataset_fingerprint(tasks)
    if manifest.get("dataset_fingerprint") != current_dataset:
        raise ValueError("数据集指纹不一致；禁止在变化后的数据上复用旧任务清单")
    unsigned = {k: v for k, v in manifest.items() if k != "manifest_id"}
    if manifest.get("manifest_id") != fingerprint(unsigned):
        raise ValueError("任务清单 manifest_id 校验失败")
    indices = []
    seen = set()
    for row in manifest.get("tasks") or []:
        index = row.get("dataset_index")
        if type(index) is not int or index < 0 or index >= len(tasks):
            raise ValueError("任务清单包含非法 dataset_index: %r" % index)
        if index in seen:
            raise ValueError("任务清单包含重复 dataset_index: %d" % index)
        if row.get("task_fingerprint") != task_fingerprint(tasks[index]):
            raise ValueError("任务 %d 指纹不一致" % index)
        expected_id = "%s-%05d" % (split, index)
        if row.get("task_id") != expected_id:
            raise ValueError("任务 %d 的 task_id 不稳定" % index)
        seen.add(index)
        indices.append(index)
    if not indices:
        raise ValueError("任务清单为空")
    return manifest, indices


def _write_json_new(path: str, value: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def _git_state() -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    base = ["git", "-c", "safe.directory=%s" % root.as_posix()]
    try:
        commit = subprocess.run(base + ["rev-parse", "HEAD"], cwd=root,
                                capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        status = subprocess.run(base + ["status", "--porcelain"], cwd=root,
                                capture_output=True, text=True, timeout=10, check=True).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except Exception as exc:
        return {"commit": None, "dirty": None,
                "error": "%s: %s" % (type(exc).__name__, exc)}


def max_gpu_temp() -> Optional[float]:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    values = [int(line.strip()) for line in output.splitlines() if line.strip().isdigit()]
    return max(values) if values else None


class ThermalGuard:
    def __init__(self, max_temp: float, cooldown: float, poll: float = 20.0) -> None:
        self.max_temp = max_temp
        self.cooldown = cooldown
        self.poll = poll
        self.pause_events = 0
        self._lock = threading.Lock()

    def wait_if_hot(self) -> None:
        with self._lock:
            temperature = max_gpu_temp()
            while temperature is not None and temperature >= self.max_temp:
                self.pause_events += 1
                print("  [thermal] %.0f°C >= %.0f°C，等待 %.0f 秒"
                      % (temperature, self.max_temp, self.poll), flush=True)
                time.sleep(self.poll)
                temperature = max_gpu_temp()
        if self.cooldown > 0:
            time.sleep(self.cooldown)


def _normalise_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip()
    if endpoint.isdigit():
        endpoint = "localhost:" + endpoint
    return endpoint.rstrip("/")


def check_endpoint_models(endpoints: List[str], expected_model: str) -> None:
    """Fail before the experiment if an endpoint is absent or serves another name."""
    for endpoint in endpoints:
        url = "http://%s/v1/models" % endpoint
        try:
            with urllib.request.urlopen(url, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))
            model_ids = {row.get("id") for row in payload.get("data", [])
                         if isinstance(row, dict) and isinstance(row.get("id"), str)}
        except Exception as exc:
            raise RuntimeError("端点健康检查失败 %s: %s: %s"
                               % (endpoint, type(exc).__name__, exc)) from exc
        if expected_model not in model_ids:
            raise RuntimeError("端点 %s 未提供模型 %s；可用=%s"
                               % (endpoint, expected_model, sorted(model_ids)))


def build_task_payload(task: Dict[str, Any], dataset_index: int, split: str,
                       manifest_id: str, distractor_pool: List[Dict[str, Any]],
                       n_distractors: int, rng: random.Random) -> Dict[str, Any]:
    payload = dict(task)
    payload["_task_id"] = "%s-%05d" % (split, dataset_index)
    payload["_dataset_split"] = split
    payload["_dataset_index"] = dataset_index
    payload["_task_fingerprint"] = task_fingerprint(task)
    payload["_manifest_id"] = manifest_id
    own = {((step.get("output") or {}).get("output") or {}).get("name")
           for step in task.get("apis", [])
           if step.get("api_name") == "ToolSearcher"
           and isinstance((step.get("output") or {}).get("output"), dict)}
    pool = [api for api in distractor_pool if api.get("name") not in own]
    count = min(n_distractors, len(pool))
    payload["_distractors"] = rng.sample(pool, count) if count else []
    return payload


def run_one(task: Dict[str, Any], cfg: AgentConfig, endpoint: str,
            dataset_index: int) -> Trajectory:
    base_url = "http://%s/v1" % endpoint
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    local_cfg = AgentConfig(**{**cfg.__dict__, "base_url": base_url})
    return run_task(task, local_cfg, client, task_index=dataset_index,
                    distractors=task.get("_distractors") or [])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q2 多端点、可复现批处理")
    parser.add_argument("--endpoints", default="localhost:8000")
    parser.add_argument("--model", default="Qwen3-8B")
    parser.add_argument("--data-dir", default=str(
        Path(__file__).resolve().parents[1] / "data/api-bank"))
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--experiment-role", choices=["development", "formal-test"],
                        default="development")
    parser.add_argument("--task-manifest", default="",
                        help="读取冻结任务清单；正式测试必须提供")
    parser.add_argument("--write-task-manifest", default="",
                        help="把本次 offset/limit/filter 选择冻结成新清单")
    parser.add_argument("--out", default="")
    parser.add_argument("--tag", default="",
                        help="兼容旧入口：在输出文件扩展名前追加标签")
    parser.add_argument("--manifest-only", action="store_true",
                        help="只生成并校验任务清单，不连接模型")
    parser.add_argument("--allow-append", action="store_true",
                        help="仅用于恢复性调试；正式实验禁止追加")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--distractors", type=int, default=0)
    parser.add_argument("--exclude-apis", default="")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--self-report", action="store_true")
    parser.add_argument("--consistency-k", type=int, default=0)
    parser.add_argument("--consistency-temp", type=float, default=0.7)
    parser.add_argument("--verify-policy", default="direct",
                        choices=["direct", "random_check", "confidence_check",
                                 "vanilla", "always", "random", "exception",
                                 "provenance", "threshold", "fused"])
    parser.add_argument("--verifier", default="critic",
                        choices=["critic", "consistency", "hybrid"])
    parser.add_argument("--verify-budget", type=int, default=0)
    parser.add_argument("--risk-threshold", type=float, default=0.60)
    parser.add_argument("--calibrator", default=AgentConfig.calibrator_path)
    parser.add_argument("--confidence-cooldown", type=float, default=1.0)
    parser.add_argument("--random-check-rate", type=float, default=0.50)
    parser.add_argument("--max-checks-per-task", type=int, default=1)
    parser.add_argument("--no-revision-guard", action="store_true")
    parser.add_argument("--replan-after", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--feedback-mode", choices=["realistic", "diagnostic"],
                        default="realistic")
    parser.add_argument("--max-temp", type=float, default=75.0)
    parser.add_argument("--cooldown", type=float, default=0.0)
    parser.add_argument("--skip-endpoint-health", action="store_true",
                        help="仅调试使用；跳过 /v1/models 预检")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.limit <= 0 or args.offset < 0 or args.concurrency <= 0:
        raise SystemExit("--limit/--concurrency 必须为正，--offset 不能为负")
    if args.verify_budget < 0 or args.max_checks_per_task < 0:
        raise SystemExit("预算和检查次数不能为负")
    if not 0 <= args.random_check_rate <= 1:
        raise SystemExit("--random-check-rate 必须位于 [0,1]")
    if args.verify_policy in {"direct", "vanilla"} and args.verify_budget != 0:
        raise SystemExit("direct/vanilla 的 --verify-budget 必须为 0")
    if args.verify_policy in {"random_check", "confidence_check"}:
        if args.verifier != "critic":
            raise SystemExit("Q2 核心 random_check/confidence_check 固定使用 critic")
        if args.verify_budget < FROZEN_CONFIDENCE_K + 1:
            raise SystemExit("核心检查策略至少需要 --verify-budget 4")
        if args.model != FROZEN_CONFIDENCE_MODEL:
            raise SystemExit(
                "核心检查策略必须使用冻结评分模型名 %s；收到 %s"
                % (FROZEN_CONFIDENCE_MODEL, args.model))
        load_calibrator(args.calibrator)  # fail fast before any model request
    if args.experiment_role == "formal-test":
        if args.split != "test":
            raise SystemExit("formal-test 必须使用 --split test")
        if not args.manifest_only and not args.task_manifest:
            raise SystemExit("formal-test 运行必须使用冻结的 --task-manifest")
        if args.allow_append:
            raise SystemExit("formal-test 禁止 --allow-append")
    if args.manifest_only and (not args.write_task_manifest or args.task_manifest):
        raise SystemExit("--manifest-only 必须搭配新的 --write-task-manifest，不能读取旧清单")
    if not args.manifest_only and not args.out:
        raise SystemExit("执行任务时必须提供 --out")

    endpoints = [_normalise_endpoint(value) for value in args.endpoints.split(",") if value.strip()]
    if not endpoints:
        raise SystemExit("--endpoints 不能为空")
    all_tasks = load_level3(args.data_dir, args.split)
    excluded = {name.strip() for name in args.exclude_apis.split(",") if name.strip()}

    if args.task_manifest:
        if args.write_task_manifest or excluded or args.offset != 0:
            raise SystemExit("读取任务清单时不能再写清单、过滤 API 或设置 offset")
        manifest, selected_indices = load_task_manifest(
            args.task_manifest, all_tasks, args.split)
        if (args.experiment_role == "formal-test"
                and manifest.get("role") != "formal-test"):
            raise SystemExit("formal-test 必须读取 role=formal-test 的任务清单")
    else:
        eligible = [index for index, task in enumerate(all_tasks)
                    if not (task_apis(task) & excluded)]
        selected_indices = eligible[args.offset:args.offset + args.limit]
        if not selected_indices:
            raise SystemExit("没有选中任务")
        manifest = create_task_manifest(
            all_tasks, selected_indices, args.split, args.experiment_role)
        if args.write_task_manifest:
            _write_json_new(args.write_task_manifest, manifest)

    if args.manifest_only:
        print("任务清单已生成：%s（%d 条，manifest=%s）" % (
            args.write_task_manifest, len(selected_indices), manifest["manifest_id"]))
        return

    output_text = args.out
    if args.tag:
        root, extension = os.path.splitext(output_text)
        output_text = "%s_%s%s" % (root, args.tag, extension)
    output = Path(output_text)
    if output.exists() and output.stat().st_size > 0 and not args.allow_append:
        raise SystemExit("输出文件已存在且非空：%s" % output)
    protocol_path = str(output) + ".protocol.json"
    summary_path = str(output) + ".summary.json"
    if not args.allow_append and (Path(protocol_path).exists() or Path(summary_path).exists()):
        raise SystemExit("协议或摘要侧车文件已存在，请更换 --out")

    method = ("direct" if args.verify_policy in {"direct", "vanilla"}
              else args.verify_policy)
    cfg = AgentConfig(
        model=args.model, max_steps=args.max_steps, temperature=args.temperature,
        enable_thinking=not args.no_thinking, self_report=args.self_report,
        consistency_k=args.consistency_k, consistency_temp=args.consistency_temp,
        verify_policy=args.verify_policy, verifier=args.verifier,
        verify_budget=args.verify_budget, risk_threshold=args.risk_threshold,
        calibrator_path=args.calibrator, confidence_cooldown=args.confidence_cooldown,
        random_check_rate=args.random_check_rate,
        max_checks_per_task=args.max_checks_per_task,
        revision_guard=not args.no_revision_guard,
        replan_after=args.replan_after, max_retries=args.max_retries,
        feedback_mode=args.feedback_mode, seed=args.seed, method=method)

    if not args.skip_endpoint_health:
        check_endpoint_models(endpoints, args.model)

    protocol = {
        "version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "endpoints": endpoints,
        "dataset_split": args.split,
        "experiment_role": args.experiment_role,
        "manifest_id": manifest["manifest_id"],
        "task_ids": [row["task_id"] for row in manifest["tasks"]],
        "config": {**cfg.__dict__, "base_url": "<per-endpoint>"},
        "distractors": args.distractors,
        "excluded_apis": sorted(excluded),
        "endpoint_health_checked": not args.skip_endpoint_health,
        "git": _git_state(),
    }
    protocol["protocol_id"] = fingerprint(protocol)
    if not args.allow_append:
        _write_json_new(protocol_path, protocol)

    pool = collect_distractor_apis(all_tasks)
    writer = TrajectoryWriter(str(output))
    lock = threading.Lock()
    guard = ThermalGuard(args.max_temp, args.cooldown)
    results: List[Trajectory] = []
    failures: List[Dict[str, Any]] = []
    started = time.time()

    print("Q2 batch | model=%s | tasks=%d | policy=%s | B=%d | manifest=%s"
          % (args.model, len(selected_indices), args.verify_policy,
             args.verify_budget, manifest["manifest_id"][:12]))

    def job(position: int, dataset_index: int) -> Trajectory:
        guard.wait_if_hot()
        task_rng = random.Random(args.seed * 1000003 + dataset_index)
        payload = build_task_payload(
            all_tasks[dataset_index], dataset_index, args.split,
            manifest["manifest_id"], pool, args.distractors, task_rng)
        return run_one(payload, cfg, endpoints[position % len(endpoints)], dataset_index)

    workers = len(endpoints) * args.concurrency
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(job, position, index): index
                      for position, index in enumerate(selected_indices)}
        for future in as_completed(future_map):
            index = future_map[future]
            try:
                trajectory = future.result()
                with lock:
                    writer.write(trajectory)
                    results.append(trajectory)
                    print("  %d/%d %s success=%s" % (
                        len(results) + len(failures), len(selected_indices),
                        trajectory.task_id, trajectory.final_success), flush=True)
            except Exception as exc:
                with lock:
                    failures.append({
                        "task_id": "%s-%05d" % (args.split, index),
                        "dataset_index": index,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
                    print("  [task error] %s-%05d %s: %s" % (
                        args.split, index, type(exc).__name__, exc), flush=True)

    elapsed = time.time() - started
    completed_ids = {trajectory.task_id for trajectory in results}
    expected_ids = set(protocol["task_ids"])
    summary = {
        "version": "q2-run-summary-v1",
        "protocol_id": protocol["protocol_id"],
        "manifest_id": manifest["manifest_id"],
        "expected_tasks": len(expected_ids),
        "completed_tasks": len(completed_ids),
        "successful_tasks": sum(bool(row.final_success) for row in results),
        "missing_task_ids": sorted(expected_ids - completed_ids),
        "duplicate_task_ids": len(completed_ids) != len(results),
        "failures": failures,
        "complete": not failures and completed_ids == expected_ids
                    and len(completed_ids) == len(results),
        "elapsed_seconds": elapsed,
        "thermal_pause_events": guard.pause_events,
        "total_model_calls": sum(row.total_model_calls for row in results),
        "total_signal_calls": sum(row.total_signal_calls for row in results),
        "total_verifier_calls": sum(row.total_verifier_calls for row in results),
    }
    if not args.allow_append:
        _write_json_new(summary_path, summary)
    print("完成 %d/%d，成功 %d，用时 %.1f 分钟；摘要：%s" % (
        len(results), len(selected_indices), summary["successful_tasks"],
        elapsed / 60.0, summary_path))
    if not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
