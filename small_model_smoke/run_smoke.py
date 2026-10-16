#!/usr/bin/env python
"""Run the existing agent loop with a tiny local model, no vLLM required."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = HERE / "cache"
TMP = HERE / "tmp"


def configure_e_drive_environment() -> None:
    """Pin every runtime/cache path before importing ML libraries."""
    for path in (CACHE, TMP, HERE / "outputs"):
        path.mkdir(parents=True, exist_ok=True)
    variables = {
        "HF_HOME": CACHE / "huggingface",
        "HF_HUB_CACHE": CACHE / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": CACHE / "huggingface" / "transformers",
        "TORCH_HOME": CACHE / "torch",
        "XDG_CACHE_HOME": CACHE,
        "MPLCONFIGDIR": CACHE / "matplotlib",
        "TEMP": TMP,
        "TMP": TMP,
    }
    for name, path in variables.items():
        resolved = path.resolve()
        if resolved.drive.upper() != "E:":
            raise RuntimeError("%s must point to E:, got %s" % (name, resolved))
        resolved.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(resolved)
    # Imports of project modules must not create __pycache__ in code/.
    sys.pycache_prefix = str((CACHE / "pycache").resolve())


configure_e_drive_environment()

import torch  # noqa: E402

sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(HERE))

from local_transformers_client import LocalTransformersClient  # noqa: E402
from react_agent import AgentConfig, run_task  # noqa: E402
from trajectory import TrajectoryWriter  # noqa: E402


def read_tasks(path: Path, limit: int) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        tasks = json.load(source)
    tasks = [task for task in tasks if isinstance(task, dict) and task.get("apis")]
    return tasks[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(HERE / "models" / "Qwen3-0.6B"))
    parser.add_argument("--fixture", default=str(HERE / "fixtures" / "level3_smoke.json"))
    parser.add_argument("--output-dir", default=str(HERE / "outputs"))
    parser.add_argument("--conditions", default="vanilla,always")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--cpu-threads", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--dtype", choices=["float32", "auto"], default="float32",
                        help="float32 is normally faster than BF16 on older CPUs")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    if out_dir.drive.upper() != "E:" or HERE not in out_dir.parents:
        parser.error("--output-dir must stay below %s on E:" % HERE)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = read_tasks(Path(args.fixture), args.limit)
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    invalid = set(conditions) - {"vanilla", "always", "provenance"}
    if invalid:
        parser.error("unsupported smoke condition(s): %s" % sorted(invalid))

    model_dir = Path(args.model_dir).resolve()
    if model_dir.drive.upper() != "E:" or HERE not in model_dir.parents:
        parser.error("--model-dir must stay below %s on E:" % HERE)
    print("Loading local model from:", model_dir, flush=True)
    client = LocalTransformersClient(
        str(model_dir),
        max_new_tokens=args.max_new_tokens,
        cpu_threads=args.cpu_threads,
        dtype=args.dtype,
    )
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "dtype": args.dtype,
        "model_dir": str(model_dir),
        "model_dir_drive": model_dir.drive,
        "cache_dir": os.environ["HF_HOME"],
        "fixture": str(Path(args.fixture).resolve()),
        "conditions": conditions,
        "task_count": len(tasks),
        "seed": args.seed,
        "note": "Engineering smoke test; not a formal benchmark result.",
    }
    (out_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for condition in conditions:
        output = out_dir / ("trajectories_%s.jsonl" % condition)
        if output.exists():
            output.unlink()
        writer = TrajectoryWriter(str(output))
        cfg = AgentConfig(
            model="Qwen3-0.6B-local-cpu",
            max_steps=args.max_steps,
            max_tokens=args.max_new_tokens,
            temperature=0.0,
            enable_thinking=False,
            verify_policy=condition,
            verifier="critic",
            verify_budget=0 if condition == "vanilla" else args.max_steps,
            risk_threshold=0.60,
            replan_after=2,
            max_retries=3,
            feedback_mode="realistic",
            seed=args.seed,
            method=("vanilla_react" if condition == "vanilla"
                    else "conf_verify_%s_critic" % condition),
            verbose=True,
        )
        started = time.perf_counter()
        print("\n=== condition=%s tasks=%d ===" % (condition, len(tasks)), flush=True)
        for index, task in enumerate(tasks):
            print("  task %d/%d: %s" % (index + 1, len(tasks), task["requirement"]), flush=True)
            trajectory = run_task(task, cfg, client, task_index=index)
            writer.write(trajectory)
            print(
                "  => replay=%s final=%s clean=%s recovered=%s stop=%s"
                % (
                    trajectory.replay_success,
                    trajectory.final_success,
                    trajectory.clean_success,
                    trajectory.recovered_success,
                    trajectory.config.get("stop_reason"),
                ),
                flush=True,
            )
        print("condition %s finished in %.1fs -> %s" % (
            condition, time.perf_counter() - started, output
        ), flush=True)


if __name__ == "__main__":
    main()
