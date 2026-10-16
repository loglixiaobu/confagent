"""Reproducible Q3 construction → promotion → frozen evaluation experiment.

The synthetic suite has an independent semantic oracle. Model calls use only
public requirements, tool schemas and past observations. No test result updates
memory. Supply the served model and runtime directory explicitly.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

from audited_client import local_client, local_runtime_identity
from apibank_env import APIBankMockEnv
from q3_tasks import build_suite, evaluate_trajectory, make_probe, oracle_action
from react_agent import AgentConfig, SYSTEM_PROMPT, run_task, tool_searcher_schema
from repair_memory import (MemoryBank, assert_disjoint_splits, candidate_from_proposal,
                           digest, json_object, memory_messages, source_certificate,
                           validate_bank, validate_transfer)

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def parsed_action(response):
    calls = response.choices[0].message.tool_calls or []
    if not calls:
        return None
    try:
        args = json.loads(calls[0].function.arguments)
    except (ValueError, TypeError):
        return None
    if not isinstance(args, dict):
        return None
    return {"api_name": calls[0].function.name, "arguments": args}


def probe_for(task):
    probe = make_probe(task)
    # A probe and the actual PR1 actor share their system instruction.
    probe["messages"][0] = {"role": "system", "content": SYSTEM_PROMPT}
    probe["messages"][1]["content"] = "Requirement: " + task["requirement"].strip()
    probe["tools"] = [tool_searcher_schema()] + probe["tools"]
    return probe


def forbidden_values(before, after, task=None):
    generic = {"0", "1", "100", "1000", "1024", "60", "24"}
    values = set()
    for action in (before, after):
        for key, value in action.get("arguments", {}).items():
            if isinstance(value, (str, int, float)) and str(value) not in generic:
                values.add(str(value))
                if key.endswith("_name") and isinstance(value, str):
                    # This suite names entities "<unique tree name> <family>".
                    values.add(value.split()[0])
    if task is not None:
        # The requirement's instance inputs also need anonymization, even when
        # a wrong candidate has already converted them to a different number.
        meta = task["_q3"]
        for key in ("source_value", "local_datetime", "entity"):
            value = meta.get(key)
            if value is not None and str(value) not in generic:
                values.add(str(value))
    return sorted(values)


def extract_repairs(task, trajectory):
    repairs, previous_wrong = [], {}
    for step in trajectory["steps"]:
        if step["action_type"] == "answer":
            continue
        after = step["action_input"]
        before = step.get("candidate_action") or after
        kind = "pre_execution"
        if oracle_action(task, before) and after["api_name"] in previous_wrong:
            before = previous_wrong[after["api_name"]]
            kind = "local_retry"
        certificate = source_certificate(task["_q3"]["id"], before, after,
                                         lambda action: oracle_action(task, action))
        observation = step.get("observation")
        actual_return = (isinstance(observation, dict) and observation.get("success") is True
                         and "error" not in observation and not observation.get("exception")
                         and oracle_action(task, {"api_name": after["api_name"],
                             "arguments": {k: v for k, v in observation.items() if k != "success"}}))
        if certificate["is_repair"] and actual_return:
            repairs.append({"source_task_id": task["_q3"]["id"], "family": task["_q3"]["family"],
                            "source_kind": kind, "requirement": task["requirement"],
                            "before": before, "after": after, "observation": observation,
                            "certificate": certificate, "step_id": step["step_id"]})
        if not oracle_action(task, after):
            previous_wrong[after["api_name"]] = after
    return repairs


def propose(client, model, experience, tools, seed):
    evidence = {key: experience[key] for key in ("requirement", "before", "after", "observation")}
    evidence["tools"] = tools
    response = client.chat.completions.create(
        model=model, temperature=0, max_tokens=320, seed=seed,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        messages=[{"role": "system", "content": "Extract one reusable, conditional tool-use lesson from an observed correction."},
                  {"role": "user", "content": (
                      "Return ONLY JSON with keys instruction (one concise conditional rule), "
                      "api_names (list of API names from the evidence), cues (empty list). "
                      "Explain the general unit/time operation and when it does NOT apply. "
                      "Never copy instance names, dates or source numeric values. Generic unit "
                      "conversion constants such as 100, 1000 and 1024 are allowed. "
                      "Do not propose a new tool or an action to execute.\nEVIDENCE:\n"
                      + json.dumps(evidence, ensure_ascii=False))}])
    raw = response.choices[0].message.content or ""
    return json_object(raw), raw


def recover_source(task, episode, client, model):
    """One extra model repair after a visible failure; no oracle labels in prompt.

    This optional source intervention is separately recorded. A fresh mock
    executes the failed action then the proposed repair, retaining actual effects.
    It is specific to this suite's single-business-action, failed-write tasks.
    """
    failed = [s for s in episode["steps"] if isinstance(s.get("observation"), dict)
              and "error" in s["observation"] and s["action_input"].get("api_name") != "ToolSearcher"]
    if not failed or any(isinstance(s.get("observation"), dict) and s["observation"].get("success") is True
                         for s in episode["steps"]):
        return None, None
    step = failed[-1]
    before = step["action_input"]
    probe = probe_for(task)
    response = client.chat.completions.create(
        model=model, temperature=0, max_tokens=384, seed=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        tools=probe["tools"][1:], tool_choice="required",
        messages=[{"role": "system", "content": (
            "Repair a failed tool call. Derive the required argument from the user input and tool documentation. "
            "Recompute units and fixed UTC offsets carefully. Preserve the entity name. "
            "A supplied source number may need conversion; when already in the target unit leave it unchanged. "
            "For a UTC timestamp use the ISO date and time with a Z suffix. Return exactly one corrected tool call.")},
            {"role": "user", "content": json.dumps({"requirement": task["requirement"],
             "failed_action": before, "observed_failure": step["observation"]}, ensure_ascii=False)}])
    after = parsed_action(response)
    env = APIBankMockEnv(task, feedback_mode="realistic")
    before_result = env.call(before["api_name"], before["arguments"])
    observation = env.call(after["api_name"], after["arguments"]).observation if after else None
    certificate = source_certificate(task["_q3"]["id"], before, after, lambda action: oracle_action(task, action))
    actual_ok = (isinstance(observation, dict) and observation.get("success") is True
                 and "error" not in observation and oracle_action(task, {"api_name": after["api_name"],
                    "arguments": {k: v for k, v in observation.items() if k != "success"}}))
    record = {"task_id": task["_q3"]["id"], "before": before, "after": after,
              "before_observation": before_result.observation, "observation": observation,
              "before_exception": before_result.exception,
              "certificate": certificate, "actual_repair": bool(certificate["is_repair"] and actual_ok)}
    if not record["actual_repair"]:
        return record, None
    experience = {"source_task_id": task["_q3"]["id"], "family": task["_q3"]["family"],
                  "source_kind": "post_episode_doc_recovery", "requirement": task["requirement"],
                  "before": before, "after": after, "observation": observation,
                  "certificate": certificate, "step_id": step["step_id"]}
    return record, experience


def run_episode(task, cfg, client, index, bank=None, mode="validated"):
    started = time.perf_counter()
    trajectory = run_task(task, cfg, client, task_index=index, memory_bank=bank, memory_mode=mode).to_dict()
    trajectory["task_id"] = task["_q3"]["id"]
    for step in trajectory["steps"]:
        step["task_id"] = trajectory["task_id"]
    trajectory["q3_family"] = task["_q3"]["family"]
    trajectory["q3_split"] = task["_q3"]["split"]
    trajectory["wall_time_s"] = time.perf_counter() - started
    trajectory["independent_evaluation"] = evaluate_trajectory(task, trajectory)
    return trajectory


def request_totals(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    usages = [(row.get("response") or {}).get("usage") for row in rows]
    return {"actual_requests": len(rows), "failed_requests": sum("error" in row for row in rows),
            "responses_with_usage": sum(usage is not None for usage in usages),
            "prompt_tokens": sum((usage or {}).get("prompt_tokens", 0) for usage in usages),
            "completion_tokens": sum((usage or {}).get("completion_tokens", 0) for usage in usages),
            "latency_s": sum(row["latency_s"] for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument("--runtime-dir", type=Path, required=True, help="Runtime directory containing the server/model manifest")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/q3_memory/run")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source-recovery", action="store_true", help="Exploratory extra doc-grounded repair call after failed construction episodes")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    seeds = [int(seed) for seed in args.seeds.split(",")]
    if not seeds or len(seeds) != len(set(seeds)):
        parser.error("Seeds must be nonempty and unique")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    identity = local_runtime_identity(args.endpoint, args.runtime_dir)
    suite = build_suite()
    assert_disjoint_splits(suite, lambda task: digest({key: probe_for(task)[key] for key in ("messages", "tools")}))
    protocol = {"version": "q3-tvrm-local-v1", "base_pr1": "84401ac1279f8f1bd6eb12aab3d245c195010317",
                "model": args.model, "temperature": 0, "thinking": False, "rounds": args.rounds,
                "reported_final_round": args.rounds,
                "seeds": seeds, "split_ids": {key: [task["_q3"]["id"] for task in tasks] for key, tasks in suite.items()},
                "suite_sha256": digest(suite), "source_policy": "always-critic-B2",
                "evaluation_policy": "vanilla-B0", "validation_target": "raw_actor_next_action",
                "source_recovery": "one_post_episode_doc_grounded_call" if args.source_recovery else "none",
                "memory_top_k": 2, "memory_char_cap": 900, "individual_gate": {"minimum_pairs": 4, "harms": 0, "minimum_fixes": 1},
                "joint_bank_gate": "zero_harm_and_positive_gain_or_empty_identity",
                "test_updates_memory": False,
                "runtime_identity": identity,
                "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted((ROOT / "code").glob("*.py"))}}
    protocol_path = args.out_dir / "protocol.json"
    if protocol_path.exists():
        if not args.resume or json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Run/protocol already exists or changed; use a new output directory")
    write_json(protocol_path, protocol)
    write_json(args.out_dir / "suite.json", suite)
    client = local_client(args.endpoint, args.out_dir / "requests.jsonl", resume=args.resume, identity=identity)
    tasks_by_id = {task["_q3"]["id"]: task for tasks in suite.values() for task in tasks}
    validation_probes = [probe_for(task) for task in suite["validation"]]

    def generate(probe, selected):
        response = client.chat.completions.create(
            model=args.model, messages=memory_messages(probe["messages"], selected), tools=probe["tools"],
            tool_choice="auto", temperature=0, max_tokens=320, seed=0, logprobs=True, top_logprobs=5,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return parsed_action(response)

    def judge(probe, action):
        return oracle_action(tasks_by_id[probe["task_id"]], action)

    current_bank, snapshots, pools, round_stats = MemoryBank(), [], [], []
    for round_index in range(1, args.rounds + 1):
        folder = args.out_dir / ("round_%d" % round_index)
        source_cfg = AgentConfig(model=args.model, max_steps=4, max_tokens=320,
                                 enable_thinking=False, verify_policy="always", verifier="critic",
                                 verify_budget=2, feedback_mode="realistic", seed=0)
        episodes, experiences, recoveries = [], [], []
        for index, task in enumerate(suite["construction"]):
            episode = run_episode(task, source_cfg, client, index, current_bank if current_bank.memories else None)
            episodes.append(episode)
            experiences.extend(extract_repairs(task, episode))
            if args.source_recovery:
                recovery, experience = recover_source(task, episode, client, args.model)
                if recovery:
                    recoveries.append(recovery)
                if experience:
                    experiences.append(experience)
        write_rows(folder / "construction.jsonl", episodes)
        write_rows(folder / "experiences.jsonl", experiences)
        write_rows(folder / "source_recovery.jsonl", recoveries)
        pool = {memory.memory_id: copy.deepcopy(memory) for memory in current_bank.memories}
        proposals = []
        for experience in experiences:
            probe = probe_for(tasks_by_id[experience["source_task_id"]])
            proposal, raw = propose(client, args.model, experience, probe["tools"], seed=0)
            record = {"source_task_id": experience["source_task_id"], "raw": raw, "proposal": proposal}
            try:
                memory = candidate_from_proposal(proposal, experience["source_task_id"], round_index,
                    experience["certificate"], probe["api_names"], forbidden_values(experience["before"], experience["after"], tasks_by_id[experience["source_task_id"]]))
                if memory.memory_id in pool:
                    pool[memory.memory_id].source_task_ids = sorted(set(pool[memory.memory_id].source_task_ids + memory.source_task_ids))
                else:
                    pool[memory.memory_id] = memory
                record["memory_id"] = memory.memory_id
            except ValueError as exc:
                record["rejection"] = str(exc)
            proposals.append(record)
        write_rows(folder / "proposals.jsonl", proposals)
        promoted = []
        joint_records = []
        for memory in sorted(pool.values(), key=lambda item: item.memory_id):
            validate_transfer(memory, validation_probes, generate, judge)
            if memory.status == "accepted":
                trial = MemoryBank(promoted + [memory], version=round_index)
                joint = validate_bank(trial, validation_probes, generate, judge)
                joint_records.append(joint)
                if joint["accepted"]:
                    promoted.append(memory)
                else:
                    memory.status = "rejected"
                    memory.validation["reason"] = "joint_bank_harm_or_no_gain"
        current_bank = MemoryBank(copy.deepcopy(promoted), version=round_index)
        full_pool = MemoryBank(copy.deepcopy(list(pool.values())), version=round_index)
        current_bank.save(folder / "accepted_bank.json")
        full_pool.save(folder / "candidate_bank.json")
        write_json(folder / "joint_validation.json", joint_records)
        stats = {"round": round_index, "construction_episodes": len(episodes), "source_repairs": len(experiences),
                 "proposal_count": len(proposals), "parsed_candidates": len(pool), "accepted": len(promoted),
                 "construction_strict_success": sum(ep["independent_evaluation"]["strict_success"] for ep in episodes),
                 "extra_source_recovery_attempts": len(recoveries),
                 "extra_source_recovery_successes": sum(row["actual_repair"] for row in recoveries),
                 "actual_requests_logged_when_round_materialized": request_totals(args.out_dir / "requests.jsonl")["actual_requests"]}
        round_stats.append(stats)
        snapshots.append(copy.deepcopy(current_bank))
        pools.append(full_pool)
        print(json.dumps({"stage": "memory_round", **stats}), flush=True)

    # All construction/promotion is finished before the first held-out test call.
    conditions = [("no_memory", None, "validated"), ("unfiltered_r1", pools[0], "unfiltered"),
                  ("validated_r1", snapshots[0], "validated"), ("unscoped_r1", snapshots[0], "unscoped")]
    if args.rounds > 1:
        conditions.append(("validated_r%d" % args.rounds, snapshots[-1], "validated"))
    all_rows = []
    for condition, bank, mode in conditions:
        frozen_hash = bank.sha256 if bank is not None else None
        rows = []
        for seed in seeds:
            cfg = AgentConfig(model=args.model, max_steps=4, max_tokens=320,
                              enable_thinking=False, verify_policy="vanilla", verify_budget=0,
                              feedback_mode="realistic", seed=seed)
            for index, task in enumerate(suite["test"]):
                trajectory = run_episode(task, cfg, client, index, bank, mode)
                trajectory["q3_condition"] = condition
                rows.append(trajectory)
                if bank is not None and bank.sha256 != frozen_hash:
                    raise AssertionError("Held-out evaluation mutated a frozen memory bank")
        write_rows(args.out_dir / "evaluation" / (condition + ".jsonl"), rows)
        all_rows.extend(rows)
        print(json.dumps({"stage": "heldout", "condition": condition, "episodes": len(rows),
                          "strict_success": sum(row["independent_evaluation"]["strict_success"] for row in rows)}), flush=True)
    totals = request_totals(args.out_dir / "requests.jsonl")
    write_json(args.out_dir / "run_summary.json", {"rounds": round_stats, "actual_model_cost": totals,
               "deterministic_cache_hits_this_process": client.cache_hits, "evaluation_episodes": len(all_rows),
               "conditions": [condition for condition, _, _ in conditions], "complete": True})
    from analyze_q3 import analyze
    analyze(args.out_dir)


if __name__ == "__main__":
    main()
