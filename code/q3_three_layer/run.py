"""One frozen, real small-model experience -> M/P/LoRA -> held-out cycle."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .backend import HFActor, append_json, digest, file_digest, write_json
from .learning import (RidgePolicy, build_memory, features, memory_instruction,
                       paired_gate, retrieve)
from .tasks import TaskWorld, build_suite, computational_verify, independent_correct
from .protocol import build_protocol


def guided_messages(messages, entry):
    messages = [dict(m) for m in messages]
    messages.append({"role": "user", "content": "Apply this verified unit contract to the CURRENT "
        "quantity, then return only JSON: " + memory_instruction(entry)})
    return messages


def utility(success, verifier_calls=0, extra_model_calls=0):
    return float(success) - 0.10 * verifier_calls - 0.02 * extra_model_calls


def compact_generation(result):
    return {k: result[k] for k in ("key", "raw", "prediction", "finish_reason",
             "prompt_tokens", "completion_tokens", "latency_s", "cache_hit")}


class Experiment:
    def __init__(self, actor, out):
        self.actor = actor
        self.out = Path(out)
        self.events = self.out / "computational_calls.jsonl"
        self.computational_calls = 0
        self.environment_calls = 0

    def record_environment(self, world, stage, task_id):
        calls = [e for e in world.history if e.get("executed") is True and e.get("kind") in ("read", "write")]
        self.environment_calls += len(calls)
        append_json(self.out / "environment_calls.jsonl", {"stage": stage,
            "task_id": task_id, "count": len(calls), "events": calls})

    def verify(self, contract, stage, task_id):
        # This call consumes public input only; no task, oracle or expected answer.
        result = computational_verify(contract)
        self.computational_calls += 1
        append_json(self.events, {"stage": stage, "task_id": task_id,
                                 "contract": contract, **result})
        return result

    def collect(self, tasks):
        records = []
        for task in tasks:
            world = TaskWorld(task)
            public = world.prepare()
            draft = self.actor.generate(public["messages"])
            world.execute(draft["prediction"])
            first = world.evaluate()
            checked = self.verify(public["contract"], "construction_check", task["task_id"])
            # Public computational agreement decides recovery, not private oracle.
            needs_repair = not prediction_equal(draft["prediction"], checked["prediction"])
            if needs_repair:
                world.execute(checked["prediction"])
            final = world.evaluate()
            self.record_environment(world, "construction_collection", task["task_id"])
            certified = (final["success"] is True and
                         independent_correct(task, checked["prediction"]) is True)
            row = {"task_id": task["task_id"], "contract": public["contract"],
                   "messages": public["messages"], "draft": compact_generation(draft),
                   "first": first, "final": final, "history": world.history,
                   "origin": "tool_assisted_repair" if needs_repair else "model_first_pass_success",
                   "trace": checked["trace"], "prediction": checked["prediction"],
                   "certified": certified, "source_verifier_calls": 1,
                   "actor": "base"}
            records.append(row)
            write_json(self.out / "source_experiences.json", records)
        return records

    def action_outcome(self, task, public, draft, action, bank, adapter, stage, world=None):
        extra_model, verifier, request = 0, 0, None
        entry = retrieve(bank, public["contract"])
        if action == "keep":
            prediction = draft["prediction"]
        elif action == "memory" and entry is not None:
            request = self.actor.generate(guided_messages(public["messages"], entry), adapter=adapter)
            prediction = request["prediction"]
            extra_model = 1
        elif action == "verify":
            prediction = self.verify(public["contract"], stage, task["task_id"])["prediction"]
            verifier = 1
        else:
            raise ValueError("Unavailable controller action")
        # Each counterfactual is an actual fresh environment with one final commit.
        if world is None:
            world = TaskWorld(task)
            world.prepare()
        world.execute(prediction)
        assessment = world.evaluate()
        self.record_environment(world, stage, task["task_id"])
        row = {"task_id": task["task_id"], "action": action,
               "prediction": prediction, "success": assessment["success"],
               "verifier_calls": verifier, "extra_model_calls": extra_model,
               "utility": utility(assessment["success"], verifier, extra_model),
               "assessment": assessment, "history": world.history,
               "generation_key": request["key"] if request else draft["key"],
               "draft_key": draft["key"], "initial_draft": draft["prediction"],
               "memory_id": digest(entry) if entry else None,
               "actor": "adapter" if adapter else "base"}
        return row

    def evaluate(self, tasks, bank, policy=None, adapter=False, m=False, p=False,
                 control=None, stage="development"):
        rows = []
        for task in tasks:
            world = TaskWorld(task)
            public = world.prepare()
            draft = self.actor.generate(public["messages"], adapter=adapter)
            entry = retrieve(bank, public["contract"]) if m else None
            if control == "always_verify":
                action = "verify"
            elif control == "visible_conversion_rule":
                action = "verify" if public["contract"]["source_unit"] != public["contract"]["target_unit"] else "keep"
            elif p:
                actions = ["keep", "verify"] + (["memory"] if entry else [])
                action = policy.choose(features(public["contract"], draft["prediction"],
                    entry is not None, budget=1, adapter_active=adapter), actions)
            else:
                action = "memory" if entry else "keep"
            row = self.action_outcome(task, public, draft, action,
                bank if m else [], adapter, stage, world=world)
            row.update({"shape": task["shape"], "split": task["split"],
                        "M": bool(m), "P": bool(p), "L": bool(adapter)})
            rows.append(row)
        return rows

    def policy_table(self, tasks, bank):
        rows = []
        for adapter in (False, True):
            for task in tasks:
                public = TaskWorld(task).prepare()
                draft = self.actor.generate(public["messages"], adapter=adapter)
                # Two public contexts, each with the exact same initial draft.
                for has_memory in (False, True):
                    selected_bank = bank if has_memory else []
                    available = retrieve(selected_bank, public["contract"]) is not None
                    vector = features(public["contract"], draft["prediction"], available,
                                      budget=1, adapter_active=adapter)
                    for action in (["keep", "verify", "memory"] if available else ["keep", "verify"]):
                        outcome = self.action_outcome(task, public, draft, action,
                            selected_bank, adapter, "construction_policy_table")
                        rows.append({**outcome, "features": vector,
                                     "memory_available": available, "reward": outcome["utility"]})
        write_json(self.out / "policy_training_rows.json", rows)
        return rows


def prediction_equal(left, right):
    from decimal import Decimal, InvalidOperation
    try:
        return (left is not None and right is not None and left["unit"] == right["unit"]
                and Decimal(str(left["value"])) == Decimal(str(right["value"])))
    except (KeyError, InvalidOperation, TypeError):
        return False


def score(rows):
    return {"n": len(rows), "successes": sum(r["success"] is True for r in rows),
            "success_rate": sum(r["success"] is True for r in rows) / len(rows) if rows else None,
            "mean_utility": sum(r["utility"] for r in rows) / len(rows) if rows else None,
            "verifier_calls": sum(r["verifier_calls"] for r in rows),
            "extra_model_calls": sum(r["extra_model_calls"] for r in rows),
            "environment_calls": sum(r["assessment"]["environment_calls"] for r in rows),
            "actions": dict(Counter(r["action"] for r in rows))}


def compare(base, candidate, require_fix=True, min_pairs=1, utility_gate=False):
    if [r["task_id"] for r in base] != [r["task_id"] for r in candidate]:
        raise ValueError("Paired outcomes must retain the exact same task order")
    gate = paired_gate([r["success"] for r in base], [r["success"] for r in candidate],
                      min_pairs=min_pairs, require_fix=require_fix)
    gain = sum(r["utility"] for r in candidate) - sum(r["utility"] for r in base)
    gate["utility_gain_total"] = gain
    gate["budget_compliant"] = all(r["verifier_calls"] <= 1 for r in candidate)
    if utility_gate:
        if gate["pass"] and not gate["budget_compliant"]:
            gate.update({"pass": False, "reason": "budget_violation"})
        elif gate["pass"] and gain <= 1e-12:
            gate.update({"pass": False, "reason": "no_utility_gain"})
    return gate


def admit_memory(experiment, tasks, bank, adapter, baseline):
    candidate = experiment.evaluate(tasks, bank, adapter=adapter, m=True,
                                    stage="development_memory")
    groups = {}
    for index, task in enumerate(tasks):
        target = TaskWorld(task).prepare()["contract"]["target_unit"]
        groups.setdefault(target, []).append(index)
    decisions = {}
    accepted_targets = set()
    for target, indices in groups.items():
        b = [baseline[i] for i in indices]
        c = [candidate[i] for i in indices]
        gate = compare(b, c, min_pairs=4)
        gate["task_ids"] = [r["task_id"] for r in b]
        members = {}
        for index in indices:
            contract = TaskWorld(tasks[index]).prepare()["contract"]
            pair = contract["source_unit"] + "->" + contract["target_unit"]
            members.setdefault(pair, []).append(index)
        gate["members"] = {pair: compare([baseline[i] for i in member_indices],
            [candidate[i] for i in member_indices], min_pairs=2, require_fix=False)
            for pair, member_indices in members.items()}
        actual_coverage = all(candidate[i]["action"] == "memory" and candidate[i]["memory_id"]
                              for i in indices)
        gate["actual_memory_coverage"] = bool(actual_coverage)
        if (len(members) != 2 or any(len(v) != 2 for v in members.values())
                or not actual_coverage):
            gate.update({"pass": False, "reason": "missing_conversion_identity_coverage"})
        decisions[target] = gate
        if gate["pass"]:
            accepted_targets.add(target)
    accepted = [entry for entry in bank if entry["target_unit"] in accepted_targets]
    combination = experiment.evaluate(tasks, accepted, adapter=adapter, m=True,
                                      stage="development_memory_combination")
    whole = compare(baseline, combination, min_pairs=12)
    if not whole["pass"]:
        accepted = []
    return {"actor": "adapter" if adapter else "base", "bundles": decisions,
            "combination_gate": whole, "accepted": accepted, "candidate_rows": candidate,
            "combination_rows": combination}


def trainable_examples(source):
    groups = {}
    for row in source:
        if row["certified"] is True:
            key = (row["contract"]["source_unit"], row["contract"]["target_unit"])
            groups.setdefault(key, set()).add(row["contract"]["entity"])
    eligible = [r for r in source if r["certified"] is True and
        len(groups[(r["contract"]["source_unit"], r["contract"]["target_unit"])]) >= 2]
    if not any(r["contract"]["source_unit"] == r["contract"]["target_unit"] for r in eligible):
        raise ValueError("Training needs actual identity controls")
    if not any(r["contract"]["source_unit"] != r["contract"]["target_unit"] for r in eligible):
        raise ValueError("Training needs actual conversion evidence")
    return [{"task_id": r["task_id"], "messages": r["messages"],
             "prediction": r["prediction"], "origin": r["origin"]} for r in eligible]


def file_hashes():
    root = Path(__file__).resolve().parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to a standard local Qwen3.5 HF checkpoint")
    parser.add_argument("--out", default="outputs/q3_three_layer/run")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--construction-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suite = build_suite()
    protocol = build_protocol(suite)
    if (out / "protocol.json").exists():
        if not args.resume or json.loads((out / "protocol.json").read_text(encoding="utf-8")) != protocol:
            raise ValueError("Protocol changed; a new run is required")
    else:
        write_json(out / "protocol.json", protocol)
        write_json(out / "suite.json", suite)
    actor = HFActor(args.model, out, resume=args.resume)
    experiment = Experiment(actor, out)
    source = experiment.collect(suite["construction"])
    bank = build_memory(source)
    write_json(out / "candidate_memory.json", bank)
    source_pairs = {(r["source_unit"], r["target_unit"]) for r in bank}
    examples = trainable_examples(source)
    training_ids = {r["task_id"] for r in examples}
    write_json(out / "experience_routing.json", [{"task_id": r["task_id"],
        "origin": r["origin"], "M_source_eligible": (
            r["contract"]["source_unit"], r["contract"]["target_unit"]) in source_pairs,
        "M_reason": "actual trace + independent certificate + two distinct entities",
        "P_eligible": True, "P_reason": "observed paired construction action outcomes, including failures",
        "L_eligible": r["task_id"] in training_ids,
        "L_reason": "independently correct positive target and two-entity unit-pair support"}
        for r in source])
    if args.construction_only:
        print("Construction-only technical smoke complete; NO held-out calls", flush=True)
        return
    dev = suite["development"]
    base = experiment.evaluate(dev, [], stage="development_base")
    memory_base = admit_memory(experiment, dev, bank, False, base)
    write_json(out / "memory_gate_base.json", memory_base)
    adapter_dir = out / "adapter"
    if (out / "training.json").exists():
        from peft import PeftModel
        training = json.loads((out / "training.json").read_text(encoding="utf-8"))
        if not training.get("frozen_unchanged") or not training.get("adapter_changed"):
            raise ValueError("Existing training evidence did not verify a real isolated update")
        for name, checksum in training["adapter_files"].items():
            if file_digest(adapter_dir / name) != checksum:
                raise ValueError("Saved adapter bytes/config changed")
        actor.model = PeftModel.from_pretrained(actor.model, str(adapter_dir), is_trainable=False)
        actor.has_adapter = True
        if actor.parameter_hash(True) != training["adapter_after"]:
            raise ValueError("Loaded adapter parameters do not match training identity")
        actor.adapter_hash = training["adapter_after"]
    else:
        training = actor.train_adapter(examples, adapter_dir)
    if (out / "reload_check.json").exists():
        if not json.loads((out / "reload_check.json").read_text(encoding="utf-8")).get("pass"):
            raise ValueError("Saved adapter reload failed; refusing to bypass its result")
    else:
        actor.verify_reload(source[0]["messages"], adapter_dir)
    adapted = experiment.evaluate(dev, [], adapter=True, stage="development_adapter")
    lora_gate = compare(base, adapted, min_pairs=12)
    write_json(out / "lora_gate.json", {**lora_gate, "base": base, "candidate": adapted})
    memory_adapted = admit_memory(experiment, dev, bank, True, adapted)
    write_json(out / "memory_gate_adapter.json", memory_adapted)
    table = experiment.policy_table(suite["construction"], bank)
    policy = RidgePolicy.fit(table)
    write_json(out / "policy.json", policy.to_dict())
    write_json(out / "policy_decisions_construction.json", [{"task_id": r["task_id"],
        "actor": r["actor"], "memory_available": r["memory_available"],
        "before_action": "keep", "after_action": policy.choose(r["features"],
            ["keep", "verify"] + (["memory"] if r["memory_available"] else []))}
        for r in table if r["action"] == "keep"])
    use_l = lora_gate["pass"]
    accepted_bank = (memory_adapted if use_l else memory_base)["accepted"]
    chosen_actor_base = adapted if use_l else base
    policy_dev = experiment.evaluate(dev, accepted_bank, policy, adapter=use_l,
        m=bool(accepted_bank), p=True, stage="development_policy")
    policy_gate = compare(chosen_actor_base, policy_dev, min_pairs=12,
                          require_fix=False, utility_gate=True)
    use_p = policy_gate["pass"]
    proposed = (policy_dev if use_p else experiment.evaluate(dev, accepted_bank,
        adapter=use_l, m=bool(accepted_bank), stage="development_composition"))
    composition_gate = compare(base, proposed, min_pairs=12, require_fix=False, utility_gate=True)
    selected = {"M": bool(accepted_bank), "P": use_p, "L": use_l}
    if not composition_gate["pass"]:
        selected = {"M": False, "P": False, "L": False}
    selection = {"lora_gate": lora_gate, "policy_gate": policy_gate,
        "composition_gate": composition_gate, "selected": selected,
        "accepted_memory": accepted_bank if selected["M"] else [],
        "policy_rows": policy_dev, "proposed_rows": proposed,
        "versions": {"M_c": digest(bank), "P_c": digest(policy.to_dict()),
                     "L_c": actor.adapter_hash}, "implementation_hashes": file_hashes(),
        "protocol_sha256": digest(protocol)}
    frozen = out / "selection_frozen.json"
    if frozen.exists():
        previous = json.loads(frozen.read_text(encoding="utf-8"))
        if digest(previous) != digest(selection):
            raise ValueError("Frozen selection or implementation changed; start a new output directory")
        selection = previous
    else:
        write_json(frozen, selection)
    # No memory/policy/adapter/threshold updates after this line.
    heldout = suite["source_test"] + suite["chain_test"] + suite["qa_test"]
    results, summaries = [], {}
    for m in (False, True):
        for p in (False, True):
            for l in (False, True):
                label = f"candidate_M{int(m)}P{int(p)}L{int(l)}"
                rows = experiment.evaluate(heldout, bank, policy, adapter=l, m=m, p=p, stage=label)
                for row in rows:
                    row["condition"] = label
                results.extend(rows)
                summaries[label] = {"all": score(rows), **{shape: score([r for r in rows if r["shape"] == shape])
                    for shape in ("single_write", "read_write", "text_qa")}}
                write_json(out / "evaluation.json", results)
    for control in ("always_verify", "visible_conversion_rule", "accepted"):
        if control == "accepted":
            rows = experiment.evaluate(heldout, selection["accepted_memory"], policy,
                adapter=selected["L"], m=selected["M"], p=selected["P"], stage=control)
        else:
            rows = experiment.evaluate(heldout, [], control=control, stage=control)
        for row in rows:
            row["condition"] = control
        results.extend(rows)
        summaries[control] = {"all": score(rows), **{shape: score([r for r in rows if r["shape"] == shape])
            for shape in ("single_write", "read_write", "text_qa")}}
    write_json(out / "evaluation.json", results)
    baseline_rows = [r for r in results if r["condition"] == "candidate_M0P0L0"]
    for label in summaries:
        comparison = [r for r in results if r["condition"] == label]
        summaries[label]["paired_vs_base"] = compare(baseline_rows, comparison, require_fix=False)
    totals = {"actual_model_requests_in_this_process": actor.actual_calls,
              "cache_hits_in_this_process": actor.cache_hits,
              "actual_computational_calls_in_this_process": experiment.computational_calls,
              "actual_environment_calls_in_this_process": experiment.environment_calls,
              "logical_heldout_episodes": len(results), "source_origins": dict(Counter(r["origin"] for r in source)),
              "training_verified": training["adapter_changed"] and training["frozen_unchanged"]}
    write_json(out / "summary.json", {"conditions": summaries, "execution": totals})
    print("DONE: " + json.dumps(totals), flush=True)


if __name__ == "__main__":
    main()
