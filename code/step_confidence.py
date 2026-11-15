#!/usr/bin/env python
"""Reproducible pre-execution confidence experiment on API-Bank reference prefixes.

Model-visible states and offline gold labels are separate files. This is local
step evaluation, not an online task-success benchmark or real API execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import threading
import time
from decimal import Decimal, InvalidOperation
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from openai import OpenAI

from apibank_env import (APIBankMockEnv, arithmetic_value, api_schema_to_tool, load_level3,
                         parameter_schema, structured_equal, typed_equal)
from react_agent import SYSTEM_PROMPT, extract_confidence_features
from run_batch import max_gpu_temp

ROOT = Path(__file__).resolve().parents[1]
VERSION = "reference-prefix-v1"
GENERATOR_PROMPT = SYSTEM_PROMPT + "\nKeep your reasoning brief, at most 80 words. Do not restate the tool documentation."
REQUEST_LOCK = threading.Lock()


class ConservativeGuard:
    """Global single-flight inference, cooldown, temperature hysteresis, audit log."""
    def __init__(self, out_dir, cooldown=6.0, pause_at=60.0, resume_at=50.0):
        self.cooldown, self.pause_at, self.resume_at = cooldown, pause_at, resume_at
        self.log_path = Path(out_dir) / 'thermal.jsonl'

    def wait_if_hot(self):
        temp = max_gpu_temp()
        if temp is None:
            raise RuntimeError('Temperature unavailable; refusing to submit GPU work')
        paused = temp >= self.pause_at
        while temp >= self.pause_at or (paused and temp > self.resume_at):
            with self.log_path.open('a') as f:
                f.write(json.dumps({'time': time.time(), 'max_gpu_temp': temp, 'event': 'cooling'})+'\n')
            print(f'[thermal] {temp}C; waiting until <= {self.resume_at}C', flush=True)
            time.sleep(5)
            temp = max_gpu_temp()
            if temp is None: raise RuntimeError('Temperature sensor failed during cooldown')
        with self.log_path.open('a') as f:
            f.write(json.dumps({'time': time.time(), 'max_gpu_temp': temp, 'event': 'before_request'})+'\n')

VERIFIER_PROMPT = """You audit ONE proposed API call before execution.
Use ONLY the supplied requirement, API documents, and earlier calls/results.
Check whether the API is useful at this point, required arguments are present,
types are valid, prerequisites are satisfied, and EACH value is supported.
Values may be directly quoted or derived by a valid calculation or conversion.
Do not change explicit coordinates to match your geographic knowledge. A numeric
representation change is not a semantic error. Do not demand exact wording for
email content. Never invent missing dates, locations, IDs, or defaults.
For every argument output status supported, contradicted, or unknown, with a
short exact source quotation and source identifier (requirement or history index).
unknown means insufficient evidence, NOT proven wrong. Consider the whole call,
not just JSON validity. Return ONLY JSON, no markdown or reasoning narrative:
{"api_suitable":true,"arguments":{"field":{"status":"supported",
"evidence":"source quotation","source":"requirement"}},"confidence":0.8}
confidence is your probability in [0,1] that the entire call is correct.
"""


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def read_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def schema_valid(value, schema):
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict): return False
        props = schema.get("properties", {})
        return (all(k in value for k in schema.get("required", []))
                and all(k in props and schema_valid(v, props[k]) for k, v in value.items()))
    if kind == "array":
        return isinstance(value, list) and all(schema_valid(v, schema.get("items", {})) for v in value)
    if kind == "number": return type(value) in (int, float) and math.isfinite(value)
    if kind == "integer": return type(value) in (int, float) and math.isfinite(value) and int(value) == value
    if kind == "string": return isinstance(value, str)
    if kind == "boolean": return isinstance(value, bool)
    return not schema


def literal_grounding(action, state):
    """Cheap baseline: fraction of scalar argument values present in visible evidence.

    This deliberately cannot verify reasoning, field roles, or arbitrary unit
    conversions. API documents and reference labels are not evidence values.
    """
    text=json.dumps({'requirement':state['requirement'],'history':state['history']},ensure_ascii=False).lower()
    numbers=set()
    for token in re.findall(r'(?<![\w.])[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w.])',text):
        try:numbers.add(Decimal(token))
        except InvalidOperation:pass
    found=[]
    def visit(value):
        if isinstance(value,dict):
            for item in value.values():visit(item)
        elif isinstance(value,list):
            for item in value:visit(item)
        elif type(value) in (int,float):found.append(Decimal(str(value)) in numbers)
        elif isinstance(value,str):found.append(value.lower().strip() in text if value.strip() else False)
        else:found.append(False)
    visit(action.get('arguments',{}))
    return sum(found)/len(found) if found else None


def canonical_reference_args(args, api):
    """Repair storage types in reference calls according to public API schema."""
    out = dict(args)
    for key, meta in api.get("input_parameters", {}).items():
        kind = parameter_schema(meta["type"])["type"]
        if key in out and kind in ("number", "integer") and isinstance(out[key], str):
            out[key] = float(out[key]) if kind == "number" else int(out[key])
    return out


def exclusion_reason(task_index, api_name):
    if task_index == 35:
        return "Requirement omits source, destination and travel dates"
    if api_name == "AddMeeting":
        return "Required location not specified; gold chooses an unsupported conference room"
    if api_name == "HotelAvailability":
        return "Required checkout date/stay duration not specified"
    return None


def prerequisites_met(name, prior, task):
    used = {s["api_name"] for s in prior}
    deps = {"EmailReminder": "QueryMeeting", "TravelStatus": "OrganizationMembers",
            "AddMeeting": "OrganizationMembers", "NearbyRestaurants": "Geocoding",
            "ClothingRecommendation": "GetWeatherForCoordinates",
            "MovieRecommendations": "UserMoviePreferences", "LikeCount": "UserPosts",
            "TaxCalculator": "GetOccupationSalary"}
    if name in deps and deps[name] not in used: return False
    if name == "GetWeatherForCoordinates" and any(s["api_name"] == "Geocoding" for s in task["apis"]):
        return "Geocoding" in used
    if name == "Calculator":
        total = sum(s["api_name"] == "LikeCount" for s in task["apis"])
        return sum(s["api_name"] == "LikeCount" for s in prior) == total
    return True


def prepare(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_level3(str(ROOT / "data/api-bank"))
    states, oracles, audit = [], [], []
    eligible_tasks = set()
    for ti, task in enumerate(tasks):
        env = APIBankMockEnv(task)
        docs = list(env.tool_registry.values())
        tools = [api_schema_to_tool(d) for d in docs]
        prior, messages = [], [{"role": "system", "content": GENERATOR_PROMPT},
                               {"role": "user", "content": "Requirement: " + task["requirement"].strip()}]
        for gi, gold in enumerate(task["apis"]):
            if gold["api_name"] == "ToolSearcher": continue
            state_id = f"l3-{ti:03d}-g{gi:02d}"
            reason = exclusion_reason(ti, gold["api_name"])
            audit.append({"state_id": state_id, "task_id": f"l3-{ti:03d}",
                          "api_name": gold["api_name"], "gold_index": gi,
                          "included": reason is None, "exclusion_reason": reason,
                          "requirement": task["requirement"], "reference_args": gold["input"]})
            if reason is None:
                states.append({"state_id": state_id, "task_id": f"l3-{ti:03d}",
                               "family": ti // 5, "requirement": task["requirement"],
                               "messages": list(messages), "history": list(prior),
                               "tools": tools, "api_documents": docs})
                # Only ready, unconsumed alternatives can receive a positive label.
                choices = []
                for j in range(gi, len(task["apis"])):
                    future = task["apis"][j]
                    if future["api_name"] == "ToolSearcher": continue
                    if exclusion_reason(ti, future["api_name"]): continue
                    if prerequisites_met(future["api_name"], prior, task):
                        choices.append({"gold_index": j, "api_name": future["api_name"],
                                        "arguments": future["input"]})
                oracles.append({"state_id": state_id, "choices": choices,
                                "reference_index": gi, "task_index": ti})
                eligible_tasks.add(ti)
            api = env.tool_registry[gold["api_name"]]
            args = canonical_reference_args(gold["input"], api)
            call_id = f"reference_{gi}"
            messages = messages + [
                {"role": "assistant", "content": "", "tool_calls": [{"id": call_id,
                 "type": "function", "function": {"name": gold["api_name"],
                 "arguments": json.dumps(args, ensure_ascii=False)}}]},
                {"role": "tool", "tool_call_id": call_id,
                 "content": json.dumps(gold["output"].get("output"), ensure_ascii=False)}]
            prior = prior + [{"api_name": gold["api_name"], "arguments": args,
                              "result": gold["output"].get("output")}]
    rng = random.Random(20261015)
    dev, test = [], []
    for family in range(10):
        members = sorted(i for i in eligible_tasks if i // 5 == family)
        rng.shuffle(members)
        test += members[:2]
        dev += members[2:]
    split = {"seed": 20261015, "unit": "task", "dev": [f"l3-{i:03d}" for i in sorted(dev)],
             "test": [f"l3-{i:03d}" for i in sorted(test)],
             "note": "Internal holdout; these benchmark tasks were previously inspected."}
    artifacts = {"states.jsonl": states, "oracle.jsonl": oracles, "label_audit.jsonl": audit}
    for filename, rows in artifacts.items():
        path = out / filename
        if path.exists() and read_jsonl(path) != rows:
            raise ValueError(f"Refusing to replace changed prepared data: {path}")
        write_jsonl(path, rows)
    if (out / "split.json").exists() and json.loads((out / "split.json").read_text()) != split:
        raise ValueError("Refusing to overwrite an existing different task split")
    write_json(out / "split.json", split)
    manifest = {"protocol": VERSION, "total_reference_states": len(audit),
                "included_states": len(states), "excluded_states": len(audit) - len(states),
                "exclusions": dict(Counter(a["exclusion_reason"] for a in audit if not a["included"])),
                "eligible_tasks": len(eligible_tasks), "dev_tasks": len(dev), "test_tasks": len(test),
                "data_sha256": hashlib.sha256((ROOT / "data/api-bank/test-data/level-3.json").read_bytes()).hexdigest(),
                "verifier_prompt_sha256": hashlib.sha256(VERIFIER_PROMPT.encode()).hexdigest()}
    manifest['generator_prompt_sha256'] = hashlib.sha256(GENERATOR_PROMPT.encode()).hexdigest()
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def parse_verifier(content, action):
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    repaired = False
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        # Qwen sometimes emits an anonymous {"confidence": n} inside the root
        # object. Repair only this exact wrapper defect; never infer a value.
        pattern = r',\s*\{\s*"confidence"\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*}\s*}\s*$'
        fixed, count = re.subn(pattern, r',"confidence":\1}', content)
        if count != 1: raise
        obj = json.loads(fixed)
        repaired = True
    score = obj["confidence"]
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid confidence")
    if not isinstance(obj.get("api_suitable"), bool): raise ValueError("Missing api_suitable")
    reports = obj.get("arguments")
    if not isinstance(reports, dict): raise ValueError("Invalid argument reports")
    args = action.get("arguments", {})
    if set(reports) != set(args): raise ValueError("Argument reports differ from candidate keys")
    for report in reports.values():
        if not isinstance(report, dict) or report.get("status") not in ("supported", "contradicted", "unknown"):
            raise ValueError("Invalid argument status")
    supported = sum(r["status"] == "supported" for r in reports.values())
    if repaired: obj['_format_repaired'] = 'removed_anonymous_confidence_wrapper'
    return obj, {"verifier": float(score), "support_ratio": supported / len(args) if args else None}


def request(client, kwargs, calls, kind, guard):
    with REQUEST_LOCK:
        guard.wait_if_hot()
        t0 = time.monotonic()
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            calls.append({"kind": kind, "latency_s": time.monotonic() - t0,
                          "error": type(exc).__name__ + ": " + str(exc)[:500]})
            raise
        finally:
            time.sleep(guard.cooldown)
    calls.append({"kind": kind, "latency_s": time.monotonic() - t0,
                  "usage": response.usage.model_dump() if response.usage else {},
                  "finish_reason": response.choices[0].finish_reason})
    return response


def collect_one(state, client, options, guard, temperature=0.0, sample_index=0):
    sample_id = state["state_id"] + f"-s{sample_index}"
    seed = 20261015 + sample_index * 10000 + int(state["state_id"][3:6]) * 100 + int(state["state_id"][-2:])
    row = {"sample_id": sample_id, "state_id": state["state_id"], "task_id": state["task_id"],
           "family": state["family"], "temperature": temperature, "seed": seed,
           "protocol": VERSION, "calls": [], "features": {}, "errors": []}
    try:
        response = request(client, dict(model="Qwen3-8B", messages=state["messages"], tools=state["tools"],
                           tool_choice="auto", temperature=temperature, seed=seed,
                           max_tokens=options.max_tokens, logprobs=True, top_logprobs=5,
                           extra_body={"chat_template_kwargs": {"enable_thinking": True}}), row["calls"], "generate", guard)
        choice = response.choices[0]
        row["generator_message"] = choice.message.model_dump()
        row["generator_features"] = extract_confidence_features(choice)
        # Keep raw token logprobs for reproducibility and zero-logprob diagnostics.
        row["generator_logprobs"] = choice.logprobs.model_dump() if choice.logprobs else None
        row["features"]["logprob"] = row["generator_features"].get("confidence")
        tool_calls = choice.message.tool_calls or []
        row["tool_call_count"] = len(tool_calls)
        if not tool_calls:
            row["action"] = {"api_name": "__no_tool__", "arguments": {}}
        else:
            tc = tool_calls[0]
            row["action"] = {"api_name": tc.function.name, "arguments": json.loads(tc.function.arguments)}
        if not isinstance(row["action"]["arguments"], dict): raise ValueError("Arguments must be an object")
    except Exception as exc:
        row["errors"].append("generate: " + str(exc)[:500])
        return row
    # This is the ONLY verifier input. It contains no oracle or current execution result.
    payload = {"requirement": state["requirement"], "api_documents": state["api_documents"],
               "history": state["history"], "proposed_call": row["action"]}
    row["verifier_input_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    for attempt in range(2):
        try:
            response = request(client, dict(model="Qwen3-8B", messages=[
                {"role": "system", "content": VERIFIER_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                temperature=0.0, max_tokens=768, seed=seed + attempt,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}),
                row["calls"], "verify", guard)
            row.setdefault("verifier_raw", []).append(response.choices[0].message.content)
            obj, features = parse_verifier(response.choices[0].message.content or "", row["action"])
            row["verifier_output"] = obj
            row["features"].update(features)
            break
        except Exception as exc:
            row["errors"].append("verify: " + str(exc)[:500])
    return row


def collect(options):
    out = Path(options.out_dir)
    states = read_jsonl(out / "states.jsonl")
    amendments=out/'protocol_amendments.json'
    if amendments.exists():
        excluded={sid for item in json.loads(amendments.read_text()) for sid in item.get('state_ids',[])}
        states=[s for s in states if s['state_id'] not in excluded]
    if options.partition != "all":
        split = json.loads((out / "split.json").read_text())
        states = [s for s in states if s["task_id"] in split[options.partition]]
    if options.limit: states = states[:options.limit]
    endpoints = options.endpoints.split(',')
    if any(e not in ("127.0.0.1:18000", "127.0.0.1:18001", "localhost:18000", "localhost:18001") for e in endpoints):
        raise ValueError("This experiment is restricted to managed GPU 0/1 endpoints 18000/18001")
    clients = [OpenAI(base_url=f"http://{ep}/v1", api_key="EMPTY", timeout=240, max_retries=0,
                      http_client=httpx.Client(trust_env=False, timeout=240)) for ep in endpoints]
    target = out / options.output
    old = read_jsonl(target)
    if len({r['sample_id'] for r in old}) != len(old): raise ValueError("Duplicate sample IDs in output")
    completed = {r["sample_id"] for r in old}
    jobs = [(s, i) for s in states for i in range(options.samples)
            if s["state_id"] + f"-s{i}" not in completed]
    guard = ConservativeGuard(out, options.cooldown, options.pause_at, options.resume_at)
    # One sequential queue per endpoint avoids accidentally overloading one GPU.
    queues = [jobs[i::len(clients)] for i in range(len(clients))]
    lock = threading.Lock()
    count = 0
    def worker(client, queue):
        nonlocal count
        for state, i in queue:
            row = collect_one(state, client, options, guard, 0.0 if i == 0 else 0.7, i)
            with lock:
                with target.open('a') as f:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
                    f.flush()
                count += 1
                print(f"{count}/{len(jobs)} {row['sample_id']} api={row.get('action',{}).get('api_name')} "
                      f"logprob={row['features'].get('logprob')} verifier={row['features'].get('verifier')} "
                      f"errors={len(row['errors'])}", flush=True)
    with ThreadPoolExecutor(max_workers=len(clients)) as pool:
        futures = [pool.submit(worker, client, queue) for client, queue in zip(clients, queues)]
        for future in as_completed(futures): future.result()
    print(f"Saved {target}", flush=True)


def label_candidate(state, oracle, row):
    # API docs say relative humidity is a percentage, but the reference passes
    # fractions (0.5, 0.9). Neither convention can be a trusted binary label.
    if any(c['gold_index'] == oracle['reference_index'] and c['api_name'] == 'ClothingRecommendation'
           for c in oracle['choices']):
        return {'y': None, 'reason': 'reference_humidity_unit_conflict'}
    if 'action' not in row:
        return {"y": None, "reason": "generation_error"}
    action = row['action']
    if action['api_name'] == '__no_tool__':
        # A final response/clarification is not an API call. Report separately.
        return {"y": None, "reason": "no_tool_action"}
    if row.get('tool_call_count') != 1:
        return {"y": None, "reason": "multiple_calls_require_joint_evaluation"}
    docs = {d['name']: d for d in state['api_documents']}
    if action['api_name'] not in docs:
        return {"y": 0, "reason": "unknown_api"}
    tool = api_schema_to_tool(docs[action['api_name']])['function']
    if not schema_valid(action['arguments'], tool['parameters']):
        return {"y": 0, "reason": "invalid_schema"}
    env = APIBankMockEnv({"apis": []})
    env.tool_registry = docs
    candidates = [c for c in oracle['choices'] if c['api_name'] == action['api_name']]
    if not candidates:
        return {"y": 0, "reason": "not_ready_or_already_completed"}
    if action['api_name']=='Calculator' and 'total' in state['requirement'].lower() and 'likes' in state['requirement'].lower():
        values=[h['result']['like_count'] for h in state['history'] if h['api_name']=='LikeCount']
        try:
            target=arithmetic_value('+'.join(str(v) for v in values))
            predicted=arithmetic_value(action['arguments'].get('formula'))
            return {'y':int(predicted==target),'reason':'verified_total_from_visible_results',
                    'expected_total':str(target)}
        except (ValueError,SyntaxError,ArithmeticError):
            return {'y':0,'reason':'invalid_arithmetic_formula'}
    for c in candidates:
        if action['api_name'] == 'EmailReminder':
            structural = {k: v for k, v in c['arguments'].items() if k != 'content'}
            proposed = {k: v for k, v in action['arguments'].items() if k != 'content'}
            if all(typed_equal(proposed.get(k), v, 'str') for k,v in structural.items()):
                if not isinstance(action['arguments'].get('content'), str) or not action['arguments']['content'].strip():
                    return {'y': 0, 'reason': 'empty_email_content'}
                return {'y': None, 'reason': 'free_text_review', 'matched_reference': c['gold_index']}
        ok, strict, fields = env._compare_args(action['arguments'], c['arguments'], action['api_name'])
        if ok:
            # Free-form email content gets human review rather than word-overlap truth.
            if action['api_name'] == 'EmailReminder' and not structured_equal(action['arguments'].get('content'), c['arguments'].get('content')):
                return {"y": None, "reason": "free_text_review", "matched_reference": c['gold_index']}
            return {"y": 1, "reason": "semantic_match", "strict": strict, "matched_reference": c['gold_index']}
    return {"y": 0, "reason": "argument_mismatch"}


def label(options):
    out = Path(options.out_dir)
    states = {s['state_id']: s for s in read_jsonl(out/'states.jsonl')}
    oracles = {s['state_id']: s for s in read_jsonl(out/'oracle.jsonl')}
    overrides_path = out/'manual_labels.json'
    overrides = json.loads(overrides_path.read_text()) if overrides_path.exists() else {}
    labeled, review = [], []
    for row in read_jsonl(out/options.output):
        if row.get('action'):
            row['features']['grounding']=literal_grounding(row['action'],states[row['state_id']])
        # Apply the same parser to all cached raw responses. This does not make
        # another model request and never uses offline labels to choose a parse.
        if row.get('action') and row.get('verifier_raw'):
            for raw in row['verifier_raw']:
                try:
                    obj, features = parse_verifier(raw or '', row['action'])
                    row['verifier_output'] = obj
                    row['features'].update(features)
                    if obj.get('_format_repaired'): row['offline_format_repair'] = obj['_format_repaired']
                    break
                except (ValueError,KeyError,TypeError):
                    continue
        decision = label_candidate(states[row['state_id']], oracles[row['state_id']], row)
        if row['sample_id'] in overrides:
            override = overrides[row['sample_id']]
            if override['y'] not in (None,0,1) or not override.get('reason'): raise ValueError('Invalid manual label')
            decision = {**decision, 'y': override['y'], 'manual_review': override['reason']}
            if override['y'] is None:
                decision = {**decision, 'auto_reason': decision['reason'], 'reason': 'manual_ungradable'}
        labeled.append({**row, 'label': decision})
        if decision['y'] is None or decision['y'] == 0:
            # Deliberately omit confidence scores from the human review material.
            review.append({'sample_id': row['sample_id'], 'reason': decision['reason'],
                           'action': row.get('action'), 'requirement': states[row['state_id']]['requirement'],
                           'history': states[row['state_id']]['history'],
                           'choices': oracles[row['state_id']]['choices']})
    write_jsonl(out/options.labeled_output, labeled)
    write_jsonl(out/'review_queue.jsonl', review)
    print(json.dumps({'samples':len(labeled),'labels':dict(Counter(str(r['label']['y']) for r in labeled)),
                      'reasons':dict(Counter(r['label']['reason'] for r in labeled))}, indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','collect','label'])
    parser.add_argument('--out-dir',default=str(ROOT/'outputs/confidence_v1'))
    parser.add_argument('--endpoints',default='127.0.0.1:18000,127.0.0.1:18001')
    parser.add_argument('--partition',choices=['all','dev','test'],default='all')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--max-tokens',type=int,default=2048)
    parser.add_argument('--samples',type=int,default=1)
    parser.add_argument('--cooldown',type=float,default=10.0)
    parser.add_argument('--pause-at',type=float,default=60.0)
    parser.add_argument('--resume-at',type=float,default=50.0)
    parser.add_argument('--output',default='candidates.jsonl')
    parser.add_argument('--labeled-output',default='labeled.jsonl')
    options=parser.parse_args()
    if options.cooldown < 6 or not 35 <= options.resume_at < options.pause_at <= 60:
        parser.error('Conservative limits: cooldown >= 6s, 35 <= resume < pause <= 60C')
    if options.command=='prepare': prepare(options.out_dir)
    elif options.command=='collect': collect(options)
    else: label(options)


if __name__=='__main__': main()
