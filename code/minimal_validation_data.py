"""Prepare additional API-Bank tasks and a finite recorded-response mock.

Task/reference data stay outside model requests. This adapter does not contact
real services. Labels target grounded tool actions, separately from mock success.
"""
from __future__ import annotations

import ast
import copy
import datetime as dt
import hashlib
import json
import math
import random
import re
from pathlib import Path

from apibank_env import (APIBankMockEnv, _norm_time, api_schema_to_tool,
                         numeric_equal, parameter_schema, typed_equal)

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261019
NUMBER_WORDS = dict(zip(
    'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split(),
    range(21)))
UNORDERED_FIELDS = {'plan_details', 'medicine_list', 'treatment_options', 'centers', 'medication_names'}


def time_value(value):
    """Normalize a clock value only, without turning dates into times."""
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r'\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM)?\s*', value, re.I)
    if not match:
        return None
    hour, minute, second = int(match[1]), int(match[2]), int(match[3] or 0)
    if minute >= 60 or second >= 60 or hour >= 24:
        return None
    if match[4]:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if match[4].upper() == 'PM' else 0)
    return hour, minute, second


def schema_valid(value, schema):
    """Validate generated JSON values before any reference equivalence check."""
    kind = schema.get('type')
    if kind == 'object':
        props = schema.get('properties', {})
        return (isinstance(value, dict) and set(value) == set(schema.get('required', props))
                and all(k in props and schema_valid(v, props[k]) for k, v in value.items()))
    if kind == 'array':
        return isinstance(value, list) and all(schema_valid(v, schema.get('items', {})) for v in value)
    if kind in ('number', 'integer'):
        return (type(value) in (int, float) and math.isfinite(value)
                and (kind != 'integer' or int(value) == value))
    if kind == 'string':
        return isinstance(value, str)
    if kind == 'boolean':
        return type(value) is bool
    return not schema


def bmi_equal(left, right):
    try:
        return (not isinstance(left, bool) and not isinstance(right, bool)
                and math.isfinite(float(left)) and math.isfinite(float(right))
                and math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=.005))
    except (ValueError, TypeError, OverflowError):
        return False


def argument_equal(field, left, right, metadata):
    if field == 'bmi' and bmi_equal(left, right):
        return True
    if field == 'time' or field.endswith(('_time', '_date')):
        clock = time_value(left)
        if clock is not None and clock == time_value(right):
            return True
    if field in UNORDERED_FIELDS and isinstance(left, list) and isinstance(right, list):
        canon = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False)
        if sorted(map(canon, left)) == sorted(map(canon, right)):
            return True
    return typed_equal(left, right, metadata.get('type', 'str'))


def visible_observations(history):
    return [h['observation'] for h in history if not (
        isinstance(h.get('observation'), dict) and 'error' in h['observation'])]


def computed_bmi_supported(value, requirement, history):
    def visit(obj):
        if isinstance(obj, dict):
            if 'bmi' in obj and bmi_equal(value, obj['bmi']):
                return True
            return any(visit(v) for v in obj.values())
        return isinstance(obj, list) and any(visit(v) for v in obj)
    if any(visit(observation) for observation in visible_observations(history)):
        return True
    weight = re.search(r'(\d+(?:\.\d+)?)\s*(?:kg|kilograms?)\b', requirement, re.I)
    height = re.search(r'(\d+(?:\.\d+)?)\s*(?:meters?|metres?)\b', requirement, re.I)
    if weight and height and float(height[1]) > 0:
        return bmi_equal(value, float(weight[1]) / float(height[1]) ** 2)
    return False


def derived_reminder_supported(value, requirement, history):
    """A default lead time is not invented from an unsupported reference."""
    interval = re.search(r'\b(\d+|' + '|'.join(NUMBER_WORDS) + r')\s+minutes?\s+(?:before|earlier)', requirement, re.I)
    if not interval:
        return False
    token = interval[1].lower()
    minutes = NUMBER_WORDS[token] if token in NUMBER_WORDS else int(token)
    try:
        proposed = dt.datetime.fromisoformat(str(_norm_time(value)))
    except (ValueError, TypeError):
        return False
    for observation in visible_observations(history):
        if not isinstance(observation, dict) or 'appointment_time' not in observation:
            continue
        try:
            appointment = dt.datetime.fromisoformat(str(_norm_time(observation['appointment_time'])))
            if proposed == appointment - dt.timedelta(minutes=minutes):
                return True
        except (ValueError, TypeError):
            continue
    return False


def read_lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write_lines(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_response_record(record, index):
    users = re.findall(r'^User: (.*)$', record['input'], re.M)
    lines = re.findall(r'^API-Request: \[(.*?)\]->(.*)$', record['input'], re.M)
    if len(users) != 1 or len(lines) < 2:
        raise ValueError('Need one requirement and at least two recorded calls')
    calls, docs = [], {}
    for expression, result in lines:
        node = ast.parse(expression, mode='eval').body
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.args:
            raise ValueError('Unsupported API request syntax')
        if any(k.arg is None for k in node.keywords):
            raise ValueError('No unpacked arguments')
        args = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
        observation = ast.literal_eval(result)
        name = node.func.id
        if name == 'ToolSearcher' and isinstance(observation, dict) and observation.get('name'):
            doc = copy.deepcopy(observation)
            if not isinstance(doc.get('input_parameters'), dict):
                raise ValueError('Incomplete tool document')
            for field in doc['input_parameters'].values():
                if str(field.get('type', '')).lower() in ('date', 'datetime', 'time'):
                    field['type'] = 'str'
                parameter_schema(str(field.get('type', 'str')))
            docs[doc['name']] = doc
            observation = doc
        calls.append({'api_name': name, 'input': args, 'output': {
            'api_name': name, 'input': args, 'output': observation, 'exception': None}})
    business = [c for c in calls if c['api_name'] != 'ToolSearcher']
    if len(business) < 2 or not docs or any(c['api_name'] not in docs for c in business):
        raise ValueError('Need at least two documented business calls')
    if any(set(c['input']) != set(docs[c['api_name']]['input_parameters']) for c in business):
        raise ValueError('Recorded arguments differ from documented required fields')
    return {'task_id': f'fresh-l3-{index:03d}', 'source_record_index': index,
            'requirement': users[0].strip(), 'response': record.get('output', ''),
            'apis': calls, 'api_documents': list(docs.values())}


def prepare(out_dir, max_tasks=30):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'data/api-bank/training-data/lv3-response-train.json'
    records = json.loads(source.read_text())
    old = json.loads((ROOT / 'data/api-bank/test-data/level-3.json').read_text())
    normalize = lambda s: ' '.join(s.lower().split())
    old_requirements = {normalize(r['requirement']) for r in old}
    seen, eligible, excluded = set(), [], {}
    for i, record in enumerate(records):
        try:
            task = parse_response_record(record, i)
            signature = normalize(task['requirement'])
            if signature in old_requirements or signature in seen:
                raise ValueError('Duplicate or previously used requirement')
            seen.add(signature)
            eligible.append(task)
        except (ValueError, SyntaxError, KeyError, TypeError) as exc:
            reason = re.sub(r'0x[0-9a-fA-F]+', '<address>', str(exc).split('\n')[0])[:120]
            excluded[reason] = excluded.get(reason, 0) + 1
    random.Random(SEED).shuffle(eligible)
    tasks = eligible[:max_tasks]
    protocol = {'version': 'confidence-minimal-validation-v1', 'source_file': str(source.relative_to(ROOT)),
        'source_sha256': file_sha(source), 'source_records': len(records),
        'eligible_tasks': len(eligible), 'selected_task_ids': [t['task_id'] for t in tasks],
        'selection_seed': SEED, 'selection': 'parseable single-user tasks with complete schemas; shuffled before model calls',
        'old_level3_requirement_overlap': 0, 'initial_tasks': 20, 'maximum_tasks': max_tasks,
        'extension_rule': 'After initial 20, append remaining preselected tasks up to 30 if fewer than 10 call errors; never select by AUROC',
        'source_type_normalization': 'date/datetime/time input fields represented as JSON strings',
        'primary_policy': {'model': 'Qwen3-8B', 'thinking': True, 'temperature': 0, 'max_tokens': 2048, 'max_steps': 8},
        'probe_policy': {'model': 'Qwen3-8B', 'thinking': False, 'temperature': .7, 'k': 3, 'max_tokens': 4096},
        'environment': 'finite recorded-response mock; schemas, numeric storage and repeat read handling normalized; no real API effects',
        'mock_limit': 'An otherwise plausible unrecorded parameter combination can be rejected. Mock rejection is not the label authority.',
        'labels': 'binary, score-blind reference/visible-evidence review; current return not supplied to pre-call confidence',
        'task_outcomes': 'not used as call labels; no claim of task-success improvement',
        'excluded_record_reasons': excluded}
    frozen = out / 'validation_protocol.json'
    if frozen.exists() and json.loads(frozen.read_text()) != protocol:
        raise ValueError('Validation protocol changed; use a fresh output directory')
    frozen.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    target = out / 'validation_tasks.jsonl'
    content = ''.join(json.dumps(t, ensure_ascii=False) + '\n' for t in tasks)
    if target.exists() and target.read_text() != content:
        raise ValueError('Prepared validation tasks changed')
    target.write_text(content)
    return tasks, protocol


def scalar_supported(value, requirement, history):
    """Visible evidence excludes earlier proposed arguments and future results."""
    observations = visible_observations(history)
    text = json.dumps({'requirement': requirement, 'observations': observations}, ensure_ascii=False).lower()
    if isinstance(value, dict):
        return all(scalar_supported(v, requirement, history) for v in value.values())
    if isinstance(value, list):
        return all(scalar_supported(v, requirement, history) for v in value)
    if type(value) in (int, float):
        tokens = re.findall(r'(?<![\w.])[+-]?\d+(?:\.\d+)?(?![\w.])', text)
        tokens += [str(NUMBER_WORDS[word]) for word in re.findall(r'\b[a-z]+\b', text) if word in NUMBER_WORDS]
        return any(numeric_equal(value, token) for token in tokens)
    if isinstance(value, str):
        value = value.strip().lower()
        if not value:
            return False
        if value in text:
            return True
        clock = time_value(value)
        if clock is not None:
            for candidate in re.findall(r'(?<![\d:])\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?(?![\d:])', text):
                if time_value(candidate) == clock:
                    return True
        # Numeric reference storage strings are evidence values too. IDs remain
        # checked as text here: numeric coercion belongs to declared fields.
        for match in re.findall(r'\d{4}-\d{1,2}-\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?', text):
            if _norm_time(value) == _norm_time(match):
                return True
        return False
    return value is None or isinstance(value, bool)


class RecordedTask:
    def __init__(self, task):
        self.task = task
        self.env = APIBankMockEnv(task)
        self.documents = {d['name']: d for d in task['api_documents']}

    def tools(self):
        from collect_new_signals import SEARCH_TOOL
        return [copy.deepcopy(SEARCH_TOOL)] + [api_schema_to_tool(d) for d in self.documents.values()]

    def matching_references(self, action):
        name, args = action['api_name'], action['arguments']
        fields = self.documents.get(name, {}).get('input_parameters', {})
        return [r for r in self.task['apis'] if r['api_name'] == name and
                set(args) == set(r['input']) and (
                    all(argument_equal(k, args[k], r['input'][k], fields.get(k, {})) for k in args)
                    or self.env._compare_args(args, r['input'], name)[0])]

    def search(self, args):
        words = re.findall(r'[a-z]+', str(args.get('keywords', '')).lower())
        ranked = []
        for doc in self.documents.values():
            text = (doc['name'] + ' ' + doc.get('description', '')).lower()
            score = sum(len(word) for word in words if len(word) > 2 and word in text)
            ranked.append((score, doc))
        score, doc = max(ranked, key=lambda pair: pair[0], default=(0, None))
        return copy.deepcopy(doc) if score else {'error': 'No documented tool matched the query.'}

    def call(self, action):
        if action['api_name'] == 'ToolSearcher':
            return self.search(action['arguments'])
        matches = self.matching_references(action)
        if not matches:
            return {'error': 'No recorded mock response is available for this call.'}
        return copy.deepcopy(matches[0]['output']['output'])

    def label(self, action, history):
        name, args = action['api_name'], action['arguments']
        if not isinstance(args, dict):
            return {'y': 0, 'reason': 'invalid_argument_schema', 'unsupported_fields': []}
        if name == 'ToolSearcher':
            result = self.search(args)
            return {'y': int('error' not in result), 'reason': 'useful_tool_document' if 'error' not in result else 'search_miss',
                    'unsupported_fields': []}
        if name not in self.documents:
            return {'y': 0, 'reason': 'unknown_tool', 'unsupported_fields': []}
        fields = self.documents[name]['input_parameters']
        if set(args) != set(fields):
            return {'y': 0, 'reason': 'required_argument_mismatch', 'unsupported_fields': sorted(set(args) ^ set(fields))}
        schema = api_schema_to_tool(self.documents[name])['function']['parameters']
        if not schema_valid(args, schema):
            return {'y': 0, 'reason': 'invalid_argument_schema', 'unsupported_fields': [
                k for k, value in args.items() if not schema_valid(value, schema['properties'][k])]}
        matches = self.matching_references(action)
        unsupported = []
        for field, value in args.items():
            if scalar_supported(value, self.task['requirement'], history):
                continue
            if field == 'bmi' and computed_bmi_supported(value, self.task['requirement'], history):
                continue
            if field == 'reminder_time' and derived_reminder_supported(value, self.task['requirement'], history):
                continue
            # Long generated explanatory prose is reviewed via the reference
            # semantic matcher, rather than demanded verbatim from the user.
            if isinstance(value, str) and len(value.split()) >= 5 and matches:
                continue
            unsupported.append(field)
        if unsupported:
            return {'y': 0, 'reason': 'arguments_without_visible_evidence', 'unsupported_fields': unsupported}
        if not matches:
            description = (name + ' ' + self.documents[name].get('description', '')).lower()
            read_only = (re.search(r'retriev|find|search|locat|check|calculat|evaluat|compar|recommend|information|details|description|review', description)
                         and not re.search(r'register|schedul|enroll|reminder|sharing|update|secure access|booking', description))
            if read_only:
                return {'y': 1, 'reason': 'grounded_read_only_call_unrecorded_reference_review', 'unsupported_fields': []}
            return {'y': 0, 'reason': 'reference_argument_mismatch_review', 'unsupported_fields': []}
        return {'y': 1, 'reason': 'reference_and_visible_evidence_supported', 'unsupported_fields': []}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, default=ROOT / 'outputs/confidence_minimal_v1')
    args = p.parse_args()
    tasks, protocol = prepare(args.out_dir)
    print(json.dumps({'tasks': len(tasks), 'eligible': protocol['eligible_tasks'],
                      'source_overlap': protocol['old_level3_requirement_overlap']}, ensure_ascii=False))
