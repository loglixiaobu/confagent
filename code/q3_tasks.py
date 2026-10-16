"""Bounded synthetic Q3 tasks and a standalone semantic oracle (no model calls).

build_suite() returns 8/16/24 API-Bank-compatible construction/validation/test
tasks. ``_q3`` contains semantic source inputs, never cached expected answers.
oracle_action(task, {api_name, arguments}) checks one action independently of
the replay references and PR1 labels. evaluate_trajectory(task, traj_dict)
checks PR1 action_input/observation steps and a user-facing final_answer;
failed attempts may precede exactly one successful substantive side effect.
make_probe() exposes only the requirement and public protocol/tool schema.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import re

from apibank_env import api_schema_to_tool


PROBE_SYSTEM = ('You complete tool tasks. Call exactly one available tool with grounded '
                'arguments. Use the units required by its documentation.')
_FAMILIES = {
    'radius': ('SetAlertRadius', 'station_name', 'radius_meters', 'meters', 'float'),
    'humidity': ('SetHumidityTarget', 'room_name', 'humidity_percent', 'percent', 'float'),
    'allocation': ('AllocateCache', 'cache_name', 'size_kib', 'KiB', 'int'),
    'reminder': ('ScheduleUTCReminder', 'reminder_name', 'utc_time', 'UTC', 'str'),
}
_NAMES = ('Alder', 'Birch', 'Cedar', 'Dogwood', 'Elm', 'Fir',
          'Ginkgo', 'Hazel', 'Iris', 'Juniper', 'Kauri', 'Larch')
_VALUES = {
    'radius': ('1.375', '860.5', '2.048', '1725', '0.625', '2450.25',
               '3.125', '960.75', '0.875', '3180.5', '4.096', '4525.25'),
    'humidity': ('0.375', '62.5', '0.4825', '71.25', '0.615', '46.75',
                 '0.735', '58.25', '0.0825', '67.5', '0.5675', '81.25'),
    'allocation': ('1.375', '2304', '2.125', '3584', '0.8125', '5376',
                   '3.625', '6912', '4.875', '8448', '6.125', '10496'),
}
_TIMES = ('00:35', '14:10', '23:40', '06:25', '00:20', '18:50',
          '01:15', '07:45', '00:05', '20:30', '23:55', '09:40')
_OFFSETS = ('+05:30', '+00:00', '-03:30', '+00:00', '+09:45', '+00:00',
            '-07:00', '+00:00', '+12:45', '+00:00', '-02:30', '+00:00')


def _doc(family):
    api, entity, field, unit, kind = _FAMILIES[family]
    rules = {
        'radius': 'Save an alert radius in meters. 1 kilometer = 1000 meters.',
        'humidity': 'Save relative humidity in percent (0 to 100). A fraction of 1 = 100 percent.',
        'allocation': 'Allocate cache capacity in integer KiB. 1 MiB = 1024 KiB.',
        'reminder': 'Schedule one reminder at a UTC ISO 8601 timestamp, with Z or +00:00. '
                    'Convert the supplied fixed UTC offset; daylight-saving rules do not apply.',
    }
    description = f'Required saved value in {unit}.'
    if family != 'reminder':
        description += ' A JSON number is preferred; exact decimal numeric strings are accepted.'
    return {'name': api, 'description': rules[family], 'input_parameters': {
        entity: {'type': 'str', 'description': 'Exact name supplied by the user.'},
        field: {'type': kind, 'description': description}},
        'output_parameters': {'success': {'type': 'bool', 'description':
            'True after saving; the response also echoes all saved input fields.'}}}


def build_suite():
    """Build deterministic, disjoint tasks; replay references are for the mock only."""
    suite = {}
    for split, indices in [('construction', range(2)), ('validation', range(2, 6)),
                           ('test', range(6, 12))]:
        tasks = suite[split] = []
        for family, (api, entity_key, value_key, unit, _) in _FAMILIES.items():
            for i in indices:
                conversion = i % 2 == 0
                variant = 'conversion' if conversion else 'already_target'
                entity = f'{_NAMES[i]} {family}'
                meta = {'id': f'q3-{split}-{family}-{i:02d}', 'split': split,
                        'family': family, 'variant': variant, 'entity': entity}
                if family == 'reminder':
                    day = (datetime(2027, 1, 4) + timedelta(days=19 * i)).date()
                    local = f'{day.isoformat()}T{_TIMES[i]}:00'
                    offset = _OFFSETS[i]
                    meta.update(local_datetime=local, utc_offset=offset)
                    stamp = datetime.fromisoformat(local + offset).astimezone(timezone.utc)
                    result = stamp.isoformat().replace('+00:00', 'Z')
                    when = (f'{local} local time with fixed UTC offset {offset}' if conversion
                            else f'{local} UTC (offset +00:00)')
                    requirement = (f'Schedule a reminder named "{entity}" at {when}. '
                                   'After scheduling, give its UTC ISO date and time.')
                else:
                    source = _VALUES[family][i]
                    source_unit = ({'radius': 'kilometers', 'humidity': 'fraction',
                                    'allocation': 'MiB'}[family] if conversion else unit)
                    meta.update(source_value=source, source_unit=source_unit)
                    factor = {'radius': 1000, 'humidity': 100, 'allocation': 1024}[family]
                    number = Decimal(source) * (factor if conversion else 1)
                    result = int(number) if number == number.to_integral_value() else float(number)
                    if family == 'radius':
                        request = f'Set the alert radius for station "{entity}" to {source} {source_unit}.'
                    elif family == 'humidity':
                        amount = (f'a fraction of {source} (1.0 means 100 percent)' if conversion
                                  else f'{source} percent')
                        request = f'Set the relative humidity target for room "{entity}" to {amount}.'
                    else:
                        request = f'Allocate {source} {source_unit} for the cache named "{entity}".'
                    requirement = f'{request} After saving, state the saved value in {unit}.'
                args = {entity_key: entity, value_key: result}
                tasks.append({'requirement': requirement, 'response': f'Saved {result} {unit}.',
                    'apis': [{'api_name': 'ToolSearcher', 'input': {'keywords': api},
                              'output': {'output': _doc(family), 'exception': None}},
                             {'api_name': api, 'input': args,
                              'output': {'output': {'success': True, **args}, 'exception': None}}],
                    '_q3': meta})
    return suite


def _decimal(value):
    if type(value) not in (int, float, str):
        raise ValueError('Expected a JSON number or exact numeric string')
    value = Decimal(str(value).strip())
    if not value.is_finite():
        raise ValueError('Nonfinite number')
    return value


def _utc(value):
    if not isinstance(value, str):
        raise ValueError('UTC timestamp must be a string')
    stamp = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    if stamp.utcoffset() != timedelta(0):
        raise ValueError('An explicit UTC timestamp is required')
    return stamp


def _expected(task):
    """Recompute from semantic inputs, never from apis, response, or PR1 checks."""
    meta = task['_q3']
    family = meta['family']
    api, name_key, value_key, _, _ = _FAMILIES[family]
    if family == 'reminder':
        local = datetime.fromisoformat(meta['local_datetime'])
        sign = -1 if meta['utc_offset'][0] == '-' else 1
        hours, minutes = map(int, meta['utc_offset'][1:].split(':'))
        result = (local - sign * timedelta(hours=hours, minutes=minutes)).replace(tzinfo=timezone.utc)
    else:
        factors = {'kilometers': Decimal(1000), 'meters': Decimal(1),
                   'fraction': Decimal(100), 'percent': Decimal(1),
                   'MiB': Decimal(1024), 'KiB': Decimal(1)}
        result = _decimal(meta['source_value']) * factors[meta['source_unit']]
    return api, {name_key: meta['entity'], value_key: result}


def oracle_action(task, action):
    """Reject missing/extra fields, wrong names/units/types, searches, and no-action."""
    if not isinstance(action, dict):
        return False
    try:
        api, expected = _expected(task)
        args = action.get('arguments')
        if action.get('api_name') != api or not isinstance(args, dict) or args.keys() != expected.keys():
            return False
        for field, value in expected.items():
            actual = args[field]
            if isinstance(value, Decimal):
                actual = _decimal(actual)
                if task['_q3']['family'] == 'allocation' and actual != actual.to_integral_value():
                    return False
            elif isinstance(value, datetime):
                actual = _utc(actual)
            elif not isinstance(actual, str):
                return False
            if actual != value:
                return False
        return True
    except (KeyError, ValueError, TypeError, InvalidOperation, OverflowError):
        return False


def _final_answer(task, trajectory):
    raw = trajectory.get('final_answer')
    if not isinstance(raw, str):
        return False
    text = re.sub(r'<think\b[^>]*>.*?</think\s*>', '', raw, flags=re.S | re.I).strip()
    if not text or re.search(r'</?think\b', text, re.I):
        return False
    records = [trajectory, trajectory.get('config') or {}]
    records += [s for s in trajectory.get('steps', []) if s.get('action_type') == 'answer']
    if any(r.get(k) in ('length', 'max_tokens', 'token_limit') for r in records
           for k in ('finish_reason', 'final_finish_reason', 'terminal_finish_reason', 'stop_reason')):
        return False
    _, expected = _expected(task)
    value = list(expected.values())[1]
    if isinstance(value, datetime):
        stamps = re.findall(r'\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?'
                            r'\s*(?:Z|\+00:00|UTC)\b', text)
        try:
            return bool(stamps) and all(_utc(s.replace('UTC', '+00:00').replace(' +', '+')) == value
                                        for s in stamps)
        except ValueError:
            return False
    units = {'radius': r'(?:meters?|metres?|m)', 'humidity': r'(?:%|percent(?:age)?)',
             'allocation': r'KiB'}[task['_q3']['family']]
    number = r'[+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?'
    quantities = re.findall(r'(?<![\w.])(' + number + r')\s*' + units + r'(?!\w)',
                            text, flags=re.I)
    # A unit or the exact saved field also grounds a following value, e.g.
    # "saved value in meters is 1375" or "size_kib: 2304".
    field = re.escape(list(expected)[1])
    quantities += re.findall(r'(?<!\w)(?:' + units + '|' + field + r')(?!\w)[`\"\'*]*\s*'
                             r'(?:(?:is|are|was|were)(?:\s+(?:now|exactly))?|:|=)\s*[`\"\'*]*('
                             + number + r')(?!\w|\.\d)', text, flags=re.I)
    return bool(quantities) and all(_decimal(q.replace(',', '')) == value for q in quantities)


def evaluate_trajectory(task, trajectory):
    """Count substantive argument errors, retain them after recovery, check actual effects."""
    actions, effects = [], []
    for step in trajectory.get('steps', []):
        if step.get('action_type') == 'answer':
            continue
        action = step.get('action_input') or {}
        if not isinstance(action, dict):
            action = {}
        if action.get('api_name') == 'ToolSearcher':
            continue
        correct = oracle_action(task, action)
        actions.append(correct)
        obs = step.get('observation')
        succeeded = (isinstance(obs, dict) and obs.get('success') is True
                     and 'error' not in obs and not obs.get('exception') and not step.get('exception'))
        if succeeded:
            echoed = {k: v for k, v in obs.items() if k != 'success'}
            effects.append(correct and oracle_action(task, {'api_name': action.get('api_name'),
                                                           'arguments': echoed}))
    final_valid = _final_answer(task, trajectory)
    strict = len(effects) == 1 and effects[0] and final_valid
    errors = len(actions) - sum(actions)
    return {'strict_success': bool(strict), 'action_correct': sum(actions), 'action_errors': errors,
            'final_answer_valid': final_valid, 'successful_side_effects': len(effects),
            'clean_success': bool(strict and not errors), 'recovered_success': bool(strict and errors)}


def public_task(task):
    """Copy an API-Bank task without private oracle metadata; replay references remain."""
    return deepcopy({k: v for k, v in task.items() if k != '_q3'})


def make_probe(task):
    """Build an actor-visible probe: no reference arguments, results, or labels."""
    meta = task['_q3']
    doc = _doc(meta['family'])
    return {'task_id': meta['id'], 'split': meta['split'], 'family': meta['family'],
            'variant': meta['variant'], 'messages': [
                {'role': 'system', 'content': PROBE_SYSTEM},
                {'role': 'user', 'content': task['requirement']}],
            'tools': [api_schema_to_tool(doc)], 'api_names': [doc['name']]}
