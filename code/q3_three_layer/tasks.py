"""Frozen numeric smoke tasks, actual local effects, and an independent oracle.

build_suite() supplies 18 construction, 12 development and three disjoint six-task
held-out sets. TaskWorld.prepare() returns raw-form actor messages and a public
contract; chain tasks execute ReadMeasurement before exposing its observation.
execute(prediction) records writes/answers, including wrong numeric writes.
evaluate() judges final state independently; it never supplies a controller input.
computational_verify(contract) is a public tool, separate from independent_correct.
"""
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import json
import re


_NUMBER = r'-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?'
_UNITS = ('km', 'm', 'fraction', 'percent', 'MiB', 'KiB')
_PRIMITIVES = {'radius': ('km', 'm', '1 km = 1000 m.'),
               'humidity': ('fraction', 'percent', '1 fraction = 100 percent.'),
               'allocation': ('MiB', 'KiB', '1 MiB = 1024 KiB.')}
_TOOL_FACTORS = {('km', 'm'): '1000', ('m', 'm'): '1',
                 ('fraction', 'percent'): '100', ('percent', 'percent'): '1',
                 ('MiB', 'KiB'): '1024', ('KiB', 'KiB'): '1'}
_NAMES = ('Alder', 'Birch', 'Cedar', 'Dogwood', 'Elm', 'Fir', 'Ginkgo', 'Hazel',
          'Iris', 'Juniper', 'Kauri', 'Larch', 'Maple', 'Nettle', 'Olive', 'Pine')
_SYSTEM = 'Return only JSON with keys "value" (number or numeric string) and "unit" (string).'


def _numeric(value):
    if type(value) not in (str, int, float, Decimal):
        raise ValueError('A finite number is required')
    text = str(value)
    if not re.fullmatch(_NUMBER, text):
        raise ValueError('Malformed numeric value')
    number = Decimal(text)
    if not number.is_finite():
        raise ValueError('Nonfinite number')
    return number


def _canonical(number):
    if not number:
        return '0'
    if abs(number.adjusted()) > 30:
        return str(number)
    text = format(number, 'f')
    return text.rstrip('0').rstrip('.') if '.' in text else text


def _prediction(value):
    if not isinstance(value, dict) or set(value) != {'value', 'unit'}:
        raise ValueError('Prediction requires exactly value and unit')
    if not isinstance(value['unit'], str) or not value['unit'].strip() or len(value['unit']) > 32:
        raise ValueError('A unit string is required')
    return {'value': _canonical(_numeric(value['value'])), 'unit': value['unit'].strip()}


def parse_prediction(text):
    """Parse one complete finite JSON value/unit object; no fences, prose or duplicates."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON field')
            result[key] = value
        return result
    try:
        if not isinstance(text, str) or len(text) > 4096:
            return None
        value = json.loads(text, parse_float=Decimal, parse_int=Decimal, object_pairs_hook=unique)
        return _prediction(value)
    except (ValueError, TypeError, InvalidOperation, OverflowError):
        return None


def _write_schema(target, definition):
    return {'name': 'WriteSetting', 'description': definition,
            'parameters': {'value': {'type': 'number'},
                           'unit': {'type': 'string', 'description': f'Target unit: {target}'}}}


def build_suite():
    """Build balanced conversion/identity tasks with no repeated semantic instance."""
    suite = {}
    groups = [('construction', 0, 6, 'single_write'), ('development', 6, 10, 'single_write'),
              ('source_test', 10, 12, 'single_write'), ('chain_test', 12, 14, 'read_write'),
              ('qa_test', 14, 16, 'text_qa')]
    for split, start, end, shape in groups:
        tasks = suite[split] = []
        for primitive, (source, target, definition) in _PRIMITIVES.items():
            for index in range(start, end):
                conversion = index % 2 == 0
                source_unit = source if conversion else target
                group = Decimal(index // 2)
                if primitive == 'radius':
                    value = (Decimal('1.375') + group * Decimal('0.247') if conversion
                             else Decimal('860.5') + group * Decimal('73.25'))
                elif primitive == 'humidity':
                    value = (Decimal('0.275') + group * Decimal('0.0375') if conversion
                             else Decimal('43.25') + group * Decimal('4.125'))
                else:
                    value = (Decimal('1.375') + group * Decimal('0.3125') if conversion
                             else Decimal('2304') + group * Decimal('768'))
                entity = f'{_NAMES[index]} {primitive}'
                value = _canonical(value)
                if shape == 'single_write':
                    requirement = (f'For "{entity}", save the reading {value} {source_unit} '
                                   'with WriteSetting in its required target unit.')
                elif shape == 'read_write':
                    requirement = (f'Fetch the measurement for "{entity}" with ReadMeasurement, '
                                   'then save it with WriteSetting in its required target unit.')
                else:
                    requirement = (f'Report for "{entity}":\n| Reading | Value | Unit |\n'
                                   f'| measured | {value} | {source_unit} |\n'
                                   f'Express the reading in {target}. {definition}')
                task = {'task_id': f'three-{split}-{primitive}-{index:02d}', 'split': split,
                        'shape': shape, 'primitive': primitive,
                        'variant': 'conversion' if conversion else 'identity',
                        'requirement': requirement,
                        '_spec': {'entity': entity, 'source_value': value,
                                  'source_unit': source_unit, 'target_unit': target}}
                if shape != 'text_qa':
                    task['write_schema'] = _write_schema(target, definition)
                if shape == 'read_write':
                    task['read_schema'] = {'name': 'ReadMeasurement', 'parameters': {'entity': 'string'}}
                tasks.append(task)
    return suite


def computational_verify(contract):
    """Execute the public conversion tool from current public source values only."""
    source, target = contract['source_unit'], contract['target_unit']
    factor = _TOOL_FACTORS.get((source, target))
    if factor is None:
        raise ValueError('Unsupported public unit contract')
    value = _canonical(_numeric(contract['source_value']))
    result = _canonical(_numeric(value) * Decimal(factor))
    return {'prediction': {'value': result, 'unit': target}, 'trace': {
        'tool': 'ConvertUnits', 'source_value': value, 'source_unit': source,
        'target_unit': target, 'multiplier': factor, 'result': result,
        'executed': True, 'success': True}}


def independent_correct(task, prediction):
    """Independent unit arithmetic from the private semantic fixture; no tool table."""
    try:
        prediction = _prediction(prediction)
        spec = task['_spec']
        quantity = Decimal(spec['source_value'])
        source, target = spec['source_unit'], spec['target_unit']
        if source == target and target in ('m', 'percent', 'KiB'):
            expected = quantity
        elif source == 'km' and target == 'm':
            expected = quantity * (Decimal(10) ** 3)
        elif source == 'fraction' and target == 'percent':
            expected = quantity.scaleb(2)
        elif source == 'MiB' and target == 'KiB':
            expected = quantity * (Decimal(2) ** 10)
        else:
            return False
        return prediction['unit'] == target and Decimal(prediction['value']) == expected
    except (KeyError, ValueError, TypeError, InvalidOperation, OverflowError):
        return False


def _public_contract(requirement, shape, schema=None, observation=None):
    """Extract current public evidence without consulting the private oracle fixture."""
    entity = re.search(r'"([^\"]+)"', requirement).group(1)
    if shape == 'read_write':
        value, source = observation['value'], observation['unit']
    elif shape == 'text_qa':
        match = re.search(r'\| measured \| (' + _NUMBER + r') \| (\w+) \|', requirement)
        value, source = match.groups()
    else:
        match = re.search(r'\breading (' + _NUMBER + r') (' + '|'.join(_UNITS) + r')\b', requirement)
        value, source = match.groups()
    target = (re.search(r'Express the reading in (\w+)\.', requirement).group(1)
              if shape == 'text_qa' else schema['parameters']['unit']['description'].split(': ', 1)[1])
    return {'source_value': _canonical(_numeric(value)), 'source_unit': source,
            'target_unit': target, 'entity': entity}


class TaskWorld:
    """Local read/write state; wrong well-typed writes succeed and remain in history."""
    def __init__(self, task):
        self.task = deepcopy(task)
        self.history = []
        self.saved = None
        self.answer = None
        self._prepared = None
        spec = task['_spec']
        self._measurements = {spec['entity']: {'value': spec['source_value'], 'unit': spec['source_unit']}}

    def read_measurement(self, entity):
        """Execute the local read tool and retain its actual returned observation."""
        observation = {'entity': entity, **deepcopy(self._measurements[entity])}
        self.history.append({'kind': 'read', 'tool': 'ReadMeasurement', 'executed': True,
                             'arguments': {'entity': entity}, 'result': deepcopy(observation)})
        return observation

    def prepare(self):
        if self._prepared is not None:
            return deepcopy(self._prepared)
        task = self.task
        raw = task['requirement']
        if task['shape'] != 'text_qa':
            raw += '\nTool schema: ' + json.dumps(task['write_schema'], separators=(',', ':'))
        messages = [{'role': 'system', 'content': _SYSTEM}, {'role': 'user', 'content': raw}]
        observation = None
        if task['shape'] == 'read_write':
            entity = re.search(r'"([^\"]+)"', task['requirement']).group(1)
            observation = self.read_measurement(entity)
            messages.extend([{'role': 'assistant', 'content': '', 'tool_calls': [{
                'id': 'measurement', 'type': 'function', 'function': {'name': 'ReadMeasurement',
                'arguments': json.dumps({'entity': entity})}}]},
                {'role': 'tool', 'tool_call_id': 'measurement', 'name': 'ReadMeasurement',
                 'content': json.dumps(observation, separators=(',', ':'))}])
        contract = _public_contract(task['requirement'], task['shape'], task.get('write_schema'), observation)
        self._prepared = {'messages': messages, 'contract': contract}
        return deepcopy(self._prepared)

    def execute(self, prediction):
        """Commit a numeric result; None/malformed predictions cause no side effect."""
        self.prepare()
        try:
            chosen = _prediction(prediction)
        except (ValueError, TypeError, InvalidOperation, OverflowError):
            self.history.append({'kind': 'invalid_prediction', 'executed': False,
                                 'prediction': deepcopy(prediction)})
            return {'success': False, 'error': 'invalid_prediction'}
        qa = self.task['shape'] == 'text_qa'
        result = {'success': True, **chosen}
        event = {'kind': 'answer' if qa else 'write', 'tool': None if qa else 'WriteSetting',
                 'executed': True, 'prediction': chosen, 'result': deepcopy(result)}
        if not qa:
            event['arguments'] = {'entity': self._prepared['contract']['entity'], **chosen}
        self.history.append(event)
        if qa:
            self.answer = deepcopy(chosen)
        else:
            self.saved = deepcopy(chosen)
        return result

    def evaluate(self):
        """Judge final actual state; earlier wrong writes stay visible after recovery."""
        writes = [event for event in self.history if event['kind'] == 'write']
        predictions = [event['prediction'] for event in self.history
                       if event['kind'] in ('write', 'answer')]
        wrong = sum(not independent_correct(self.task, event['prediction']) for event in writes)
        final = self.answer if self.task['shape'] == 'text_qa' else self.saved
        success = independent_correct(self.task, final)
        return {'success': success, 'action_correct': sum(independent_correct(self.task, p) for p in predictions),
                'wrong_writes': wrong, 'overwrites': max(0, len(writes) - 1),
                'clean_success': success and wrong == 0,
                'recovered_success': success and wrong > 0,
                'environment_calls': sum(event.get('tool') in ('ReadMeasurement', 'WriteSetting')
                                         and event['executed'] for event in self.history)}
