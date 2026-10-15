#!/usr/bin/env python
"""Binary labels for all original calls, recorded no-tool responses, and tasks.

Offline only. Never assign a failed task's label to its earlier correct calls.
Never invent termination logprobs or treat a runner stop as a model decision.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

from apibank_env import APIBankMockEnv, api_schema_to_tool, load_level3, numeric_equal
from offline_confidence_utils import (ROOT, adjudicate, dump, dump_lines, fact,
                                      observed_success, semantic_signature, sha, schema_valid)

VERSION = 'legacy-binary-v2'
POLICY = {
    'labels': {'0': '错误', '1': '正确'},
    'label_authority': 'User explicitly requested binary correctness, including fabricated missing inputs and incomplete tasks. Codex source/context review; not independent human double annotation.',
    'call_target': 'Tool choice, grounded arguments, required dependencies and documented units. Numeric storage and equivalent arithmetic artifacts remain repaired. Correct retries stay locally correct.',
    'missing_input': 'Executing with invented required dates, places or other task values is incorrect. Do not reject hidden reference values and then accept equally unsupported model values.',
    'meeting': 'Invented meeting location is an error independently of the old attendees schema bug. Record environment schema defect as a cause, never as a third label.',
    'hotel': 'No stated stay duration or checkout: committing a checkout value without grounding is incorrect, including old gold-matching calls.',
    'travel_task_35': 'All business calls invent missing travel values and are incorrect. A relevant document search can still be correct.',
    'salary': 'Pass-through salary is locally valid. An annual/monthly transformation without source evidence is incorrect. For the task endpoint, the requirement supplies the requested monthly frame; no annual frame is silently assumed.',
    'coordinates': 'Changing the sign of a recorded coordinate without a recorded verification is incorrect under the evidence-grounding target. This is not a claim about world geography.',
    'humidity': 'Passing fractional humidity unchanged to a percentage API without resolving units is incorrect. A final task can still recover if its recorded result/answer meets the goal.',
    'search': 'Judge all four original searches by recorded query/document capability. No search calls excluded.',
    'completion': 'Task label is goal completion at the actual episode end. Missing required work or required derived output is incorrect; prior errors do not permanently disqualify a recovered task.',
    'actual_results': 'A semantically valid call rejected by the faulty simulator stays call-correct, but cannot count as an actual successful side effect or returned result. Failure cause includes environment rejection.',
    'goal_rules': {
        'meeting_reminders': 'Successful meeting lookup and every reminder specified in that task reference chain, grounded in actual returned meetings. Do not add hidden AccountInfo-like prerequisites.',
        'weather_clothing': 'Actual weather and clothing returns plus a grounded final weather/clothing answer; validate the observed 15C Sunny example against its returned clothing options.',
        'book_meeting': 'Members, all relevant travel statuses, grounded meeting fields and an actually successful reservation.',
        'restaurants': 'Actual geocoding and an actual correct restaurant-search return. Empty results are a valid query outcome.',
        'movies': 'Preference/watched/recommendation results and a recorded list filtering out watched movies. Raw recommendations containing watched titles do not complete the explicit filter goal.',
        'likes': 'All actual post counts plus an actual correct calculator result or a recorded correct final total. Offline reviewer calculation alone does not complete the task.',
        'weather_city': 'Actual geocoding and an actual correct weather return.',
        'travel': 'Grounded flight and hotel query returns, including a grounded checkout duration.',
        'salary': 'Actual occupation salary and actual correctly grounded net-salary return in the requirement frame.',
        'update': 'Actual successful grounded PersonalInfoUpdate. AccountInfo is optional.'},
    'end_records': 'Each of 50 tasks has a binary endpoint outcome. Only 18 original no-tool model responses are additional model events; 32 runner-imposed endings are not fabricated model stop actions.',
    'score_missing': 'The original log does not retain no-tool response logprobs. Their labels remain binary; their score is missing, never zero or the previous call score.',
    'metric_population': 'All 198 calls in primary AUROC; all labels are binary. Additional 18 no-tool events retained without scores; no claim that call AUROC evaluates stopping quality.',
    'source_preservation': 'Original results and v1 audit are immutable. Annotate fixed recorded behavior, do not infer a new rollout or performance improvement.',
}

SEARCH_REVIEWS = {
    'l3-010:1': (0, 'search_no_result', '搜索 location 没有返回工具文档，未满足检索目标。'),
    'l3-030:3': (1, 'relevant_weather_document', '搜索 weather city 返回天气查询文档，能够说明现有天气工具的参数。'),
    'l3-035:3': (1, 'relevant_flight_document', '搜索 flight search parameters 返回 FlightSearch 文档；检索本身没有编造出行参数。'),
    'l3-042:5': (0, 'retrieval_capability_mismatch', '搜索 net salary calculation 却返回已使用的职业工资查询工具，没有提供扣税计算能力。'),
}


def binary_result(y, code, reason, evidence=(), **extra):
    if type(y) is not int or y not in (0, 1): raise ValueError('A binary label is required')
    return {'y': y, 'status': '正确' if y else '错误', 'reason_code': code,
            'reason': reason, 'evidence': list(evidence), 'version': VERSION, **extra}


def label_call(ti, task, step, history, task_id):
    action = step['action_input']
    name = action['api_name']
    previous = adjudicate(ti, task, action, history, step['step_id'], task_id)
    if previous['y'] is not None:
        previous = copy.deepcopy(previous)
        previous['version'] = VERSION
        return previous
    ev = previous['evidence'] + [{'source': 'candidate_call', 'value': action}]
    code = previous['reason_code']
    if name == 'ToolSearcher':
        y, reason_code, reason = SEARCH_REVIEWS[f"{task_id}:{step['step_id']}"]
        return binary_result(y, reason_code, reason,
                             ev + [{'source': 'recorded_search_return', 'value': step['observation']}],
                             error_type=None if y else 'tool_retrieval_failure')
    if code == 'missing_travel_inputs':
        return binary_result(0, 'fabricated_travel_inputs', '用户没有提供出发地、目的地和日期，调用却擅自填入；属于无依据参数。', ev,
                             error_type='unsupported_arguments', unsupported_fields=list(action['arguments']))
    if code == 'meeting_schema_and_location_conflict':
        return binary_result(0, 'fabricated_meeting_location', '没有用户或此前工具提供会议地点，模型自行填入会议室。旧 schema 的类型错误另外记录，不掩盖地点问题。', ev,
                             error_type='unsupported_arguments', unsupported_fields=['location'],
                             environment_issues=['attendees_advertised_as_string_instead_of_array'])
    if code == 'missing_checkout_or_stay_duration':
        return binary_result(0, 'fabricated_checkout_date', '没有入住时长或退房日期依据，模型自行确定 checkout，包括碰巧与隐藏参考一致的情况。', ev,
                             error_type='unsupported_arguments', unsupported_fields=['check_out_date'])
    if code == 'salary_period_unspecified':
        return binary_result(0, 'unsupported_salary_period_conversion', '返回工资未声明为年薪，没有依据直接除以 12 改变计算周期。', ev,
                             error_type='unsupported_conversion', unsupported_fields=['salary'])
    if code == 'geocoding_sign_conflict':
        return binary_result(0, 'unverified_coordinate_sign_change', '未记录独立核验，就修改此前工具返回的经度符号；按参数依据规则判错，不对真实地理坐标下结论。', ev,
                             error_type='unsupported_value_change', unsupported_fields=['longitude'])
    if code == 'humidity_unit_conflict':
        return binary_result(0, 'humidity_unit_not_resolved', '文档要求百分比，未核实或转换就把历史中的 0.7 原样作为百分比传入。', ev,
                             error_type='unverified_units', unsupported_fields=['humidity'],
                             environment_issues=['weather_and_reference_humidity_unit_inconsistency'])
    raise ValueError(f'Unreviewed binary case: {task_id}:{step["step_id"]} {code}')


def final_body(raw):
    raw = raw or ''
    if '<think>' in raw and '</think>' not in raw:
        return '', True
    return re.sub(r'<think>.*?</think>', '', raw, flags=re.S).strip(), False


def successful(calls, name, require_correct=True):
    return [r for r in calls if r['action']['api_name'] == name
            and observed_success(r['original_step']['observation'])
            and (not require_correct or r['revision']['y'] == 1)]


def goal(name, passed, records=(), detail=''):
    return {'goal': name, 'passed': bool(passed), 'evidence_steps': [r['step_id'] for r in records], 'detail': detail}


def task_outcome(ti, task, trajectory, calls):
    """Goal checks use actual recorded results; do not replace them with gold."""
    family = ti // 5
    text, unclosed = final_body(trajectory.get('final_answer'))
    checks = []
    get = lambda name, correct=True: successful(calls, name, correct)
    if family == 0:
        lookups, mails = get('QueryMeeting'), get('EmailReminder')
        checks.append(goal('查询会议', bool(lookups), lookups))
        for ref in [s for s in task['apis'] if s['api_name'] == 'EmailReminder']:
            args = ref['input']
            matched = [r for r in mails if all(r['action']['arguments'].get(k) == args[k] for k in ('recipient', 'time', 'location'))]
            checks.append(goal(f"发送提醒：{args['time']} / {args['location']}", bool(matched), matched))
    elif family == 1:
        weather, clothing = get('GetWeatherForCoordinates'), get('ClothingRecommendation', False)
        checks += [goal('取得实际天气返回', bool(weather), weather), goal('取得穿衣建议返回', bool(clothing), clothing)]
        # Task 8 has the sole observed clothing return. Its 15C/Sunny/70%
        # result and final answer are correct even though the argument unit
        # handling was faulty. Do not require an error-free trajectory.
        options = clothing[-1]['original_step']['observation'].get('clothing_options', []) if clothing else []
        supported_answer = bool(weather and clothing and text and options and all(v.lower() in text.lower() for v in options))
        checks.append(goal('给出有返回依据的天气与穿衣答案', supported_answer, clothing,
                           '对原记录中 15°C/Sunny 样例，返回衣物与最终回答一致；不因之前有错永久判失败。'))
    elif family == 2:
        members, statuses, reservations = get('OrganizationMembers'), get('TravelStatus'), get('AddMeeting')
        checks.append(goal('取得组织成员', bool(members), members))
        names = members[-1]['original_step']['observation']['members'] if members else []
        queried = {r['action']['arguments']['member_name'] for r in statuses}
        checks.append(goal('核查全部成员出差状态', bool(names) and set(names) <= queried, statuses))
        checks.append(goal('以有依据的参数实际预约成功', bool(reservations), reservations))
    elif family == 3:
        geos, restaurants = get('Geocoding'), get('NearbyRestaurants')
        checks += [goal('取得地点坐标', bool(geos), geos), goal('取得正确餐厅查询返回', bool(restaurants), restaurants)]
    elif family == 4:
        prefs, watched, recs = get('UserMoviePreferences'), get('UserWatchedMovies'), get('MovieRecommendations')
        checks += [goal('取得偏好', bool(prefs), prefs), goal('取得已看列表', bool(watched), watched), goal('取得推荐候选', bool(recs), recs)]
        remaining = []
        if watched and recs:
            seen = set(watched[-1]['original_step']['observation'])
            remaining = [v for v in recs[-1]['original_step']['observation']['recommended_movies'] if v not in seen]
        # No filtered final list is recorded in the four old 'success' cases.
        # Do not count a list constructed by this offline reviewer as agent work.
        filtered = bool(text and remaining and all(v.lower() in text.lower() for v in remaining))
        checks.append(goal('记录排除已看电影后的列表', filtered, recs,
                           '原运行器在推荐 API 后提前按 success 停止；没有记录完成过滤的输出。'))
    elif family == 5:
        posts, counts, calculators = get('UserPosts'), get('LikeCount'), get('Calculator')
        checks.append(goal('取得用户帖子列表', bool(posts), posts))
        ids = posts[-1]['original_step']['observation']['post_ids'] if posts else []
        by_id = {r['action']['arguments']['post_id']: r for r in counts}
        complete = bool(ids) and set(ids) <= set(by_id)
        checks.append(goal('取得全部帖子点赞数', complete, counts))
        total = sum(Decimal(str(by_id[pid]['original_step']['observation']['like_count'])) for pid in ids) if complete else None
        actual_calc = bool(total is not None and any(numeric_equal(r['original_step']['observation'], total) for r in calculators))
        answer_values = re.findall(r'(?<![\w.])\d+(?:\.\d+)?(?![\w.])', text)
        answer_matches = bool(total is not None and any(numeric_equal(v, total) for v in answer_values))
        checks.append(goal('产生正确总数的实际返回或最终答案', actual_calc or answer_matches, calculators,
                           f'实际已查齐总数={total}；实际计算返回={actual_calc}；最终答案给出总数={answer_matches}。'))
    elif family == 6:
        geos, weather = get('Geocoding'), get('GetWeatherForCoordinates')
        checks += [goal('取得城市坐标', bool(geos), geos), goal('取得正确天气返回', bool(weather), weather)]
    elif family == 7:
        flights, hotels = get('FlightSearch'), get('HotelAvailability')
        checks += [goal('取得有输入依据的航班查询返回', bool(flights), flights),
                   goal('以有依据的入住退房时间取得酒店查询返回', bool(hotels), hotels)]
    elif family == 8:
        salaries, taxes = get('GetOccupationSalary'), get('TaxCalculator')
        checks += [goal('取得职业工资', bool(salaries), salaries),
                   goal('取得需求周期下有依据的税后工资返回', bool(taxes), taxes,
                        '以需求的月薪框架解释未声明其他周期的返回金额；不自行引入年薪假设。')]
    elif family == 9:
        updates = get('PersonalInfoUpdate')
        confirmed = [r for r in updates if isinstance(r['original_step']['observation'], dict)
                     and r['original_step']['observation'].get('status') == 'success']
        checks.append(goal('实际更新成功', bool(confirmed), confirmed, '不要求用户未请求的 AccountInfo 调用。'))
    else:
        raise ValueError('Unreviewed task family')
    y = int(bool(checks) and all(c['passed'] for c in checks))
    stop = trajectory['config']['stop_reason']
    causes = []
    if not y:
        if any(r['revision']['y'] == 0 for r in calls): causes.append('incorrect_recorded_calls')
        if any(r['revision']['y'] == 1 and not observed_success(r['original_step']['observation']) for r in calls):
            causes.append('simulator_rejected_semantically_valid_call')
        if stop == 'success': causes.append('runner_success_check_omitted_required_goal')
        if stop in ('repeated_call', 'max_steps'): causes.append('runner_limit_or_repetition_stop')
        if unclosed: causes.append('no_final_answer_unclosed_thinking')
        if not calls: causes.append('no_tool_work_recorded')
        if stop == 'final_answer' and not unclosed: causes.append('model_response_before_all_goals_completed')
    return {'task_id': trajectory['task_id'], 'y': y, 'status': '正确' if y else '错误',
            'old_task_success': bool(trajectory['final_success']), 'stop_reason': stop,
            'end_origin': 'model_no_tool_response' if stop == 'final_answer' else 'runner',
            'goal_checks': checks, 'missing_goals': [c['goal'] for c in checks if not c['passed']],
            'failure_causes': causes, 'final_answer_text': text, 'unclosed_thinking': unclosed,
            'call_errors': sum(r['revision']['y'] == 0 for r in calls),
            'version': VERSION}


def annotate(source, out, data_dir=None):
    source, out = Path(source), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    trajectories = [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    data_dir = Path(data_dir) if data_dir is not None else ROOT/'data/api-bank'
    tasks = load_level3(str(data_dir))
    protocol = {'version': VERSION, 'policies': POLICY, 'search_reviews': SEARCH_REVIEWS,
                'source_file': source.resolve().relative_to(ROOT).as_posix(), 'source_sha256': sha(source),
                'task_data_sha256': sha(data_dir/'test-data/level-3.json'),
                'labeler_sha256': sha(__file__), 'offline_utils_sha256': sha(ROOT/'code/offline_confidence_utils.py'),
                'comparison_code_sha256': sha(ROOT/'code/apibank_env.py'),
                'decisions_frozen_before_scoring': True}
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError('Protocol differs; use a new versioned output directory')
    dump(out/'protocol.json', protocol)
    calls, ends, outcomes, annotated, review = [], [], [], [], []
    seen_ids = set()
    for row_index, trajectory in enumerate(trajectories):
        tid = trajectory['task_id']
        ti = int(tid.split('-')[-1])
        task = tasks[ti]
        docs = APIBankMockEnv(task).tool_registry
        revised = copy.deepcopy(trajectory)
        history, signatures, local = [], set(), []
        had_error = False
        for i, step in enumerate(trajectory['steps']):
            event_id = f"{tid}:call-{step['step_id']:02d}"
            if event_id in seen_ids: raise ValueError('Duplicate call')
            seen_ids.add(event_id)
            decision = label_call(ti, task, step, history, tid)
            assert decision['y'] in (0, 1)
            action = step['action_input']
            signature = semantic_signature(action, history, docs)
            row = {'sample_id': event_id, 'event_type': 'tool_call', 'task_id': tid,
                   'step_id': step['step_id'], 'family': ti//5, 'action': copy.deepcopy(action),
                   'original_label': step['matched_gold'], 'revision': decision,
                   'original_step': copy.deepcopy(step), 'source_row_index': row_index,
                   'source_step_index': i, 'scope': 'auxiliary_search' if action['api_name']=='ToolSearcher' else 'business_call',
                   'first_semantic_attempt': signature not in signatures, 'before_first_mock_error': not had_error,
                   'canonical_schema_valid': schema_valid(action['arguments'], api_schema_to_tool(docs[action['api_name']])['function']['parameters']) if action['api_name'] in docs else True,
                   'score_available': True}
            local.append(row); calls.append(row); signatures.add(signature)
            revised['steps'][i]['binary_revision'] = decision
            review.append({'event_id': event_id, 'event_type': 'tool_call', 'requirement': task['requirement'],
                           'api_document': docs.get(action['api_name']), 'action': action,
                           'prior_history': copy.deepcopy(history), 'revision': decision})
            if not observed_success(step['observation']): had_error = True
            history.append({'step_id': step['step_id'], 'action': action, 'observation': step['observation']})
        endpoint = task_outcome(ti, task, trajectory, local)
        outcomes.append(endpoint)
        revised['binary_task_outcome'] = endpoint
        if endpoint['end_origin'] == 'model_no_tool_response':
            terminal = {'sample_id': f'{tid}:no-tool-response', 'event_type': 'model_no_tool_response',
                        'task_id': tid, 'family': ti//5, 'y': endpoint['y'], 'status': endpoint['status'],
                        'missing_goals': endpoint['missing_goals'], 'failure_causes': endpoint['failure_causes'],
                        'final_answer_text': endpoint['final_answer_text'], 'unclosed_thinking': endpoint['unclosed_thinking'],
                        'score_available': False, 'score_missing_reason': 'Original runner did not save logprobs for responses without tool calls.',
                        'label_meaning': '任务在该次无工具输出结束时是否完成；不是把合理澄清也声称为错误操作。', 'version': VERSION}
            ends.append(terminal)
            review.append({'event_id': terminal['sample_id'], 'event_type': terminal['event_type'],
                           'requirement': task['requirement'], 'prior_history': history,
                           'final_answer_text': endpoint['final_answer_text'], 'unclosed_thinking': endpoint['unclosed_thinking'],
                           'y': endpoint['y'], 'missing_goals': endpoint['missing_goals'], 'goal_checks': endpoint['goal_checks']})
        annotated.append(revised)
    dump_lines(out/'calls_binary.jsonl', calls)
    dump_lines(out/'end_responses_binary.jsonl', ends)
    dump_lines(out/'events_binary.jsonl', calls+ends)
    dump_lines(out/'task_outcomes_binary.jsonl', outcomes)
    dump_lines(out/'trajectories_binary.jsonl', annotated)
    dump_lines(out/'score_blind_review.jsonl', review)
    fieldnames = ['event_id', 'event_type', 'task_id', 'step_id', 'api', 'arguments', 'old_label',
                  'y', 'status', 'reason', 'missing_goals', 'score_available']
    with (out/'events_binary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames); writer.writeheader()
        for r in calls:
            writer.writerow({'event_id': r['sample_id'], 'event_type': 'tool_call', 'task_id': r['task_id'],
                             'step_id': r['step_id'], 'api': r['action']['api_name'],
                             'arguments': json.dumps(r['action']['arguments'], ensure_ascii=False),
                             'old_label': int(r['original_label']), 'y': r['revision']['y'], 'status': r['revision']['status'],
                             'reason': r['revision']['reason'], 'score_available': True})
        for r in ends:
            writer.writerow({'event_id': r['sample_id'], 'event_type': r['event_type'], 'task_id': r['task_id'],
                             'y': r['y'], 'status': r['status'], 'reason': '; '.join(r['failure_causes']) if not r['y'] else '任务目标已完成',
                             'missing_goals': '; '.join(r['missing_goals']), 'score_available': False})
    with (out/'task_outcomes_binary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        fields = ['task_id','old_task_success','y','status','stop_reason','end_origin','missing_goals','failure_causes']
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for row in outcomes:
            writer.writerow({**{k:row[k] for k in fields},'missing_goals':'; '.join(row['missing_goals']),
                             'failure_causes':'; '.join(row['failure_causes'])})
    errors = [r for r in calls if not r['revision']['y']]
    summary = {'version': VERSION, 'calls': len(calls), 'call_labels': dict(Counter(r['revision']['status'] for r in calls)),
               'error_types': dict(Counter(r['revision'].get('error_type',r['revision']['reason_code']) for r in errors)),
               'error_reasons': dict(Counter(r['revision']['reason_code'] for r in errors)),
               'call_label_transitions': dict(Counter(f"{int(r['original_label'])}->{r['revision']['y']}" for r in calls)),
               'recorded_no_tool_responses': len(ends), 'end_response_labels': dict(Counter(r['status'] for r in ends)),
               'end_response_scores_recorded': 0, 'labeled_model_events': len(calls)+len(ends),
               'tasks': len(outcomes), 'task_labels': dict(Counter(r['status'] for r in outcomes)),
               'old_task_success_count': sum(r['old_task_success'] for r in outcomes),
               'task_transitions': dict(Counter(f"{int(r['old_task_success'])}->{r['y']}" for r in outcomes)),
               'runner_endings_not_model_stop_events': sum(r['end_origin']=='runner' for r in outcomes),
               'source_sha256': protocol['source_sha256'], 'protocol_sha256': sha(out/'protocol.json'),
               'artifact_sha256': {p:sha(out/p) for p in ('calls_binary.jsonl','end_responses_binary.jsonl','task_outcomes_binary.jsonl')},
               'gpu_used': False, 'model_requests': 0}
    if sha(source) != protocol['source_sha256']: raise ValueError('Source changed')
    dump(out/'label_summary.json', summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default=str(ROOT/'results/traj_main_clean.jsonl'))
    parser.add_argument('--out-dir',default=str(ROOT/'outputs/legacy_binary_v2'))
    parser.add_argument('--data-dir', default=str(ROOT/'data/api-bank'), help='API-Bank data directory containing test-data/level-3.json')
    args=parser.parse_args()
    annotate(args.source,args.out_dir,args.data_dir)


if __name__=='__main__': main()
