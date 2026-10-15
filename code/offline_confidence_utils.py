"""Portable CPU-only rules, provenance helpers, and logprob metrics.

Internal review proposals are resolved by relabel_binary_steps into binary
labels before any artifact is written. No model clients or GPU imports.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
from decimal import Decimal
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from apibank_env import APIBankMockEnv, _norm_time, arithmetic_value, numeric_equal, typed_equal
ROOT = Path(__file__).resolve().parents[1]
SIGNALS = ['geo_mean_prob', 'p10_prob', 'min_prob', 'tail_geo_mean_prob', 'frac_highprob']
MAIN = 'geo_mean_prob'

EMAIL_REVIEWS = {
    'l3-000:2': 'Body accurately identifies Meeting with the client, its time and Room 1; recipient matches requirement. Does not assert that all reminders were sent.',
    'l3-001:2': 'Body Reminder: Meeting with the client is brief but factual; structured time/location and recipient identify the actual meeting.',
    'l3-002:2': 'Body Reminder: Meeting with the client is factual; structured time/location and recipient identify the actual meeting.',
    'l3-003:2': 'Body names Meeting, Room 1 and 2021-01-01 10:00:00, all supported by the earlier meeting response.',
    'l3-004:2': 'Body names Meeting, its time and Room 1, all supported by the earlier meeting response.',
}

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

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def dump_lines(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def observed_success(obs):
    return obs is not None and not (isinstance(obs, dict) and 'error' in obs)

def fact(history, api):
    return next((h for h in reversed(history)
                 if h['action']['api_name'] == api and observed_success(h['observation'])), None)

def evidence(source, value):
    return {'source': source, 'value': value}

def result(y, code, explanation, ev=(), **extra):
    return {'y': y, 'status': {1: '正确', 0: '错误', None: '无法确定'}[y],
            'reason_code': code, 'reason': explanation, 'evidence': list(ev), **extra}

def normalized(value, type_name=''):
    if type_name in ('float', 'int', 'number', 'integer'):
        if isinstance(value, bool): return value
        try:
            number = Decimal(str(value))
            if number.is_finite(): return {'numeric': str(number.normalize())}
        except (ArithmeticError, ValueError): pass
    if isinstance(value, str): return _norm_time(value)
    if isinstance(value, list): return [normalized(v) for v in value]
    return value

def semantic_signature(action, history, docs):
    """Deduplicate retries without merging identical formulas in changed states.

    Labels/scores do not enter the key. Calculator evidence retains observed
    posts/counts: a premature calculation and its later supported retry differ.
    """
    name, args = action['api_name'], action['arguments']
    props = docs.get(name, {}).get('input_parameters', {})
    values = {k: normalized(v, props.get(k, {}).get('type', '')) for k, v in args.items()}
    if name == 'PersonalInfoUpdate' and isinstance(values.get('address'), str):
        values['address'] = values['address'].strip().rstrip('.')
    if name == 'Calculator':
        try: values['formula'] = {'arithmetic_result': str(arithmetic_value(args['formula']).normalize())}
        except (ArithmeticError, ValueError, SyntaxError): pass
    dependency_apis = {
        'Calculator': ('UserPosts', 'LikeCount'), 'NearbyRestaurants': ('Geocoding',),
        'GetWeatherForCoordinates': ('Geocoding',), 'TaxCalculator': ('GetOccupationSalary',),
        'EmailReminder': ('QueryMeeting',), 'MovieRecommendations': ('UserMoviePreferences',),
        'LikeCount': ('UserPosts',), 'TravelStatus': ('OrganizationMembers',),
        'AddMeeting': ('OrganizationMembers', 'TravelStatus'),
        'ClothingRecommendation': ('GetWeatherForCoordinates',),
    }.get(name, ())
    facts = {json.dumps({'action': h['action'], 'result': h['observation']}, sort_keys=True)
             for h in history if h['action']['api_name'] in dependency_apis and observed_success(h['observation'])}
    return json.dumps({'api': name, 'arguments': values, 'dependencies': sorted(facts)}, sort_keys=True)

def equal_fields(args, expected, doc):
    props = doc.get('input_parameters', {})
    return set(args) == set(expected) and all(
        typed_equal(args[k], value, props.get(k, {}).get('type', 'str')) for k, value in expected.items())

def adjudicate(task_index, task, action, history, step_id, task_id):
    """No confidence, old label, current mock rejection or future history input."""
    name, args = action['api_name'], action['arguments']
    docs = APIBankMockEnv(task).tool_registry
    refs = [s for s in task['apis'] if s['api_name'] == name]
    req = evidence('requirement', task['requirement'].strip())
    if name == 'ToolSearcher':
        return result(None, 'auxiliary_search_no_semantic_oracle',
                      '辅助检索单列：旧 mock 的 matched_gold 不能判定检索意图是否满足。', [req])
    if task_index == 35:
        return result(None, 'missing_travel_inputs', '需求缺少出发地、目的地和日期，不能按隐藏参考值判定调用对错。', [req])
    if name == 'AddMeeting':
        return result(None, 'meeting_schema_and_location_conflict',
                      '旧工具 schema 将 attendees 错误声明为 string，原 API 为 list(str)；需求也没有会议地点，不能归因成干净的模型参数错误。',
                      [req, evidence('original_tool_schema.attendees', {'type': 'string'}),
                       evidence('api_document.attendees', docs[name]['input_parameters']['attendees'])],
                      issues=['original_schema_conflict', 'missing_meeting_location'])
    if name == 'HotelAvailability':
        return result(None, 'missing_checkout_or_stay_duration',
                      '用户只给出行日期，未给入住时长或退房日期；参考的一晚住宿没有公开依据。', [req])
    if name == 'ClothingRecommendation':
        return result(None, 'humidity_unit_conflict',
                      '文档要求百分比湿度，历史/参考却使用 0–1 小数；现有材料不足以形成可信二分类标签。',
                      [evidence('api_document.humidity', docs[name]['input_parameters']['humidity']),
                       evidence('prior_weather', fact(history, 'GetWeatherForCoordinates'))])
    if name not in docs:
        return result(0, 'unknown_api', '工具不在本任务公开工具集合中。', [req], error_type='wrong_api')

    if name == 'Calculator':
        posts = fact(history, 'UserPosts')
        if posts is None:
            return result(0, 'missing_post_lookup', '尚未取得用户帖子列表即开始总点赞数计算。', [req], error_type='missing_prerequisite')
        post_ids = posts['observation']['post_ids']
        counts = {}
        for h in history:
            if h['action']['api_name'] == 'LikeCount' and observed_success(h['observation']):
                pid = h['action']['arguments']['post_id']
                if pid in post_ids: counts[pid] = h
        missing = [pid for pid in post_ids if pid not in counts]
        offline_total = sum(s['output']['output']['like_count'] for s in task['apis'] if s['api_name'] == 'LikeCount')
        try: value = arithmetic_value(args.get('formula'))
        except (ArithmeticError, ValueError, SyntaxError):
            return result(0, 'invalid_formula', '公式不符合受限四则运算语法或不能安全计算。', [req], error_type='invalid_arithmetic')
        answer_only_y = int(value == Decimal(str(offline_total)))
        if missing:
            return result(0, 'calculation_before_all_counts_observed',
                          '尚未获取全部帖子点赞数，公式使用了当时未观察到的数值；遵循原系统提示“参数来自需求或此前返回”的过程正确性口径。',
                          [evidence(f"step:{posts['step_id']}.UserPosts", posts['observation']),
                           evidence('observed_like_counts', {str(k): h['observation']['like_count'] for k, h in counts.items()}),
                           evidence('missing_post_ids', missing)], error_type='missing_prerequisite',
                          answer_only_y=answer_only_y,
                          answer_only_note='仅看事后真实总数，该公式结果正确；主口径要求当时有充分依据。')
        total = sum(Decimal(str(counts[pid]['observation']['like_count'])) for pid in post_ids)
        ok = value == total
        reference_formulas = [s['input']['formula'] for s in refs]
        reference_conflict = any(arithmetic_value(f) != total for f in reference_formulas)
        return result(int(ok), 'verified_sum_from_prior_returns' if ok else 'wrong_total',
                      '用此前实际返回的全部点赞数独立求和；等价表达式均接受，不受错误参考公式影响。' if ok else '公式计算结果与此前实际返回的点赞数总和不符。',
                      [evidence(f"step:{h['step_id']}.LikeCount", {'post_id': pid, **h['observation']}) for pid, h in counts.items()]
                      + [evidence('independent_expected_sum', str(total)), evidence('candidate_result', str(value))],
                      answer_only_y=answer_only_y, reference_formula_conflict=reference_conflict,
                      error_type=None if ok else 'wrong_arithmetic_result')

    if name in ('GetWeatherForCoordinates', 'NearbyRestaurants'):
        coords = fact(history, 'Geocoding')
        expected = refs[0]['input'] if refs else {}
        source = evidence('requirement/reference_numeric_fields', expected)
        if coords:
            expected = {**expected, **coords['observation']}
            source = evidence(f"step:{coords['step_id']}.Geocoding", coords['observation'])
        if name == 'NearbyRestaurants':
            if coords is None:
                return result(0, 'missing_geocoding', '尚未取得该地点坐标。', [req], error_type='missing_prerequisite')
            km = re.search(r'within\s+(\d+(?:\.\d+)?)\s*km', task['requirement'], re.I)
            if km is None: return result(None, 'unreviewed_distance_requirement', '距离需求无法按固定规则解析。', [req])
            meters = Decimal(km[1]) * 1000
            if not numeric_equal(args.get('distance'), meters, integer=True):
                return result(0, 'kilometers_not_converted_to_meters',
                              '需求为 1 km，API distance 单位是米，应传 1000，而实际传入 1。',
                              [req, evidence('api_document.distance', docs[name]['input_parameters']['distance']),
                               evidence('required_distance_meters', str(meters)), evidence('candidate_distance', args.get('distance'))],
                              error_type='distance_unit_conversion')
        if all(numeric_equal(args.get(k), expected.get(k)) for k in ('latitude', 'longitude')):
            return result(1, 'coordinates_and_units_supported', '坐标与需求或此前返回数值一致；数值字符串及尾零差异不构成语义错误。', [req, source])
        if coords and numeric_equal(args.get('latitude'), expected.get('latitude')) and numeric_equal(args.get('longitude'), -Decimal(str(expected['longitude']))):
            return result(None, 'geocoding_sign_conflict',
                          '调用改变了 Geocoding 返回的经度符号；该基准坐标与地理语义存在争议，不能仅用参考值认定真实语义错误。单列“违背可见返回”事实。',
                          [req, source, evidence('candidate_coordinates', args)], visible_evidence_contradiction=True)
        return result(0, 'coordinate_value_mismatch', '坐标既不匹配需求也不匹配此前返回，且不属于已知符号争议。', [req, source], error_type='wrong_coordinates')

    if name == 'TaxCalculator':
        salary = fact(history, 'GetOccupationSalary')
        if salary is None:
            return result(0, 'missing_salary_lookup', '尚未获得职业工资。', [req], error_type='missing_prerequisite')
        amount = salary['observation']['salary']
        ev = [req, evidence(f"step:{salary['step_id']}.GetOccupationSalary", salary['observation']),
              evidence('api_document.salary', docs[name]['input_parameters']['salary'])]
        if numeric_equal(args.get('salary'), amount):
            return result(1, 'returned_salary_passed_through', '将实际返回工资原值传入扣税 API；数值表示差异不构成语义错误，最终月薪回答另行评估。', ev)
        return result(None, 'salary_period_unspecified',
                      '候选将返回工资除以 12；需求要求月薪，但 API 未声明返回工资或扣税周期。不能仅因不同于参考原值就认定其转换错误。', ev,
                      candidate_salary=args.get('salary'), ratio_to_returned=float(Decimal(str(args['salary'])) / Decimal(str(amount))))

    if name == 'EmailReminder':
        meetings = fact(history, 'QueryMeeting')
        if meetings is None:
            return result(0, 'missing_meeting_lookup', '尚未取得会议信息。', [req], error_type='missing_prerequisite')
        address = re.search(r'[\w.+-]+@[\w.-]+\.[A-Za-z]+', task['requirement'])
        matched = [m for m in meetings['observation']['meetings']
                   if _norm_time(args.get('time')) == _norm_time(m['meeting_time'])
                   and args.get('location') == m['meeting_location']]
        if not address or args.get('recipient') != address[0] or not matched:
            return result(0, 'wrong_email_schedule_or_recipient', '收件人或会议时间地点不符合需求与此前返回。', [req], error_type='wrong_email_fields')
        key = f'{task_id}:{step_id}'
        review = EMAIL_REVIEWS.get(key)
        if review is None: return result(None, 'unreviewed_email_text', '邮件自由文本尚未完成无分数语义审查。', [req])
        return result(1, 'email_facts_reviewed', '结构化字段匹配实际会议；自由文本事实经无分数上下文审查一致。',
                      [req, evidence(f"step:{meetings['step_id']}.QueryMeeting", matched),
                       evidence('score_blind_content_review', review)])

    if name == 'PersonalInfoUpdate':
        for reference in refs:
            expected = reference['input']
            if (args.get('username') == expected['username'] and args.get('password') == expected['password']
                    and isinstance(args.get('address'), str)
                    and args['address'].strip().rstrip('.') == expected['address'].strip().rstrip('.')):
                return result(1, 'address_update_supported', '用户名、密码和地址由用户明确给出；允许地址末尾句号，不要求多余的 AccountInfo 前置调用。', [req])
        return result(0, 'wrong_account_or_address', '账户或地址字段与需求不符。', [req], error_type='wrong_update_fields')

    dependency = {'TravelStatus': 'OrganizationMembers', 'LikeCount': 'UserPosts',
                  'MovieRecommendations': 'UserMoviePreferences'}.get(name)
    if dependency and fact(history, dependency) is None:
        return result(0, 'missing_dependency', f'尚未获得 {dependency} 的实际返回。', [req], error_type='missing_prerequisite')
    for reference in refs:
        if equal_fields(args, reference['input'], docs[name]):
            ev = [req, evidence('reference_argument_check', reference['input'])]
            if dependency:
                h = fact(history, dependency)
                ev.append(evidence(f"step:{h['step_id']}.{dependency}", h['observation']))
            return result(1, 'requirement_or_prior_fact_match', '工具及参数符合需求或此前实际返回；这是局部调用判定，不保证完整任务完成。', ev)
    return result(None, 'requires_additional_semantic_review', '不直接沿用旧判错，需要额外语义审查。', [req])

def tied_risk(p, y):
    """Expected risk at each coverage under random ordering inside exact score ties."""
    order = np.argsort(-p, kind='stable')
    scores, errs = p[order], 1-y[order]
    risks=[]
    i=0
    previous_errors=0.0
    while i<len(y):
        j=i+1
        while j<len(y) and scores[j] == scores[i]: j+=1
        mean_error=float(errs[i:j].mean())
        for k in range(1,j-i+1): risks.append((previous_errors+k*mean_error)/(i+k))
        previous_errors += float(errs[i:j].sum())
        i=j
    return np.arange(1,len(y)+1)/len(y), np.array(risks)

def capture(p, y, budget=0.2):
    if not (1-y).sum(): return None
    remaining=budget*len(y)
    detected=0.0
    for score in sorted(set(p)):
        idx=p==score
        fraction=min(1.0,max(0.0,remaining/int(idx.sum())))
        detected+=fraction*float((1-y[idx]).sum())
        remaining-=min(remaining,int(idx.sum()))
        if remaining<=0: break
    return detected/float((1-y).sum())

def metrics(p, y):
    p=np.clip(np.asarray(p,dtype=float),0,1)
    y=np.asarray(y,dtype=int)
    result={'n':len(y),'correct':int(y.sum()),'errors':int((1-y).sum()),
            'AUROC':None,'error_AUPRC':None,'Brier':None,'NLL':None,'ECE':None,
            'AURC':None,'error_capture_at_20pct':None}
    if not len(y): return result
    q=np.clip(p,1e-6,1-1e-6)
    result['Brier']=float(np.mean((p-y)**2))
    result['NLL']=float(np.mean(-y*np.log(q)-(1-y)*np.log(1-q)))
    ece=0.0
    bins=[]
    for i in range(5):
        mask=(p>=i/5)&((p<(i+1)/5) if i<4 else (p<=1))
        n=int(mask.sum())
        if n:
            avg,acc=float(p[mask].mean()),float(y[mask].mean())
            ece+=n/len(y)*abs(avg-acc)
            bins.append({'lo':i/5,'hi':(i+1)/5,'n':n,'confidence':avg,'accuracy':acc})
    result['ECE']=ece
    result['bins']=bins
    if len(set(y))==2:
        result['AUROC']=float(roc_auc_score(y,p))
        result['error_AUPRC']=float(average_precision_score(1-y,1-p))
    result['AURC']=float(tied_risk(p,y)[1].mean())
    result['error_capture_at_20pct']=capture(p,y)
    return result

def read_lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def get_score(row, name):
    raw = row['original_step'].get('raw_logprobs') or {}
    value = raw.get(name)
    if not raw.get('available') or raw.get('degenerate'): return None
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        return None
    return float(value)

def auc_ci(p, y, task_ids, n_boot, seed=20261015):
    """Resample entire tasks, retaining all their calls and score ties."""
    groups = np.unique(task_ids)
    if not len(groups): return {'ci95': None, 'valid_resamples': 0, 'total_resamples': n_boot}
    indices = {g: np.flatnonzero(task_ids == g) for g in groups}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = np.concatenate([indices[g] for g in rng.choice(groups, len(groups), replace=True)])
        if len(np.unique(y[idx])) == 2:
            values.append(float(roc_auc_score(y[idx], p[idx])))
    return {'ci95': np.quantile(values, [.025, .975]).tolist() if values else None,
            'valid_resamples': len(values), 'total_resamples': n_boot,
            'unit': 'task', 'seed': seed,
            'caveat': 'Single-class resamples omitted from AUROC interval; few error tasks limit precision.'}

def summarize_cohort(pairs, n_boot=0):
    result = {'n': len(pairs), 'tasks': len({r['task_id'] for r, _ in pairs}),
              'correct': sum(y for _, y in pairs), 'errors': sum(1-y for _, y in pairs),
              'error_tasks': sorted({r['task_id'] for r, y in pairs if not y}), 'signals': {}}
    for name in SIGNALS:
        available = [(r, y, get_score(r, name)) for r, y in pairs if get_score(r, name) is not None]
        p = np.array([score for _, _, score in available], dtype=float)
        y = np.array([label for _, label, _ in available], dtype=int)
        stat = metrics(p, y)
        stat['missing_scores'] = len(pairs)-len(available)
        stat['correct_mean'] = float(p[y == 1].mean()) if (y == 1).any() else None
        stat['error_mean'] = float(p[y == 0].mean()) if (y == 0).any() else None
        stat['random_error_AUPRC'] = float((1-y).mean()) if len(y) else None
        stat['random_error_capture_at_20pct'] = .2 if (1-y).sum() else None
        if n_boot:
            stat['AUROC_bootstrap'] = auc_ci(p, y, np.array([r['task_id'] for r, _, _ in available]), n_boot)
        result['signals'][name] = stat
    return result
