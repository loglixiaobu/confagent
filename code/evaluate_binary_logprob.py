#!/usr/bin/env python
"""Evaluate all 198 binary-labeled calls; retain unscored end decisions."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from offline_confidence_utils import SIGNALS, MAIN, auc_ci, get_score, read_lines, summarize_cohort, ROOT, dump, sha


def checked_inputs(out):
    summary=json.loads((out/'label_summary.json').read_text())
    protocol=json.loads((out/'protocol.json').read_text())
    if sha(out/'protocol.json')!=summary['protocol_sha256']:raise ValueError('Protocol changed')
    if sha(ROOT/protocol['source_file'])!=summary['source_sha256']:raise ValueError('Original source changed')
    for file,digest in summary['artifact_sha256'].items():
        if sha(out/file)!=digest:raise ValueError(f'Label artifact changed: {file}')
    calls=read_lines(out/'calls_binary.jsonl')
    source={(t['task_id'],s['step_id']):s for t in read_lines(ROOT/protocol['source_file']) for s in t['steps']}
    if len(calls)!=len(source) or {(r['task_id'],r['step_id'])for r in calls}!=set(source):raise ValueError('Missing/duplicate call')
    for r in calls:
        if r['original_step']!=source[(r['task_id'],r['step_id'])]:raise ValueError('Original score or action changed')
        if r['revision']['y']not in(0,1):raise ValueError('Nonbinary label')
    return calls,read_lines(out/'end_responses_binary.jsonl'),read_lines(out/'task_outcomes_binary.jsonl'),summary


def figure(out,calls,ends,tasks,stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    a=axes[0,0]
    correct=[sum(r['original_label']for r in calls),sum(r['revision']['y']for r in calls)]
    wrong=[len(calls)-v for v in correct]
    a.bar(['Original labels','Binary v2'],correct,color='#238b45',label='Correct')
    a.bar(['Original labels','Binary v2'],wrong,bottom=correct,color='#cb181d',label='Incorrect')
    for i in range(2):
        a.text(i,correct[i]/2,str(correct[i]),ha='center',va='center',color='white')
        a.text(i,correct[i]+wrong[i]/2,str(wrong[i]),ha='center',va='center',color='white')
    a.set(title='All 198 original tool calls; no label exclusions',ylabel='Calls');a.legend()
    a=axes[0,1]
    for key,signal,label in [('original_calls',MAIN,'Original mean probability'),('binary_calls',MAIN,'Binary mean probability'),('binary_calls','p10_prob','Binary p10 probability')]:
        yy=np.array([int(r['original_label'])if key=='original_calls'else r['revision']['y']for r in calls])
        pp=np.array([get_score(r,signal)for r in calls])
        fpr,tpr,_=roc_curve(yy,pp)
        auc=stats[key]['signals'][signal]['AUROC']
        a.plot(fpr,tpr,label=f'{label}: {auc:.3f}')
    a.plot([0,1],[0,1],'--',color='gray');a.set(title='AUROC on exactly the same 198 calls',xlabel='False positive rate',ylabel='True positive rate');a.legend(fontsize=8)
    a=axes[1,0]
    groups=[('Tool calls (n=198)',[r['revision']['y']for r in calls]),('No-tool responses (n=18)',[r['y']for r in ends]),('Task outcomes (n=50)',[r['y']for r in tasks])]
    for i,(name,ys)in enumerate(groups):
        a.barh(i,sum(ys),color='#238b45');a.barh(i,len(ys)-sum(ys),left=sum(ys),color='#cb181d')
        a.text(sum(ys)/2,i,str(sum(ys)),ha='center',va='center',color='white')
        a.text(sum(ys)+(len(ys)-sum(ys))/2,i,str(len(ys)-sum(ys)),ha='center',va='center',color='white')
    a.set(yticks=range(3),yticklabels=[g[0]for g in groups],title='Different populations: do not add task outcomes')
    a=axes[1,1]
    for y,color,label in [(1,'#2171b5','Correct calls'),(0,'#cb181d','Incorrect calls')]:
        p=[get_score(r,MAIN)for r in calls if r['revision']['y']==y]
        a.hist(p,bins=np.linspace(.84,1,17),alpha=.5,density=True,color=color,label=label)
    a.set(title='Binary labels: original mean token probabilities',xlabel='Mean token probability',ylabel='Density');a.legend(fontsize=9)
    fig.suptitle('Binary review of recorded behavior — CPU only, no new rollouts',fontsize=13)
    fig.savefig(out/'binary_logprob_diagnostic.png',dpi=170);plt.close(fig)


def report(out,calls,ends,tasks,summary,results):
    m=results['cohorts']['binary_calls']['signals'][MAIN]
    ci=m['AUROC_bootstrap']['ci95']
    counts=summary['call_labels'];endcounts=summary['end_response_labels'];taskcounts=summary['task_labels']
    lines=['# 二元标签修订与 logprob 评估（v2）','',
           '日期：2026-10-15。按用户确认的规则，仅使用 **正确（1）／错误（0）** 两个标签。全部为 CPU 离线处理，模型调用 0 次。原始数据及 v1 审计保留。','',
           f'**198 次工具调用全部纳入：{counts["正确"]} 正确、{counts["错误"]} 错误。** 另补原日志中 18 次无工具输出的任务结束标签：{endcounts["正确"]} 正确、{endcounts["错误"]} 错误。'
           f'50 个任务的结束结果为 {taskcounts["正确"]} 完成、{taskcounts["错误"]} 未完成。三个统计对象分别报告，不能混成一个准确率。','',
           f'原有 geo_mean_prob 在这 198 次调用上的 AUROC：旧标签 0.5401 → 二元标签 **{m["AUROC"]:.4f}**。'
           f'按任务 bootstrap 95% 区间为 [{ci[0]:.3f}, {ci[1]:.3f}]。'
           '这是已经审阅过的旧轨迹重评，不是独立测试，也没有训练新校准器。','',
           '## 1. 固定二元判定规则','',
           '- 关键地点、日期、时长没有来源而自行填写：错误。','- 未查询齐所需数据就计算、无依据改变坐标或工资周期：错误。',
           '- API 单位不符或关键单位未经核实就执行：错误。','- 数字与数字字符串、等价算式、地址句号等表示差异不造成语义错误。',
           '- 环境有 schema/参考缺陷时记录原因；同一次调用里其他无依据参数仍判错，不设第三类标签。',
           '- 先前调用正确，最后漏做、未产出必要结果：结束时任务标签为错误；不把先前正确调用一并改错。',
           '- 中间出错后实际完成目标：任务可以正确，不再要求整条历史从未失败。','',
           '这是一套可复现的“任务依据与工具协议”评分口径。工资周期、坐标、湿度等旧争议项也按该口径明确给 0/1，原因和环境责任保留在元数据中；不声称它们经过外部独立事实核验。','',
           '## 2. 工具调用的错误类型','', '| 错误类型 | 次数 |', '| --- | ---: |']
    names={'fabricated_travel_inputs':'编造未提供的出行参数','fabricated_meeting_location':'编造会议地点',
           'fabricated_checkout_date':'编造退房日期','unsupported_salary_period_conversion':'无依据改变工资周期',
           'unverified_coordinate_sign_change':'未核验就改变坐标符号','humidity_unit_not_resolved':'湿度单位未核实/转换',
           'kilometers_not_converted_to_meters':'公里未转换为米','calculation_before_all_counts_observed':'未查齐数据即计算',
           'search_no_result':'检索未返回结果','retrieval_capability_mismatch':'检索能力不匹配'}
    for code,n in summary['error_reasons'].items():lines.append(f'| {names.get(code,code)} | {n} |')
    lines+=['','与旧 111 个负标签相比，78 个表示/参考误判仍修为正确；另外 7 个旧正标签因无依据参数、单位等问题改为错误。'
            '不是把此前所有争议项机械改为错误：两个有用的工具文档搜索判为正确。','',
            '## 3. 漏做和提前结束如何计入','',
            '原始 198 步只包含工具调用。另有 18 次模型输出没有工具调用，原运行器直接以 final_answer 结束，其中还包括未闭合思考、未形成正式答案的输出。'
            '这些输出逐一补充二元任务结束标签，纳入 `events_binary.csv`，包括零调用任务。','',
            '另 32 个结束由运行器触发（success、repeated_call 或 max_steps），只记任务结果，不伪装成模型主动做出的停止决策。'
            '结束标签表示目标是否完成；例如任务未完成但提出合理澄清，仍是任务未完成，不把澄清请求本身描述成不合理行为。','',
            '例子：l3-000 的查询和已发送提醒仍正确，但遗漏另一场应发送提醒，结束标签错误；l3-014 的预约 API 返回错误，却声称预约成功，结束标签错误。'
            'l3-026 两次提前求和调用错误，但后来已查齐数据并给出正确总数 60，任务结束标签正确。','',
            '**结束输出的 logprob 未被原日志保存。** 这 18 个事件保留标签，但不能重算其置信度 AUROC；不以 0 或最后一次工具调用分数填充。'
            '所以当前调用分数评估覆盖 198/198 调用，覆盖已标注模型事件 198/216；停止质量的置信度评估仍缺原始分数。','',
            '## 4. 同一批任务为什么从 13/50 变成 14/50','',
            '这是修改完成判定后的重新审计，不是能力提升。判定依据是实际日志中的任务目标是否完成；不会把被模拟器拒绝的正确调用假装成一次实际成功执行。'
            '不要求额外的 AccountInfo，也不要求中途从未失败。电影任务要求有排除已看电影的记录，求和任务要求实际计算结果或正确最终总数。','',
            '| 任务类别 | 旧通过数 | 本次完成数 |', '| --- | ---: | ---: |']
    family_names=['会议提醒','天气与穿衣','安排会议','餐厅查询','电影推荐与过滤','点赞总数','城市天气','机票酒店','税后工资','个人信息更新']
    for family,name in enumerate(family_names):
        group=[r for r in tasks if int(r['task_id'][3:])//5==family]
        lines.append(f'| {name} | {sum(r["old_task_success"]for r in group)} | {sum(r["y"]for r in group)} |')
    lines+=['','电影过滤、部分求和任务被原运行器过早标 success，原因记录为运行器遗漏目标，不能直接归咎于模型。'
            '机酒任务的退房参数没有依据，即使匹配隐藏参考也不算有依据地完成。'
            '天气、餐厅等任务中，一些参数语义正确的调用被旧模拟器拒绝，既标出正确调用，也在任务失败原因中记环境拒绝。','',
            '会议提醒按该任务参考链中指定的提醒目标检查；会议数与返回数据的差异不再临时改变目标数量。'
            '工资任务按需求的月薪框架解释未声明其他周期的返回金额，不自行假设年薪。具体每个目标的证据见任务 CSV/JSONL。','',
            '## 5. 原有 logprob 重新评估','',
            '高分应更正确。全部分数原样复用，无新采样、拟合或事后翻转。错误 AUPRC 随机基线为 40/198。'
            'Brier/ECE 是把原始分数当正确概率时的描述性误差，不代表已校准。','',
            '| 信号 | AUROC | 错误 AUPRC | Brier | ECE | 最低分 20% 的错误捕获率 |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,stat in results['cohorts']['binary_calls']['signals'].items():
        lines.append(f'| {name} | {stat["AUROC"]:.4f} | {stat["error_AUPRC"]:.4f} | {stat["Brier"]:.4f} | {stat["ECE"]:.4f} | {stat["error_capture_at_20pct"]:.4f} |')
    lines+=['','| 分析集合 | 调用数 | 错误数 | 平均概率 AUROC |', '| --- | ---: | ---: | ---: |']
    for name,stat in results['cohorts'].items():
        value=stat['signals'][MAIN]['AUROC']
        lines.append(f'| {name} | {stat["n"]} | {stat["errors"]} | {value:.4f} |'if value is not None else f'| {name} | {stat["n"]} | {stat["errors"]} | 不可计算 |')
    lines+=['','主分析包含全部 198 次调用。业务调用子集、去重、首次 mock 报错之前只是另报的敏感性检查，不替代主结果。'
            '去重键包含任务、API、语义参数和相关前置证据；同公式在信息查询齐全前后仍是不同状态。'
            '置信区间按任务重采样，有效重采样次数记录在 JSON 中。历史依然受到旧模拟器错误反馈影响，不能当作修复环境后的新执行结果。','',
            '![二元标签与分数诊断](binary_logprob_diagnostic.png)','',
            '## 6. 文件与复现','',
            '- `events_binary.csv`：198 次调用 + 18 次原始无工具输出；所有新标签均为 0/1。',
            '- `calls_binary.jsonl`：全部 198 次调用及原分数、原标签、新标签和理由。',
            '- `end_responses_binary.jsonl`：18 次无工具输出的二元标签，明确标记分数缺失。',
            '- `task_outcomes_binary.csv` / `.jsonl`：50 个任务，逐项列出未完成目标和环境/模型/运行器原因。',
            '- 原始 `results/traj_main_clean.jsonl` 保持不变；逐步原字段另保存在 `calls_binary.jsonl`。',
            '- `protocol.json`、`label_summary.json`、`score_blind_review.jsonl`：冻结口径、数量、无分数审查依据。',
            '- `logprob_metrics.json` / `.csv`：分数评估与敏感性结果。','',
            '只重算已发布的 198 步指标，不需要模型服务或 API-Bank 原始数据；安装 `requirements-analysis.txt` 后运行评估入口即可。重新生成标签时，另按仓库数据说明准备 API-Bank 的 `test-data/level-3.json`，通过 `--data-dir` 指定。`tests/fixtures/` 仅包含选定的回归样例，不是完整数据集。','',
            '```bash','cd confagent', 'python -m pip install -r requirements-analysis.txt',
            'OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python code/relabel_binary_steps.py --data-dir data/api-bank',
            'OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python code/evaluate_binary_logprob.py','```']
    (out/'二元标签与logprob评估.md').write_text('\n'.join(lines)+'\n')


def evaluate(out,n_boot=1000):
    out=Path(out)
    calls,ends,tasks,summary=checked_inputs(out)
    pack=lambda seq,original=False:[(r,int(r['original_label'])if original else r['revision']['y'])for r in seq]
    groups={'original_calls':pack(calls,True),'binary_calls':pack(calls),
            'binary_business_calls':pack([r for r in calls if r['scope']=='business_call']),
            'binary_first_semantic_attempt':pack([r for r in calls if r['first_semantic_attempt']]),
            'binary_before_first_mock_error':pack([r for r in calls if r['before_first_mock_error']])}
    results={'version':'legacy-binary-v2','source_sha256':summary['source_sha256'],
             'labels_sha256':summary['artifact_sha256'],'evaluation_code_sha256':sha(__file__),
             'metric_code_sha256':sha(ROOT/'code/offline_confidence_utils.py'),
             'cohorts':{},'primary_signal':MAIN,'primary_population':'binary_calls',
             'no_tool_scores_missing':len(ends),'model_events_scored':len(calls),'model_events_labeled':len(calls)+len(ends),
             'gpu_used':False,'model_requests':0,'new_calibrator_fitted':False}
    for name,pairs in groups.items():
        stat=summarize_cohort(pairs)
        if name!='original_calls':
            valid=[(r,y)for r,y in pairs if get_score(r,MAIN)is not None]
            pp=np.array([get_score(r,MAIN)for r,_ in valid]);yy=np.array([y for _,y in valid])
            stat['signals'][MAIN]['AUROC_bootstrap']=auc_ci(pp,yy,np.array([r['task_id']for r,_ in valid]),n_boot)
        results['cohorts'][name]=stat
    dump(out/'logprob_metrics.json',results)
    records=[]
    for name,stat in results['cohorts'].items():
        for signal,m in stat['signals'].items():
            records.append({'cohort':name,'signal':signal,**{k:m[k]for k in ('n','correct','errors','AUROC','error_AUPRC','Brier','ECE','error_capture_at_20pct')}})
    with (out/'logprob_metrics.csv').open('w',newline='',encoding='utf-8-sig')as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    figure(out,calls,ends,tasks,results['cohorts'])
    report(out,calls,ends,tasks,summary,results)
    for name,stat in results['cohorts'].items():
        print(json.dumps({'cohort':name,'n':stat['n'],'errors':stat['errors'],'geo_mean_AUROC':stat['signals'][MAIN]['AUROC']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',default=str(ROOT/'outputs/legacy_binary_v2'))
    parser.add_argument('--bootstrap',type=int,default=1000)
    args=parser.parse_args()
    if args.bootstrap<1:parser.error('bootstrap must be positive')
    evaluate(args.out_dir,args.bootstrap)
