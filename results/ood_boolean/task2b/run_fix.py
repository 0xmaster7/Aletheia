"""Run the one preregistered Boolean plan-polarity fix and replay all 100 items."""
from __future__ import annotations
import hashlib,json,os,sys,time
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/ood_boolean/task2b';REQ=OUT/'requests.jsonl';MAN=OUT/'frozen_manifest.json';CALLS=OUT/'calls.jsonl';PLANS=OUT/'plans_and_scores.jsonl';SUMMARY=OUT/'results_summary.json';CHECK=OUT/'checkpoint.json'
CAP=.30;IN_RATE=.15;OUT_RATE=.60
BASE=ROOT/'results/prof_feedback/pf-20261006-hybrid300-final-01'
def sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()
def shafile(p:Path)->str:return sha(p.read_bytes())
def canon(v:Any)->str:return sha(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())
def rjsonl(p:Path):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def wjson(p:Path,x:Any):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def append(p:Path,x:Any):
    with p.open('a',encoding='utf-8',buffering=1) as f:f.write(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
def verify():
    m=json.loads(MAN.read_text());rows=rjsonl(REQ)
    if shafile(REQ)!=m['request_manifest_sha256'] or len(rows)!=100:raise RuntimeError('Task 2b frozen request set changed')
    checks={'baseline_calls_sha256':ROOT/'logs/prof_feedback/pf-20261006-hybrid300-final-01/calls.jsonl','baseline_requests_sha256':BASE/'requests.jsonl','fix_policy_sha256':OUT/'config/fix_policy.txt','operator_fix_sha256':OUT/'config/boolean_polarity_fix.py'}
    for key,p in checks.items():
        if shafile(p)!=m[key]:raise RuntimeError(f'frozen dependency changed: {p.name}')
    x=dict(m);fingerprint=x.pop('manifest_sha256')
    if canon(x)!=fingerprint:raise RuntimeError('Task 2b frozen manifest fingerprint mismatch')
    return m,rows
def run():
    m,rows=verify()
    if CALLS.exists():raise RuntimeError('Task 2b call log exists; no retries or reruns allowed')
    from results.extensions.run_extensions import client_or_raise
    client=client_or_raise();spend=0.0;done=0
    for r in rows:
        cap=int(r['max_completion_tokens']);guard=(int(r['local_input_tokens'])+100)*IN_RATE/1e6+cap*OUT_RATE/1e6
        if spend+guard>CAP:
            wjson(CHECK,{'status':'hard_stop_guard_before_call','completed':done,'expected':100,'actual_spend_usd':round(spend,10),'hard_stop_usd':CAP,'next_call':r['call_id']});break
        body={k:r[k] for k in ('model','messages','temperature','max_completion_tokens','response_format')}
        start=time.time()
        try:response=client.chat.completions.create(**body)
        except Exception as exc:
            append(CALLS,{'call_id':r['call_id'],'status':'request_error','error_type':type(exc).__name__,'error':str(exc).replace(os.environ.get('OPENAI_API_KEY',''),'[redacted]')[:1000],'usage_prompt_tokens':None,'usage_completion_tokens':None,'actual_cost_usd':None})
            wjson(CHECK,{'status':'request_error_unknown_usage_stop','completed':done+1,'expected':100,'actual_spend_usd':round(spend,10),'call_id':r['call_id']});break
        usage=response.usage
        if usage is None or getattr(usage,'prompt_tokens',None) is None or getattr(usage,'completion_tokens',None) is None:
            append(CALLS,{'call_id':r['call_id'],'status':'usage_missing','response':response.model_dump(mode='json'),'actual_cost_usd':None})
            wjson(CHECK,{'status':'usage_missing_stop','completed':done+1,'expected':100,'actual_spend_usd':round(spend,10),'call_id':r['call_id']});break
        pt=int(usage.prompt_tokens);ct=int(usage.completion_tokens);pd=getattr(usage,'prompt_tokens_details',None);cached=int(getattr(pd,'cached_tokens',0) or 0)
        cost=((pt-cached)*IN_RATE+cached*IN_RATE/2+ct*OUT_RATE)/1e6
        choice=response.choices[0];finish=choice.finish_reason;status='truncated_failure' if finish=='length' else 'completed'
        append(CALLS,{'call_id':r['call_id'],'source_index':r['source_index'],'question_id':r['question_id'],'status':status,'finish_reason':finish,'response_text':choice.message.content or '',
                      'usage_prompt_tokens':pt,'usage_completion_tokens':ct,'usage_cached_prompt_tokens':cached,'actual_cost_usd':cost,'elapsed_seconds':time.time()-start,'model':response.model,'system_fingerprint':getattr(response,'system_fingerprint',None)})
        spend+=cost;done+=1;wjson(CHECK,{'status':'running' if spend<CAP else 'hard_stop_reached','completed':done,'expected':100,'actual_spend_usd':round(spend,10),'hard_stop_usd':CAP,'last_call':r['call_id']})
        print(json.dumps({'call':r['call_id'],'status':status,'usage_prompt_tokens':pt,'usage_completion_tokens':ct,'spend_usd':round(cost,8),'task2b_spend_usd':round(spend,8)}),flush=True)
        if spend>=CAP:break
    return json.loads(CHECK.read_text())
def score():
    m,requests=verify();calls=rjsonl(CALLS);byid={c['call_id']:c for c in calls}
    baseline=json.load(open(BASE/'scored_calls.json'));old={(r['source_index'],r['arm']):r for r in baseline}
    old_call_rows=rjsonl(ROOT/'logs/prof_feedback/pf-20261006-hybrid300-final-01/calls.jsonl')
    old_call_by_source={r['source_index']:r for r in old_call_rows if r['arm']=='aletheia'}
    from scripts.lib.prof_feedback import QueryPlan
    from scripts.lib.prof_feedback_final import execute_final_plan
    from scripts.lib.evaluation_scorer import score_answer
    from results.ood_boolean.task2b.config.boolean_polarity_fix import apply_boolean_polarity_fix
    fm=json.load(open(BASE/'frozen_manifest.json'));ec=fm['entity_count_map'];pc={tuple(k.split('\t',1)):int(v) for k,v in fm['entity_property_count_map'].items()}
    results=[]
    for r in requests:
        call=byid.get(r['call_id'],{'status':'not_run'});pred=None;status='failed';plan0=plan1=None;complete=None
        fresh_before=None;fresh_before_correct=False;oldplan=None
        if call.get('status')=='completed':
            try:
                plan0=QueryPlan.from_object(json.loads(call['response_text'])['query_plan']);plan1=apply_boolean_polarity_fix(r['question'],plan0)
                _p0,fresh_answer,complete=execute_final_plan(r['question'],plan0,r['retrieved_items'],ec,pc)
                fresh_before=fresh_answer.get('answer');fresh_before_correct=bool(score_answer('boolean',fresh_before,r['ground_truth_answer'])) if fresh_answer.get('status')=='answered' else False
                _p1,answer,complete=execute_final_plan(r['question'],plan1,r['retrieved_items'],ec,pc);pred=answer.get('answer');status=answer.get('status','abstained')
                old_raw=old_call_by_source[r['source_index']]
                oldplan=QueryPlan.from_object(json.loads(old_raw['response_text'])['query_plan']).as_dict()
            except (ValueError,KeyError,TypeError,json.JSONDecodeError):plan0=plan1=None;complete=None;status='invalid_plan'
        correct=bool(score_answer('boolean',pred,r['ground_truth_answer'])) if status=='answered' else False
        oldrow=old[(r['source_index'],'aletheia')]
        results.append({'call_id':r['call_id'],'question_id':r['question_id'],'source_index':r['source_index'],'question':r['question'],'gold':r['ground_truth_answer'],'flagged':r['source_gold_flagged'],
                        'saved_prediction':oldrow.get('prediction'),'saved_correct':bool(oldrow['correct']),'new_prediction':pred,'new_correct':correct,'new_answer_status':status,
                        'fresh_plan_prediction_before_fix':fresh_before,'fresh_plan_correct_before_fix':fresh_before_correct,'old_saved_plan':oldplan,
                        'plan_before':plan0.as_dict() if plan0 else None,'plan_after_fix':plan1.as_dict() if plan1 else None,'fix_changed_negated':bool(plan0 and plan1 and plan0.negated!=plan1.negated),'complete_history':complete,'call_status':call.get('status')})
    flips={'wrong_to_right':[r for r in results if not r['saved_correct'] and r['new_correct']],'right_to_wrong':[r for r in results if r['saved_correct'] and not r['new_correct']]}
    per_intent={'boolean':{'before':{'n':100,'correct':sum(r['saved_correct'] for r in results)},'after':{'n':100,'correct':sum(r['new_correct'] for r in results)}}}
    calls_spend=sum(float(c.get('actual_cost_usd') or 0) for c in calls);pt=sum(int(c.get('usage_prompt_tokens') or 0) for c in calls);ct=sum(int(c.get('usage_completion_tokens') or 0) for c in calls)
    same_call_flips={'wrong_to_right':[r for r in results if not r['fresh_plan_correct_before_fix'] and r['new_correct']],'right_to_wrong':[r for r in results if r['fresh_plan_correct_before_fix'] and not r['new_correct']]}
    non_polarity_drift=[]
    for r in results:
        if r['plan_before'] and r['old_saved_plan']:
            changed=[k for k in r['plan_before'] if k!='negated' and r['plan_before'][k]!=r['old_saved_plan'][k]]
            if changed:non_polarity_drift.append({'question_id':r['question_id'],'fields':changed})
    fresh_before_correct=sum(r['fresh_plan_correct_before_fix'] for r in results)
    summ={'run_id':'task2b_confirmatory_boolean_polarity_fix','manifest_sha256':m['manifest_sha256'],'calls_logged':len(calls),'expected_calls':100,'status':json.loads(CHECK.read_text())['status'],'model':'gpt-4o-mini','temperature':0.0,'k':80,'hard_stop_usd':CAP,'actual_spend_usd':round(calls_spend,10),'prompt_tokens':pt,'completion_tokens':ct,'saved_baseline_correct':sum(r['saved_correct'] for r in results),'saved_baseline_accuracy':sum(r['saved_correct'] for r in results)/100,'fresh_plan_before_fix_correct':fresh_before_correct,'fresh_plan_before_fix_accuracy':fresh_before_correct/100,'after_correct':sum(r['new_correct'] for r in results),'after_accuracy':sum(r['new_correct'] for r in results)/100,'abstained_after':sum(r['new_answer_status']!='answered' for r in results),'fix_changed_negated':sum(r['fix_changed_negated'] for r in results),'same_fresh_plan_flips':{k:[{f:r[f] for f in ('question_id','source_index','question','gold','fresh_plan_prediction_before_fix','new_prediction','fresh_plan_correct_before_fix','new_correct','new_answer_status','plan_before','plan_after_fix')} for r in v] for k,v in same_call_flips.items()},'saved_baseline_to_fixed_fresh_run_flips':{k:[{f:r[f] for f in ('question_id','source_index','question','gold','saved_prediction','new_prediction','saved_correct','new_correct','new_answer_status','plan_before','plan_after_fix')} for r in v] for k,v in flips.items()},'fresh_vs_saved_planner_other_field_differences_count':len(non_polarity_drift),'fresh_vs_saved_planner_other_field_differences':non_polarity_drift,'per_intent':per_intent,'truncated_failures':sum(c.get('status')=='truncated_failure' for c in calls),'request_errors':sum(c.get('status')=='request_error' for c in calls)}
    wjson(PLANS,results);wjson(SUMMARY,summ);return summ
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--run',action='store_true');g.add_argument('--score',action='store_true');a=p.parse_args();print(json.dumps(run() if a.run else score(),indent=2))
