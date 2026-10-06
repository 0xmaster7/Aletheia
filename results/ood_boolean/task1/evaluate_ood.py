"""Prepare, run once/resumably, and score the frozen answer-level paraphrase set.

Preparation and scoring are offline. ``--run`` sends API requests.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys, time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/ood_boolean/task1'
PARAPHRASES=OUT/'paraphrases.jsonl'
REQUESTS=OUT/'requests.jsonl'
MANIFEST=OUT/'frozen_manifest.json'
CALLS=OUT/'calls.jsonl'
CHECKPOINT=OUT/'checkpoint.json'
REPORT=OUT/'results_summary.json'
SCORED=OUT/'scored_calls.json'
BASE=ROOT/'results/prof_feedback/pf-20261006-hybrid300-final-01'
GEN_LOGS=[OUT/'paraphrase_generation_calls.jsonl',OUT/'paraphrase_generation_calls_final.jsonl']
INPUT_RATE=.15
OUTPUT_RATE=.60
TASK1_CAP=.40

def sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()
def sha_file(p:Path)->str:return sha(p.read_bytes())
def canonical(value:Any)->str:return sha(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode())
def read_json(p:Path)->Any:return json.loads(p.read_text())
def read_jsonl(p:Path)->list[dict[str,Any]]:return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
GENERATION_SPEND=sum(float(r.get('actual_cost_usd') or 0) for p in GEN_LOGS for r in read_jsonl(p)) if all(p.exists() for p in GEN_LOGS) else .00481005
def write_json(p:Path,x:Any)->None:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def write_jsonl(p:Path,rows:list[dict[str,Any]])->None:p.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows))

def token_count(req:dict[str,Any],enc:Any)->int:
    total=3
    for m in req['messages']:
        total+=4+len(enc.encode(m['role']))+len(enc.encode(m['content']))
    if req.get('response_format'):
        total+=len(enc.encode(json.dumps(req['response_format'],ensure_ascii=False,separators=(',',':'))))
    return total

def prepare()->dict[str,Any]:
    import numpy as np
    import tiktoken
    from rank_bm25 import BM25Okapi
    from scripts.analysis.final_hybrid_confirmatory import parse_corpus_facts
    from scripts.lib.prof_feedback import build_control_request,parse_fact_template,property_type_from_fact
    from scripts.lib.prof_feedback_final import build_final_evaluation_request,KNOWN_RELATION_LIST,TEMPORAL_CUE_MAP,COUNT_CUES
    from results.extensions.run_extensions import load_corpus
    if not PARAPHRASES.is_file():raise RuntimeError('Frozen paraphrase file missing')
    for p in (REQUESTS,MANIFEST,CALLS):
        if p.exists():raise RuntimeError(f'Refusing to overwrite frozen output: {p.name}')
    paraphrases=read_jsonl(PARAPHRASES)
    if len(paraphrases)!=300 or Counter(r['intent'] for r in paraphrases)!=Counter({'historical':100,'aggregation':100,'boolean':100}):
        raise RuntimeError('Paraphrase manifest must have exactly 100 rows per intent')
    facts,context_sha,arrow_sha=load_corpus()
    texts=[f['text'] for f in facts]
    bm25=BM25Okapi([re.findall(r'[A-Za-z0-9]+',t.lower()) for t in texts])
    entities={r['entity'] for r in paraphrases}
    counts=Counter();property_counts=Counter();raw_counts=Counter()
    for f in facts:
        for entity in entities:
            if f['text'].startswith(entity+' '):
                raw_counts[entity]+=1
                parsed=parse_fact_template(f['text'])
                if parsed and parsed[0]==entity:
                    counts[entity]+=1;property_counts[(entity,property_type_from_fact(f['text']))]+=1
    reliable={e:counts[e] for e in entities if counts[e]==raw_counts[e]}
    from scripts.lib.prof_feedback import approved_evaluation_prompt_and_schema
    _,schema=approved_evaluation_prompt_and_schema();enc=tiktoken.get_encoding('o200k_base')
    requests=[]
    for pos,item in enumerate(paraphrases,1):
        q=item['paraphrased_question'];scores=bm25.get_scores(re.findall(r'[A-Za-z0-9]+',q.lower()))
        order=np.argsort(scores)[::-1][:80]
        retrieved=[{'fact_idx':facts[int(ix)]['fact_idx'],'text':facts[int(ix)]['text']} for ix in order]
        for arm in ('aletheia','direct','cot'):
            req=build_final_evaluation_request(q,retrieved) if arm=='aletheia' else build_control_request(arm,q,retrieved)
            req['max_completion_tokens']={'aletheia':1024,'direct':512,'cot':1024}[arm]
            source={k:item[k] for k in ('paraphrase_id','source_index','source_question_id','intent','entity','gold','flagged','source_gold_issue_codes','source_support_status','original_question','paraphrased_question')}
            source.update({'call_id':f'ood-{pos:03d}-{arm}','arm':arm,'model':'gpt-4o-mini','temperature':0.0,'k':80,
                           'messages':req['messages'],'response_format':req.get('response_format'),'max_completion_tokens':req['max_completion_tokens'],
                           'retrieved_items':retrieved,'local_input_tokens':token_count(req,enc)})
            requests.append(source)
    if len(requests)!=900:raise RuntimeError('Expected 900 requests')
    write_jsonl(REQUESTS,requests)
    rel_path=ROOT/'scripts/lib/prof_feedback_final.py';base_path=ROOT/'scripts/lib/prof_feedback.py';scorer_path=ROOT/'scripts/lib/evaluation_scorer.py';prompt=ROOT/'scripts/prompts/evaluation_final_v3.txt'
    cues={'temporal':TEMPORAL_CUE_MAP,'relations':KNOWN_RELATION_LIST,'count':[x.pattern for x in COUNT_CUES]}
    from scripts.lib.config import DATASET_REVISION
    manifest={
      'experiment':'answer-level OOD paraphrases from confirmatory sample','code_commit':os.popen(f'git -C "{ROOT}" rev-parse HEAD').read().strip(),
      'git_branch':os.popen(f'git -C "{ROOT}" branch --show-current').read().strip(),'dirty_worktree':True,
      'n':300,'intent_counts':{'historical':100,'aggregation':100,'boolean':100},'paraphrases_sha256':sha_file(PARAPHRASES),
      'request_sha256':sha_file(REQUESTS),'requests_n':len(requests),'all_arms_same_k80_context':True,
      'dataset_revision':DATASET_REVISION,'fact_context_sha256':context_sha,'arrow_sha256':arrow_sha,
      'prompt_sha256':sha_file(prompt),'operator_sha256':sha_file(rel_path),'base_operator_sha256':sha_file(base_path),
      'scorer_sha256':sha_file(scorer_path),'cue_lists_sha256':canonical(cues),'cue_lists':cues,
      'caps':{'aletheia':1024,'direct':512,'cot':1024},'model':'gpt-4o-mini','temperature':0.0,
      'scorer':'scripts/lib/evaluation_scorer.py fixed intent-specific scorer','retrieval':'BM25Okapi, lowercase [A-Za-z0-9]+, numpy.argsort(scores)[::-1][:80]',
      'entity_fact_counts':dict(reliable),'entity_property_fact_counts':{e+'\t'+p:int(n) for (e,p),n in property_counts.items()},
      'generation_attempt_spend_usd':round(GENERATION_SPEND,10),'task_hard_stop_usd':TASK1_CAP,
      'input_usd_per_million':INPUT_RATE,'output_usd_per_million':OUTPUT_RATE,'retries':0,
      'python':sys.version,'numpy':np.__version__,'rank_bm25':'0.2.2','openai_sdk':__import__('openai').__version__,'tiktoken':tiktoken.__version__,
    }
    manifest['frozen_manifest_sha256']=canonical(manifest);write_json(MANIFEST,manifest)
    return manifest

def verify()->tuple[dict[str,Any],list[dict[str,Any]]]:
    m=json.loads(MANIFEST.read_text());req=read_jsonl(REQUESTS)
    if sha_file(PARAPHRASES)!=m['paraphrases_sha256'] or sha_file(REQUESTS)!=m['request_sha256']:
        raise RuntimeError('Frozen paraphrases or requests changed')
    from scripts.lib.prof_feedback_final import KNOWN_RELATION_LIST,TEMPORAL_CUE_MAP,COUNT_CUES
    cues={'temporal':TEMPORAL_CUE_MAP,'relations':KNOWN_RELATION_LIST,'count':[x.pattern for x in COUNT_CUES]}
    checks={
      'prompt_sha256':ROOT/'scripts/prompts/evaluation_final_v3.txt',
      'operator_sha256':ROOT/'scripts/lib/prof_feedback_final.py',
      'base_operator_sha256':ROOT/'scripts/lib/prof_feedback.py',
      'scorer_sha256':ROOT/'scripts/lib/evaluation_scorer.py',
    }
    for key,path in checks.items():
        if sha_file(path)!=m[key]:raise RuntimeError(f'Frozen configuration file changed: {path.name}')
    if canonical(cues)!=m['cue_lists_sha256']:raise RuntimeError('Frozen cue/relation lists changed')
    frozen=dict(m);fingerprint=frozen.pop('frozen_manifest_sha256')
    if canonical(frozen)!=fingerprint:raise RuntimeError('Frozen manifest fingerprint mismatch')
    if len(req)!=900:raise RuntimeError('Frozen requests are not 900 calls')
    return m,req

def run()->dict[str,Any]:
    m,requests=verify()
    if CALLS.exists():raise RuntimeError('Call log exists; refusing to retry or rerun')
    from results.extensions.run_extensions import client_or_raise,append_jsonl,write_json
    client=client_or_raise();spent=GENERATION_SPEND;done=0
    for row in requests:
        cap=row['max_completion_tokens'];guard=(row['local_input_tokens']+100)*INPUT_RATE/1e6+cap*OUTPUT_RATE/1e6
        if spent+guard>TASK1_CAP:
            write_json(CHECKPOINT,{'status':'hard_stop_guard_before_call','completed':done,'expected':900,'task_api_spend_including_generation':round(spent,10),'hard_stop':TASK1_CAP,'next_call':row['call_id']});break
        body={k:row[k] for k in ('model','messages','temperature','max_completion_tokens')}
        if row['arm']=='aletheia':body['response_format']=row['response_format']
        start=time.time()
        try:response=client.chat.completions.create(**body)
        except Exception as exc:
            append_jsonl(CALLS,{'call_id':row['call_id'],'arm':row['arm'],'source_index':row['source_index'],'status':'request_error','error_type':type(exc).__name__,'error':str(exc).replace(os.environ.get('OPENAI_API_KEY',''),'[redacted]')[:1000],'usage_prompt_tokens':None,'usage_completion_tokens':None,'actual_cost_usd':None})
            write_json(CHECKPOINT,{'status':'request_error_unknown_usage_stop','completed':done+1,'expected':900,'task_api_spend_including_generation':round(spent,10),'failed_call':row['call_id']});break
        usage=response.usage
        if usage is None or getattr(usage,'prompt_tokens',None) is None or getattr(usage,'completion_tokens',None) is None:
            append_jsonl(CALLS,{'call_id':row['call_id'],'arm':row['arm'],'source_index':row['source_index'],'status':'usage_missing','response':response.model_dump(mode='json'),'actual_cost_usd':None})
            write_json(CHECKPOINT,{'status':'usage_missing_stop','completed':done+1,'expected':900,'task_api_spend_including_generation':round(spent,10),'failed_call':row['call_id']});break
        pt=int(usage.prompt_tokens);ct=int(usage.completion_tokens);details=getattr(usage,'prompt_tokens_details',None);cached=int(getattr(details,'cached_tokens',0) or 0)
        # Frozen campaign rate; cached prompt tokens were billed at the cached-input rate.
        cost=((pt-cached)*INPUT_RATE+cached*(INPUT_RATE/2)+ct*OUTPUT_RATE)/1e6
        choice=response.choices[0];status='completed';finish=choice.finish_reason
        if finish=='length':status='truncated_failure'
        content=choice.message.content or ''
        rec={'call_id':row['call_id'],'arm':row['arm'],'source_index':row['source_index'],'status':status,'finish_reason':finish,'response_text':content,
             'usage_prompt_tokens':pt,'usage_completion_tokens':ct,'usage_cached_prompt_tokens':cached,'actual_cost_usd':cost,'elapsed_seconds':time.time()-start,
             'model':response.model,'system_fingerprint':getattr(response,'system_fingerprint',None)}
        append_jsonl(CALLS,rec);spent+=cost;done+=1
        write_json(CHECKPOINT,{'status':'running' if spent<TASK1_CAP else 'hard_stop_reached','completed':done,'expected':900,'generation_attempt_spend_usd':round(GENERATION_SPEND,10),'evaluation_spend_usd':round(spent-GENERATION_SPEND,10),'task_api_spend_including_generation':round(spent,10),'hard_stop':TASK1_CAP,'last_call':row['call_id']})
        print(json.dumps({'call':row['call_id'],'status':status,'usage_prompt_tokens':pt,'usage_completion_tokens':ct,'spend_usd':round(cost,8),'task_total_usd':round(spent,8)}),flush=True)
        if spent>=TASK1_CAP:break
    return json.loads(CHECKPOINT.read_text())

def score()->dict[str,Any]:
    from scripts.lib.prof_feedback import QueryPlan
    from scripts.lib.prof_feedback_final import execute_final_plan
    from scripts.lib.evaluation_scorer import score_answer
    from results.extensions.run_extensions import write_json
    m,requests=verify();calls=read_jsonl(CALLS);byid={r['call_id']:r for r in calls};scored=[]
    ec=m['entity_fact_counts'];pc={tuple(k.split('\t',1)):int(v) for k,v in m['entity_property_fact_counts'].items()}
    for req in requests:
        rec=byid.get(req['call_id'],{'status':'not_run'});prediction=None;astatus='abstained'
        if rec.get('status')=='completed':
            if req['arm']=='aletheia':
                try:
                    obj=json.loads(rec['response_text']);plan=QueryPlan.from_object(obj['query_plan']);_,ans,_=execute_final_plan(req['paraphrased_question'],plan,req['retrieved_items'],ec,pc);prediction=ans.get('answer');astatus=ans.get('status','abstained')
                except (ValueError,KeyError,TypeError,json.JSONDecodeError):astatus='invalid_plan'
            else:prediction=rec.get('response_text');astatus='answered' if prediction else 'abstained'
        correct=bool(score_answer(req['intent'],prediction,req['gold'])) if rec.get('status')=='completed' else False
        scored.append({'call_id':req['call_id'],'arm':req['arm'],'intent':req['intent'],'flagged':req['flagged'],'source_index':req['source_index'],'question_id':req['source_question_id'],'question':req['paraphrased_question'],'original_question':req['original_question'],'gold':req['gold'],'prediction':prediction,'answer_status':astatus,'correct':correct,'call_status':rec.get('status')})
    def summ(rs):
        n=len(rs);return {'n':n,'correct':sum(x['correct'] for x in rs),'accuracy':sum(x['correct'] for x in rs)/n if n else None}
    summary={}
    for arm in ('aletheia','direct','cot'):
        rs=[x for x in scored if x['arm']==arm];summary[arm]={'overall':summ(rs),'overall_excluding_flagged':summ([x for x in rs if not x['flagged']])}
        for intent in ('historical','aggregation','boolean'):
            ir=[x for x in rs if x['intent']==intent];summary[arm][intent]=summ(ir);summary[arm][intent+'_excluding_flagged']=summ([x for x in ir if not x['flagged']])
    source_scored=read_json(BASE/'scored_calls.json');source_score={(r['source_index'],r['arm']):r for r in source_scored}
    selected=[x for x in read_jsonl(PARAPHRASES) if x['variant']=='a']
    mapped={arm:[] for arm in ('aletheia','direct','cot')}
    for x in selected:
        for arm in mapped:
            old=source_score.get((x['source_index'],arm))
            if old is None:continue
            mapped[arm].append({'intent':x['intent'],'flagged':x['flagged'],'correct':bool(old['correct'])})
    mapped_summary={arm:{'overall':summ(rs),'overall_excluding_flagged':summ([x for x in rs if not x['flagged']]),
                         **{it:summ([x for x in rs if x['intent']==it]) for it in ('historical','aggregation','boolean')},
                         **{it+'_excluding_flagged':summ([x for x in rs if x['intent']==it and not x['flagged']]) for it in ('historical','aggregation','boolean')}} for arm,rs in mapped.items()}
    paired_summary={}
    paraphrase_rows=read_jsonl(PARAPHRASES)
    matched_positions={f'{i:03d}' for i,row in enumerate(paraphrase_rows,1) if row['variant']=='a'}
    for arm in ('aletheia','direct','cot'):
        # Select exactly one predeclared variant for every unique original ID.
        paraphrased=[x for x in scored if x['arm']==arm and x['call_id'].split('-')[1] in matched_positions]
        paired_summary[arm]={'paraphrased_matched_ids':summ(paraphrased),'original_same_ids':mapped_summary[arm]}
        for intent in ('historical','aggregation','boolean'):
            paired_summary[arm][intent]={'paraphrased':summ([x for x in paraphrased if x['intent']==intent]),
                                         'original':mapped_summary[arm][intent],
                                         'paraphrased_excluding_flagged':summ([x for x in paraphrased if x['intent']==intent and not x['flagged']]),
                                         'original_excluding_flagged':mapped_summary[arm][intent+'_excluding_flagged']}
    total_api=sum(float(r.get('actual_cost_usd') or 0) for r in calls);pt=sum(int(r.get('usage_prompt_tokens') or 0) for r in calls);ct=sum(int(r.get('usage_completion_tokens') or 0) for r in calls)
    spend_by_arm={arm:round(sum(float(r.get('actual_cost_usd') or 0) for r in calls if r['arm']==arm),10) for arm in ('aletheia','direct','cot')}
    report={'frozen_manifest_sha256':m['frozen_manifest_sha256'],'paraphrases_sha256':m['paraphrases_sha256'],'matched_comparison_plan_sha256':sha_file(OUT/'matched_comparison_plan.json'),'task1_status':json.loads(CHECKPOINT.read_text()).get('status'),'paraphrase_generation_spend_usd':round(GENERATION_SPEND,10),'evaluation_spend_usd':round(total_api,10),'task1_total_api_spend_usd':round(GENERATION_SPEND+total_api,10),'spend_by_arm_usd':spend_by_arm,'prompt_tokens':pt,'completion_tokens':ct,'calls_logged':len(calls),'calls_expected':900,'truncated_failures':sum(r.get('status')=='truncated_failure' for r in calls),'results_all_300_paraphrases':summary,'matched_only_comparison_298_unique_source_ids':paired_summary}
    write_json(SCORED,scored);write_json(REPORT,report);return report

def main():
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true');g.add_argument('--score',action='store_true');a=ap.parse_args()
    if a.prepare:print(json.dumps(prepare(),indent=2))
    elif a.run:print(json.dumps(run(),indent=2))
    else:print(json.dumps(score(),indent=2))
if __name__=='__main__':main()
