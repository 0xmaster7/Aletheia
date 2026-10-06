"""Generate the fixed paraphrase set once; makes three logged API calls."""
import json, re, sys, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from results.extensions.run_extensions import client_or_raise, read_jsonl, append_jsonl, write_json, amount_from_usage, compute_input_tokens, sha256_file
MODEL='gpt-4o-mini'; TEMP=0.0; MAX_OUTPUT=8192; LIMIT=0.40
RATES={'input':0.15,'cached_input':0.075,'output':0.60}
BANNED=re.compile(r'\b(?:first|earliest|initial|oldest|original|previous|prior|earlier|before|beginning|inaugural|founding|ancestral|primordial|genesis|foundational|debut|preliminary|embryonic|seminal|initially)\b',re.I)
def cost(usage):
 d=usage.model_dump(exclude_none=True) if hasattr(usage,'model_dump') else dict(usage)
 inp=int(d['prompt_tokens']); out=int(d['completion_tokens'])
 details=d.get('prompt_tokens_details') or {}; cached=int(details.get('cached_tokens') or 0)
 cached=min(cached,inp)
 return round(((inp-cached)*RATES['input']+cached*RATES['cached_input']+out*RATES['output'])/1_000_000,10), d

def main():
 callfile=HERE/'paraphrase_generation_calls_final.jsonl'
 output=HERE/'paraphrases.jsonl'
 if callfile.exists() or output.exists(): raise SystemExit('Generation ledger or paraphrase file exists; refusing to call again or overwrite.')
 policy=(HERE/'generation_policy.txt').read_text()
 selection=json.loads((HERE/'selection.json').read_text())
 client=client_or_raise(); all_items=[]; spend=0.0
 schema={'type':'object','additionalProperties':False,'required':['paraphrases'],'properties':{'paraphrases':{'type':'array','items':{'type':'object','additionalProperties':False,'required':['paraphrase_id','question'],'properties':{'paraphrase_id':{'type':'string'},'question':{'type':'string'}}}}}}
 for intent in ('historical','aggregation','boolean'):
  inputs=json.loads((HERE/f'generation_inputs_{intent}.json').read_text())
  messages=[{'role':'system','content':policy},{'role':'user','content':json.dumps(inputs,ensure_ascii=False)}]
  body={'model':MODEL,'messages':messages,'temperature':TEMP,'max_completion_tokens':MAX_OUTPUT,
        'response_format':{'type':'json_schema','json_schema':{'name':f'{intent}_paraphrases','strict':True,'schema':schema}}}
  local_tokens=compute_input_tokens({'messages':messages,'response_format':body['response_format']})
  upper=(local_tokens*RATES['input']+MAX_OUTPUT*RATES['output'])/1e6
  if spend+upper>LIMIT: raise SystemExit(f'Budget guard stopped before {intent} paraphrase call; no call sent.')
  try: response=client.chat.completions.create(**body)
  except Exception as e:
   append_jsonl(callfile,{'phase':'paraphrase_generation','intent':intent,'status':'request_error_no_retry','error_type':type(e).__name__,'local_input_tokens':local_tokens,'messages':messages,'request_body':{k:v for k,v in body.items() if k!='messages'}})
   raise SystemExit(f'{intent} paraphrase call failed; no retry: {type(e).__name__}')
  if response.usage is None:
   append_jsonl(callfile,{'phase':'paraphrase_generation','intent':intent,'status':'usage_missing_no_retry','local_input_tokens':local_tokens,'messages':messages,'raw_response':response.choices[0].message.content or ''})
   raise SystemExit(f'{intent} paraphrase call missing usage; no retry')
  actual,usage=cost(response.usage); spend+=actual
  raw=response.choices[0].message.content or ''
  rec={'phase':'paraphrase_generation','intent':intent,'status':'completed','model':MODEL,'temperature':TEMP,'max_completion_tokens':MAX_OUTPUT,'request_id':getattr(response,'_request_id',None),'usage':usage,'actual_cost_usd':actual,'task1_cumulative_spend_usd':round(spend,10),'local_input_tokens_o200k_base':local_tokens,'messages':messages,'response_format':body['response_format'],'response_text':raw,'finish_reason':response.choices[0].finish_reason}
  append_jsonl(callfile,rec)
  if response.choices[0].finish_reason=='length': raise SystemExit(f'{intent} paraphrases truncated; no retry or evaluation.')
  try: out=json.loads(raw)['paraphrases']
  except Exception: raise SystemExit(f'{intent} response did not parse; no retry or evaluation.')
  expected={i['paraphrase_id'] for i in inputs['items']}; got=[x.get('paraphrase_id') for x in out]
  if len(out)!=100 or set(got)!=expected or len(set(got))!=100: raise SystemExit(f'{intent} did not return exactly 100 unique requested ids; no retry or evaluation.')
  byid={x['paraphrase_id']:x['question'].strip() for x in out}
  source_byid={x['paraphrase_id']:x for x in selection['groups'][intent]}
  for pid in [i['paraphrase_id'] for i in inputs['items']]:
   q=byid[pid]
   if not q or BANNED.search(q): raise SystemExit(f'{pid}: empty or contains a forbidden temporal cue; saved generation retained, no evaluation.')
   item=source_byid[pid]
   all_items.append({k:item[k] for k in ('paraphrase_id','intent','source_index','source_question_id','original_question','entity','gold','flagged','source_gold_issue_codes','source_support_status','variant')} | {'paraphrased_question':q})
  if spend>=LIMIT: raise SystemExit('Task 1 $0.40 hard stop reached after paraphrase generation; no further calls.')
 output.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in all_items))
 print(json.dumps({'paraphrase_count':len(all_items),'per_intent':{t:sum(x['intent']==t for x in all_items) for t in ('historical','aggregation','boolean')},'generation_spend_usd':round(spend,10),'paraphrase_sha256':sha256_file(output)},indent=2))
if __name__=='__main__': main()
