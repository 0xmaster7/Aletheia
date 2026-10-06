"""Select fixed source questions/slots for the 300-item paraphrase set (offline)."""
import json, random
from pathlib import Path
root=Path(__file__).resolve().parent
base=root.parents[1]/'prof_feedback/pf-20261006-hybrid300-final-01'
requests=[json.loads(x) for x in (base/'requests.jsonl').read_text().splitlines() if x]
source=[r for r in requests if r['arm']=='aletheia']
assert len(source)==300
rng=random.Random(20261007)
by={t:sorted([r for r in source if r['intent']==t],key=lambda x:x['source_index']) for t in ('historical','aggregation','boolean')}
assert [len(by[t]) for t in by]==[98,102,100]
duplicates={r['source_index'] for r in rng.sample(by['historical'],2)}
agg_selected={r['source_index'] for r in rng.sample(by['aggregation'],100)}
slots={}
for intent,rows in by.items():
 chosen=rows if intent!='aggregation' else [r for r in rows if r['source_index'] in agg_selected]
 intent_slots=[]
 for row in chosen:
  item={'intent':intent,'source_index':row['source_index'],'source_question_id':row['question_id'],'original_question':row['question'],
        'entity':row['entity'],'gold':row['ground_truth_answer'],'flagged':row['source_gold_flagged'],
        'source_gold_issue_codes':row.get('source_gold_issue_codes',[]),'source_support_status':row.get('source_support_status')}
  item['variant']='a'; intent_slots.append(item)
  if intent=='historical' and row['source_index'] in duplicates:
   intent_slots.append({**item,'variant':'b'})
 for i,item in enumerate(intent_slots,1): item['paraphrase_id']=f'{intent}-{i:03d}'
 assert len(intent_slots)==100
 slots[intent]=intent_slots
(root/'selection.json').write_text(json.dumps({'seed':20261007,'selection_policy':'All 98 historical source questions plus a second distinct phrasing for two seeded random historical sources; 100 seeded random aggregation sources from 102; all 100 Boolean sources.','duplicates_historical_source_indices':sorted(duplicates),'omitted_aggregation_source_indices':sorted(set(r['source_index'] for r in by['aggregation'])-agg_selected),'groups':slots},ensure_ascii=False,indent=2)+'\n')
for intent,items in slots.items():
 p=root/f'generation_inputs_{intent}.json'
 p.write_text(json.dumps({'intent':intent,'items':[{'paraphrase_id':x['paraphrase_id'],'source_question_id':x['source_question_id'],'original_question':x['original_question']} for x in items]},ensure_ascii=False,indent=2)+'\n')
print('historical/aggregation/boolean:',*[len(slots[t]) for t in slots])
print('duplicate historical source indices:', sorted(duplicates))
print('omitted aggregation source indices:', sorted(set(r['source_index'] for r in by['aggregation'])-agg_selected))
