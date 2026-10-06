"""Freeze the saved 100-item Boolean context list for the one Task 2b run."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/ood_boolean/task2b';BASE=ROOT/'results/prof_feedback/pf-20261006-hybrid300-final-01'
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return sha(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())
def main():
    reqpath=OUT/'requests.jsonl';manifest_path=OUT/'frozen_manifest.json'
    if reqpath.exists():raise SystemExit('Frozen request list already exists; refusing to replace it')
    base=[json.loads(x) for x in (BASE/'requests.jsonl').read_text().splitlines() if x.strip()]
    selected=[r for r in base if r['arm']=='aletheia' and r['intent']=='boolean']
    if len(selected)!=100:raise SystemExit(f'Expected 100 Boolean rows, got {len(selected)}')
    rows=[]
    for i,r in enumerate(selected,1):
        row={k:r[k] for k in ('question_id','source_index','entity','intent','question','ground_truth_answer','source_gold_flagged','source_gold_issue_codes','source_support_status','k','model','messages','response_format','temperature','max_completion_tokens','retrieved_items','local_input_tokens')}
        row['call_id']=f'boolfix-{i:03d}';rows.append(row)
    reqpath.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows))
    m=json.loads(manifest_path.read_text())
    m.update({'request_manifest_sha256':sha(reqpath.read_bytes()),'request_count':len(rows),'selected_question_ids':[r['question_id'] for r in rows],
              'code_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'branch':subprocess.check_output(['git','-C',str(ROOT),'branch','--show-current'],text=True).strip(),
              'request_note':'One fresh Aletheia planner call per each of the same 100 confirmatory Boolean questions, using the frozen final-v3 prompt and exact saved K=80 context; only the preregistered negation-field correction is applied after parsing.'})
    m.pop('manifest_sha256',None);m['manifest_sha256']=canon(m)
    manifest_path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'request_count':len(rows),'request_sha256':m['request_manifest_sha256'],'manifest_sha256':m['manifest_sha256']},indent=2))
if __name__=='__main__':main()
