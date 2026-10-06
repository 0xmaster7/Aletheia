"""Build the frozen 300-question paraphrase set from audited source semantics."""
import json,re,hashlib
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.lib.prof_feedback import parse_fact_template, property_type_from_fact, normalize_value
from results.extensions.run_extensions import load_corpus
HERE=Path(__file__).resolve().parent
BANNED=re.compile(r'\b(?:first|earliest|initial|oldest|original|previous|prior|earlier|before|beginning|inaugural|founding|ancestral|primordial|genesis|foundational|debut|preliminary|embryonic|seminal|initially)\b',re.I)

def relation_label(value):
    if value=='value': return 'value'
    return value.replace('_',' ')

def historical_q(entity, prop, rank, i):
    r=relation_label(prop)
    ranktext=str(rank)
    templates=[
      f"For {entity}, return the {r} value on the record at position {ranktext} after distinct states are ordered by decreasing serial.",
      f"Read {entity}'s records in descending serial order, collapsing adjacent repeated values; which {r} value belongs to position {ranktext}?",
      f"Locate position {ranktext} in {entity}'s sequence of distinct states sorted from highest serial to lowest, and report its {r} value.",
      f"Under a descending-serial ordering of {entity}'s distinct {r} states, identify the value attached to rank {ranktext}.",
      f"After merging consecutive repeats in {entity}'s serial-ordered history, what {r} value is stored at slot {ranktext}, counting down from the highest serial?",
    ]
    return templates[i%len(templates)]

def aggregation_q(entity, prop, i):
    r=relation_label(prop)
    if prop=='value':
        templates=[
          f"State the cardinality of the distinct values recorded for {entity}.",
          f"For {entity}, give the size of the set of unique recorded values.",
          f"How many nonduplicate value entries belong to {entity}'s records?",
          f"Return the number of distinct values associated with {entity} across its records.",
          f"Count the different values attached to {entity} in the knowledge store.",
        ]
    else:
        templates=[
          f"State the cardinality of the distinct {r} values recorded for {entity}.",
          f"For {entity}, give the size of the set of unique {r} entries.",
          f"How many nonduplicate {r} values belong to {entity}'s records?",
          f"Return the number of distinct {r} values associated with {entity}.",
          f"Count the different {r} values attached to {entity} in the knowledge store.",
        ]
    return templates[i%len(templates)]

def boolean_q(entity, prop, target, i):
    r='value' if prop=='value' else prop.replace('_',' ')
    templates=[
      f"For {entity}, does the database contain a {r} record with value {target}?",
      f"Check {entity}'s stored facts: is {target} listed as a {r} value?",
      f"Is there a {r} entry in the records for {entity} that names {target}?",
      f"Does the knowledge store link {target} to {entity} through the {r} relation?",
      f"Inspect {entity}'s records and determine whether a {r} fact assigns the value {target}.",
    ]
    return templates[i%len(templates)]

def audited_boolean_target(audit_row, source_row):
    if audit_row['boolean_target']:
        return audit_row['boolean_target']
    question=source_row['original_question']
    entity=source_row['entity']
    prop=audit_row['question_property_type'].replace('_',' ')
    prefix=f"Was the {prop} "
    suffix=f" ever a documented fact about {entity}?"
    if question.startswith(prefix) and question.endswith(suffix):
        return question[len(prefix):-len(suffix)]
    e=re.escape(entity); p=re.escape(prop)
    patterns=[
      rf"Is it accurate that {e} was at some point recorded under the {p} of (.+?)\?",
      rf"Can you verify whether (.+?) was ever documented as the {p} for {e}\?",
      rf"In the knowledge base, does {e} have any association with (.+?) as a {p}\?",
      rf"Confirm or deny: the {p} (.+?) appears in the historical records of {e}\.",
      rf"Was (.+?) ever registered as a valid {p} for {e} at any point\?",
      rf"Has {e} at any recorded instance been linked to the {p} (.+?)\?",
      rf"Is there any factual basis for {e} having the {p} (.+?)\?",
      rf"At any juncture in the records, was (.+?) the documented {p} for {e}\?",
      rf"Can it be substantiated that {e} carried the {p} designation of (.+?)\?",
      rf"Is it the case that {e} was ever recorded under the {p} of (.+?)\?",
      rf"Does the database contain any record of {e} having the {p} of (.+?)\?",
      rf"Is there documentation showing {e} was associated with the {p} (.+?)\?",
      rf"Was (.+?) at any point logged as the {p} for {e}\?",
      rf"Has (.+?) been attributed as the {p} for {e} in any record\?",
      rf"Did {e} ever possess the {p} designation of (.+?)\?",
      rf"According to the facts, was {e} ever linked to the {p} (.+?)\?",
    ]
    for pattern in patterns:
        match=re.fullmatch(pattern,question)
        if match: return match.group(1)
    prefix="Can you confirm that "
    suffix=f" appeared as a {prop} entry for {entity}?"
    if question.startswith(prefix) and question.endswith(suffix):
        return question[len(prefix):-len(suffix)]
    prefix="Is it verifiable that "
    suffix=f" was the {prop} for {entity} at any time?"
    if question.startswith(prefix) and question.endswith(suffix):
        return question[len(prefix):-len(suffix)]
    prefix="Does the evidence support that "
    suffix=f" served as the {prop} for {entity}?"
    if question.startswith(prefix) and question.endswith(suffix):
        return question[len(prefix):-len(suffix)]
    raise SystemExit(f"Cannot extract target from saved Boolean wording: {source_row['source_question_id']}")

def main():
    out=HERE/'paraphrases.jsonl'
    if out.exists(): raise SystemExit('Frozen paraphrase file exists; refusing to overwrite.')
    selection=json.loads((HERE/'selection.json').read_text())
    audit=json.loads(Path('results/prof_feedback/pf-20261004-phaseA-05/gold_support.json').read_text())
    aud={x['question_id']:x for x in audit}
    facts,context_sha,arrow_sha=load_corpus()
    by_entity={}
    for f in facts:
        p=parse_fact_template(f['text'])
        if p:
            by_entity.setdefault(p[0],[]).append({'serial':f['fact_idx'],'text':f['text'],'value':p[1],'property':property_type_from_fact(f['text'])})
    final=[]
    for intent in ('historical','aggregation','boolean'):
      for i,s in enumerate(selection['groups'][intent]):
        a=aud[s['source_question_id']]
        assert (a['source_index']==s['source_index'] and a['intent']==intent
                and a['question']==s['original_question']
                and a['ground_truth_answer']==s['gold'])
        prop=a['question_property_type']
        if intent=='historical':
            scoped=[x for x in by_entity.get(s['entity'],[]) if prop=='value' or x['property']==prop]
            # reverse serial order; collapse adjacent repeated normalized values exactly as the final operator does
            timeline=[]; seen=None
            for x in sorted(scoped,key=lambda x:x['serial'],reverse=True):
                norm=normalize_value(x['value'])
                if norm and norm!=seen:
                    timeline.append(x); seen=norm
            gold=normalize_value(s['gold'])
            pos=next((j for j,x in enumerate(timeline) if normalize_value(x['value'])==gold and x['serial'] in a['support_serials']),None)
            if pos is None:
                pos=next((j for j,x in enumerate(timeline) if normalize_value(x['value'])==gold),None)
            if pos is None: raise SystemExit(f"No serial rank for historical source {s['source_question_id']}")
            # Verify the selected serial is part of audit support; do not infer/relabel the gold.
            if not any(normalize_value(x['value'])==gold and x['serial'] in a['support_serials'] for x in timeline):
                raise SystemExit(f"Historical support serial/value mismatch: {s['source_question_id']}")
            question=historical_q(s['entity'],prop,pos+1,i)
            rank=pos+1
        elif intent=='aggregation':
            question=aggregation_q(s['entity'],prop,i); rank=None
        else:
            target=audited_boolean_target(a,s)
            question=boolean_q(s['entity'],prop,target,i); rank=None
        cue_check=question.replace(s['entity'],' ')
        if BANNED.search(cue_check): raise SystemExit(f"Banned cue in constructed question {s['paraphrase_id']}: {question}")
        if question==s['original_question']: raise SystemExit(f"Paraphrase equals source: {s['paraphrase_id']}")
        final.append({k:s[k] for k in ('paraphrase_id','intent','source_index','source_question_id','original_question','entity','gold','flagged','source_gold_issue_codes','source_support_status','variant')} | {
          'paraphrased_question':question,'audited_property_type':prop,'temporal_rank_desc_serial':rank,
          'audit_support_serials':a['support_serials'],'paraphrase_method':'fixed structural rewrite from audit semantics; no answer text supplied to a model'
        })
    assert len(final)==300
    assert [sum(x['intent']==t for x in final) for t in ('historical','aggregation','boolean')]==[100,100,100]
    out.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in final))
    (HERE/'paraphrase_build_metadata.json').write_text(json.dumps({'n':len(final),'intent_counts':{'historical':100,'aggregation':100,'boolean':100},'source_selection_sha256':hashlib.sha256((HERE/'selection.json').read_bytes()).hexdigest(),'source_audit':'results/prof_feedback/pf-20261004-phaseA-05/gold_support.json','source_audit_sha256':hashlib.sha256(Path('results/prof_feedback/pf-20261004-phaseA-05/gold_support.json').read_bytes()).hexdigest(),'corpus_context_sha256':context_sha,'arrow_sha256':arrow_sha,'paraphrase_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'method':'deterministic structural templates, chosen by fixed index cycle; historical requested rank derived from audited support serial and source corpus; all gold labels copied unchanged; no model generation used for frozen final paraphrases'},indent=2)+'\n')
    print(json.dumps({'n':len(final),'intent_counts':{'historical':100,'aggregation':100,'boolean':100},'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'history_rank_distribution':{str(i):sum(x['temporal_rank_desc_serial']==i for x in final if x['intent']=='historical') for i in sorted({x['temporal_rank_desc_serial'] for x in final if x['intent']=='historical'})}},indent=2))
if __name__=='__main__': main()
