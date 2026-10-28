import ast
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from scripts.lib.evaluation_scorer import score_answer


def load_rows(relative_path):
    path = root / relative_path
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


own = load_rows('results/prof_feedback/router-final300-native-20261016/scores_fixed.json')
shared = load_rows('results/prof_feedback/router-final300-native-k80-20261016/scores.json')
confirm = load_rows('results/router_runs/pf-20261016-router-confirmatory-01/r2_answers.jsonl')
metrics = {}
for name, rows, expected_correct, expected_answered in [
    ('final_own_k', own, 152, 251),
    ('final_k80', shared, 151, 253),
    ('confirmatory', confirm, 152, 242),
]:
    assert len(rows) == 300
    assert len({row['question_id'] for row in rows}) == 300
    for row in rows:
        answered = row['answered']
        truncated = row['finish_reason'] == 'length'
        computed = False if truncated else bool(score_answer(row['intent'], row['answer'] if answered else None, row['gold']))
        assert computed == row['correct'], (name, row['question_id'])
    correct = sum(row['correct'] for row in rows)
    answered = sum(row['answered'] for row in rows)
    assert (correct, answered) == (expected_correct, expected_answered)
    metrics[name] = {'n': 300, 'correct': correct, 'answered': answered, 'abstained': 300 - answered,
                     'truncated': sum(row['finish_reason'] == 'length' for row in rows),
                     'intent_distribution': dict(Counter(row['intent'] for row in rows))}

assert {row['question_id'] for row in own} == {row['question_id'] for row in shared}
assert not ({row['question_id'] for row in own} & {row['question_id'] for row in confirm})
assert sum(row['route'] == row['intent'] for row in own) == 240
assert sum(row['router_intent'] == row['intent'] for row in confirm) == 247

confirm_by_id = {row['question_id']: row for row in confirm}
hybrids = load_rows('results/router_runs/pf-20261016-router-confirmatory-01/r3_hybrid_scored_calls.jsonl')
for hybrid_name, expected in [('hybrid_direct_fallback', 193), ('hybrid_cot_fallback', 192)]:
    selected = [row for row in hybrids if row['hybrid'] == hybrid_name]
    assert len(selected) == 300
    for row in selected:
        source = confirm_by_id[row['question_id']]
        computed = bool(score_answer(row['intent'], row['prediction'] if row['answered'] else None, source['gold']))
        assert computed == row['correct'], row['question_id']
        assert row['fallback_used'] == (not source['answered'])
        if source['answered']:
            assert row['prediction'] == source['answer']
    assert sum(row['correct'] for row in selected) == expected
    metrics[hybrid_name] = {'n': 300, 'correct': expected}

fact_path = 'results/poc_results/ablation_adaptive_fact_gpt4omini_factconsolidation_sh_262k.json'
fact = load_rows(fact_path)
assert fact['correct'] == 81 and fact['n_questions'] == 100
metrics['factconsolidation_adaptive_262k'] = {'n': 100, 'correct': 81}

required = [
    'results/prof_feedback/pf-20261004-fresh300-v2-01/requests.jsonl',
    'results/prof_feedback/pf-20261004-fresh300-v2-01/scored_calls.json',
    'results/prof_feedback/pf-20261005-final300-v3-01/requests.jsonl',
    'results/prof_feedback/pf-20261005-final300-v3-01/scored_calls.json',
    'results/prof_feedback/pf-20261006-hybrid300-final-01/requests.jsonl',
    'results/prof_feedback/pf-20261006-hybrid300-final-01/scored_calls.json',
    'results/prof_feedback/pf-20261004-phaseA-05/gold_support.json',
    'results/prof_feedback/router-final300-native-20261016/calls.jsonl',
    'results/prof_feedback/router-final300-native-k80-20261016/calls.jsonl',
    'results/router_runs/pf-20261016-router-confirmatory-01/r2_calls.jsonl',
]
for relative in required:
    assert (root / relative).is_file(), relative
metrics['required_local_inputs_present'] = len(required)

pipeline = root / 'scripts/lib/_pipeline.py'
scorer = root / 'scripts/lib/evaluation_scorer.py'
pipeline_hash = hashlib.sha256(pipeline.read_bytes()).hexdigest()
scorer_hash = hashlib.sha256(scorer.read_bytes()).hexdigest()
native_folder = root / 'results/prof_feedback/router-final300-native-20261016'
shared_folder = root / 'results/prof_feedback/router-final300-native-k80-20261016'
for folder in [native_folder, shared_folder]:
    manifest = json.loads((folder / 'config_manifest.json').read_text())
    assert pipeline_hash == manifest['pipeline_sha256'], 'Pipeline differs from recorded experiment'
    assert scorer_hash == manifest['scorer_sha256'], 'Scorer differs from recorded experiment'
    for name, expected_hash in manifest.get('saved_artifact_hashes', {}).items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected_hash, name
manifest = json.loads((shared_folder / 'config_manifest.json').read_text())
tree = ast.parse(pipeline.read_text())
prompt_nodes = [node for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == 'CANDIDATE_PROMPT' for target in node.targets)]
assert len(prompt_nodes) == 1
prompt = ast.literal_eval(prompt_nodes[0].value)
assert hashlib.sha256(prompt.encode()).hexdigest() == manifest['candidate_prompt_sha256']
assert (shared_folder / 'prompt_template.txt').read_text() == prompt
metrics['frozen_pipeline_scorer_prompt_and_artifact_hashes'] = 'matched'

print(json.dumps({'status': 'passed', 'checks': metrics}, indent=2))
