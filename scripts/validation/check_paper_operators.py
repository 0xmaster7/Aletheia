"""Check actual paper operators without initializing models or calling APIs."""

import ast
import contextlib
import io
import json
import re
from pathlib import Path
from types import SimpleNamespace

root = Path(__file__).resolve().parents[2]
path = root / 'scripts/lib/_pipeline.py'
tree = ast.parse(path.read_text())
function_names = {
    'tokenize', '_normalize_text', '_freshness_pick', '_collapse_state_timeline',
    '_temporal_offset', '_historical_pick', '_question_mentions_phrase',
    '_boolean_gate', '_aggregate_answers', 'run_adaptive_router_pipeline',
}
constant_names = {'TEMPORAL_OLDEST_PATTERNS', 'ORDINAL_WORD_TO_OFFSET', 'NEGATION_HINTS'}
selected = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in function_names:
        node.decorator_list = []
        selected.append(node)
    elif isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in constant_names for target in node.targets
    ):
        selected.append(node)
assert {node.name for node in selected if isinstance(node, ast.FunctionDef)} == function_names
compiled = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *selected], type_ignores=[])
namespace = {
    're': re, 'json': json, 'TOP_K': 10, 'MODEL': 'gpt-4o-mini',
    'get_client': lambda: SimpleNamespace(update_current_span=lambda **kwargs: None),
    'evaluate_answer': lambda answer, gold: {'is_correct_subem': False},
}
exec(compile(ast.fix_missing_locations(compiled), str(path), 'exec'), namespace)

checks = 0


def run(route, question, candidates, expected_answer, expected_k):
    global checks
    retrieval = {}

    def retrieve(bm25, query, indices, facts, top_k):
        retrieval.update(query=query, top_k=top_k)
        return []

    namespace['native_route'] = lambda query: route
    namespace['bm25_retrieve'] = retrieve
    namespace['_extract_candidates'] = lambda client, query, retrieved: candidates
    with contextlib.redirect_stdout(io.StringIO()):
        result = namespace['run_adaptive_router_pipeline'](
            question, 0, [], None, list(range(100)), ['fact'] * 100, None,
        )
    assert result['answer'] == expected_answer, (route, question, result['answer'], expected_answer)
    assert retrieval['top_k'] == expected_k
    checks += 1
    return retrieval


cities = [
    {'serial': 1, 'answer_entity': 'Chennai', 'fact_text': 'Alex lives in Chennai.'},
    {'serial': 2, 'answer_entity': 'Delhi', 'fact_text': 'Alex lives in Delhi.'},
    {'serial': 3, 'answer_entity': 'Vellore', 'fact_text': 'Alex lives in Vellore.'},
    {'serial': 4, 'answer_entity': 'DELHI.', 'fact_text': 'Alex returns to Delhi.'},
]
run('current_value', 'Where does Alex live now?', cities, 'DELHI.', 10)
run('historical', 'What was the first city for Alex?', cities, 'Chennai', 25)
run('historical', 'Where did Alex live previously before now?', cities, 'Vellore', 25)
run('historical', 'What was the third previous city for Alex?', cities, 'Delhi', 25)
run('historical', 'What was the fifth previous city for Alex?', cities, 'Chennai', 25)
run('historical', 'City history for Alex', cities, 'Chennai', 25)
run('historical', 'What was the second city Alex lived in?', cities, 'Chennai', 25)
run('aggregation', 'How many distinct cities for Alex?', cities, '3', 40)
run('aggregation', 'List all cities for Alex', cities, json.dumps(['Chennai', 'Delhi', 'Vellore']), 40)
run('boolean', 'Is Delhi the current city for Alex?', cities, 'True', 10)
run('boolean', 'Is Chennai the current city for Alex?', cities, 'False', 10)
run('boolean', 'Was Chennai ever a city for Alex?', cities, 'True', 10)
run('boolean', 'Is Alex not in Delhi?', cities, 'False', 10)
run('boolean', 'Is Alex not in London?', cities, 'True', 10)
run('boolean', "Isn't Alex in Delhi?", cities, 'True', 10)
aliases = [
    {'serial': 1, 'answer_entity': 'NYC', 'fact_text': 'Alex lives in NYC.'},
    {'serial': 2, 'answer_entity': 'New York City', 'fact_text': 'Alex lives in New York City.'},
]
run('aggregation', 'How many distinct cities for Alex?', aliases, '2', 40)
repeated = cities + [dict(cities[-1], serial=5, answer_entity='Delhi')]
run('historical', 'Where did Alex live before now?', repeated, 'Vellore', 25)

for route, question, expected_answer, expected_k in [
    ('current_value', 'Where now?', '(no answer)', 10),
    ('historical', 'Where first?', '(no answer)', 25),
    ('aggregation', 'How many cities?', '0', 40),
    ('aggregation', 'List cities', '[]', 40),
    ('boolean', 'Was London ever recorded?', 'False', 10),
]:
    run(route, question, [], expected_answer, expected_k)

cleaned = run('historical', 'What was the initial value recorded for Alex?', cities, 'Chennai', 25)
assert cleaned['query'] == 'Alex?'
fallback = run('historical', 'what was the first value', cities, 'Chennai', 25)
assert fallback['query'] == 'what was the first value'
print(f'OK: {checks} paper operator/retrieval-boundary cases; no model initialization or API calls.')
