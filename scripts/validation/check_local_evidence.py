"""Verify manuscript title, historical costs, latency records and frozen inputs."""

import argparse
import ast
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from decimal import Decimal
from pathlib import Path

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--require-local-caches', action='store_true')
args = parser.parse_args()


def read_json(relative):
    return json.loads((root / relative).read_text())


def rows(relative):
    return [json.loads(line) for line in (root / relative).read_text().splitlines() if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


title = 'Aletheia: A Semantic Routing Architecture with a Deterministic Operator Layer for Memory Resolution in Long-Context Language Models'
paper = root / 'paper/Aletheia_IEEE_Access_Updated.docx'
with zipfile.ZipFile(paper) as archive:
    document = ET.fromstring(archive.read('word/document.xml'))
namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
paragraphs = [''.join(text.text or '' for text in paragraph.findall('.//w:t', namespace))
              for paragraph in document.findall('w:body/w:p', namespace)]
assert next(text for text in paragraphs if text.strip()) == title
readme = (root / 'README.md').read_text()
assert title.removeprefix('Aletheia: ') in readme

cost_inputs = {
    'final_own_k': ('results/prof_feedback/router-final300-native-20261016/calls.jsonl', '0.040'),
    'final_k80': ('results/prof_feedback/router-final300-native-k80-20261016/calls.jsonl', '0.078'),
    'confirmatory': ('results/router_runs/pf-20261016-router-confirmatory-01/r2_calls.jsonl', '0.078'),
}
ledgers = {name: (rows(path), rounded) for name, (path, rounded) in cost_inputs.items()}
controls = rows('logs/prof_feedback/pf-20261005-final300-v3-01/calls.jsonl')
ledgers['direct'] = ([row for row in controls if row['arm'] == 'direct'], '0.059')
ledgers['cot'] = ([row for row in controls if row['arm'] == 'cot'], '0.071')
costs = {}
for name, (records, rounded) in ledgers.items():
    assert len(records) == 300 and len({row['question_id'] for row in records}) == 300
    assert all(row['status'] == 'completed' for row in records)
    recorded = sum(Decimal(str(row['actual_cost_usd'])) for row in records)
    calculated = sum((Decimal(row['usage_prompt_tokens']) * Decimal('0.15')
                      + Decimal(row['usage_completion_tokens']) * Decimal('0.60')) / 1000000
                     for row in records)
    assert abs(recorded - calculated) < Decimal('0.000000001')
    assert format(recorded, '.3f') == rounded
    costs[name] = {'calls': 300, 'recorded_usd': str(recorded),
                   'token_calculated_usd': str(calculated), 'paper_rounded_usd': rounded}

latency = (root / 'logs/latency/routing_latency_test.log').read_text()
expected_latency = {
    'Warmup call took': '395.50', 'Average Latency': '61.73',
    'Min Latency': '42.22', 'Max Latency': '259.32', 'Median Latency': '52.44',
}
for label, expected in expected_latency.items():
    assert re.search(re.escape(label) + r':?\s*' + re.escape(expected) + r' ms', latency)
assert 'Total Runs: 100' in latency
assert 'i7-8850H CPU @ 2.60GHz' in latency
assert 'Iteration 4' in latency
benchmark = ast.parse((root / 'benchmarks/latency/test_latency.py').read_text())
query_bank = next(node.value for node in benchmark.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'queries' for target in node.targets))
assert len(set(ast.literal_eval(query_bank))) == 4
assert any(isinstance(node, ast.BinOp)
           and isinstance(node.left, ast.Name) and node.left.id == 'queries'
           and isinstance(node.op, ast.Mult) and ast.literal_eval(node.right) == 25
           for node in ast.walk(benchmark))

confirm = read_json('results/router_runs/pf-20261016-router-confirmatory-01/manifest.json')
for relative, expected in confirm['input_file_sha256'].items():
    assert sha256(root / relative) == expected, relative
own = read_json('results/prof_feedback/router-final300-native-20261016/config_manifest.json')
dataset_revision = '7ea066982b140a19337e17e60d45d4076e042faf'
checkpoint = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
assert own['dataset_revision'] == dataset_revision
assert own['minilm_checkpoint_hash'] == checkpoint == confirm['model']['router_checkpoint_commit']

cache = Path.home() / '.cache/huggingface'
model_cache = cache / 'hub/models--sentence-transformers--all-MiniLM-L6-v2'
snapshot = model_cache / 'snapshots' / checkpoint
arrow = cache / 'datasets/ai-hyz___memory_agent_bench/default/0.0.0' / dataset_revision / 'memory_agent_bench-Conflict_Resolution.arrow'
cache_checks = {'external_caches_included_in_zip': False}
if snapshot.is_dir() and arrow.is_file():
    assert (model_cache / 'refs/main').read_text().strip() == checkpoint
    for relative, expected in confirm['model']['router_snapshot_files_sha256'].items():
        assert sha256(snapshot / relative) == expected, relative
    assert sha256(arrow) == own['dataset_arrow_sha256']
    cache_checks['status'] = 'local model snapshot and dataset bytes match frozen manifests'
    cache_checks['model_files_checked'] = len(confirm['model']['router_snapshot_files_sha256'])
else:
    assert not args.require_local_caches, 'Pinned external model/dataset caches are missing'
    cache_checks['status'] = 'external caches absent; pinned references and included inputs verified'

print(json.dumps({
    'status': 'passed', 'title': title, 'paper_sha256': sha256(paper), 'costs': costs,
    'latency': {'source': 'saved aggregate log plus benchmark script; per-query timing samples not retained',
                'mean_ms': '61.73', 'min_ms': '42.22', 'max_ms': '259.32',
                'warmup_ms': '395.50', 'timed_queries': 100, 'distinct_questions': 4},
    'included_manifest_input_hashes_checked': len(confirm['input_file_sha256']),
    'cache_checks': cache_checks,
}, indent=2))
