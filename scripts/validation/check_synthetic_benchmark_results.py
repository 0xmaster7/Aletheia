"""Check that the saved paper run matches the first 60 released questions."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
benchmark = json.loads((ROOT / "data" / "synthetic_benchmark.json").read_text(encoding="utf-8"))
results = json.loads((ROOT / "results" / "poc_results" / "synthetic_benchmark_results.json").read_text(encoding="utf-8"))

assert len(benchmark) == 13_425, f"Expected 13,425 generated questions, found {len(benchmark)}"
assert len(results) == 60, f"Expected 60 saved paper results, found {len(results)}"
first_60 = benchmark[:60]
intent_counts = Counter(item["intent"] for item in first_60)
assert intent_counts == {"historical": 20, "aggregation": 20, "boolean": 20}, intent_counts
for index, (question, result) in enumerate(zip(first_60, results), start=1):
    expected = (question["entity"], question["intent"], question["question_text"], question["ground_truth_answer"])
    actual = (result["entity"], result["intent"], result["question"], result["ground_truth"])
    assert expected == actual, f"Saved result row {index} does not match benchmark question {index}"

print("OK: 60 saved result rows match the first 60 questions; intent distribution is 20/20/20.")
