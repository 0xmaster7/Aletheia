"""Quick diagnostic: check how the router classifies the original dataset's questions.
No LLM calls — just local embedding similarity. Zero cost.
"""
import json, sys, re
from collections import Counter
from datasets import load_dataset

sys.path.insert(0, '.')
from _pipeline import native_route

ds = load_dataset("ai-hyz/MemoryAgentBench", split="Conflict_Resolution", revision="main")
row = next(s for s in ds if s["metadata"]["source"] == "factconsolidation_sh_262k")
questions = row["questions"][:100]

routes = []
for i, q in enumerate(questions):
    r = native_route(q)
    routes.append(r)
    if r != "current_value":
        print(f"  Q{i}: [{r:>13}] {q[:80]}")

dist = Counter(routes)
print(f"\n{'='*50}")
print(f"Route distribution for original 100 questions:")
for route, count in dist.most_common():
    print(f"  {route:>15}: {count}")
print(f"\nMisrouted (not current_value): {100 - dist.get('current_value', 0)}")
