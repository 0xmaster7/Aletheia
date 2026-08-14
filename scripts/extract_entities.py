"""Extract conflicting entities from cached parquet file."""
import json
import os
import re
import sys
from collections import defaultdict
import pyarrow.parquet as pq

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = "/Users/keshavnanda/.cache/huggingface/hub/datasets--ai-hyz--MemoryAgentBench/snapshots/7ea066982b140a19337e17e60d45d4076e042faf/data/Conflict_Resolution-00000-of-00001.parquet"

print("Loading parquet...")
table = pq.read_table(PARQUET)
df = table.to_pydict()

# Find the factconsolidation_sh_262k row
ctx = None
for i in range(len(df["context"])):
    meta = json.loads(df["metadata"][i]) if isinstance(df["metadata"][i], str) else df["metadata"][i]
    if meta.get("source") == "factconsolidation_sh_262k":
        ctx = df["context"][i]
        break

if not ctx:
    print("ERROR: Could not find factconsolidation_sh_262k row")
    sys.exit(1)

# Parse facts
pat = re.compile(r"(\d+)\.\s")
matches = list(pat.finditer(ctx))
facts = []
for i, m in enumerate(matches):
    idx = int(m.group(1))
    s = m.end()
    e = matches[i + 1].start() if i + 1 < len(matches) else len(ctx)
    text = ctx[s:e].strip().rstrip(".")
    facts.append((idx, text))
print(f"Parsed {len(facts)} facts.")

# Extract entities
predicate_pattern = re.compile(
    r"^(.+?)\s+(?:was created in the country of|has the genre of|was located in|"
    r"is located in|was born in|has the nationality of|was created by|"
    r"has the capital of|was written in|has the official language of|"
    r"is the capital of|was founded in|has the population of|"
    r"was produced by|has the religion of|was directed by|"
    r"was published in|was released in|has the ethnicity of|"
    r"was performed by|is a member of|has the occupation of|"
    r"was invented by|is the currency of|was composed by|"
    r"was discovered by|is in the continent of|has the language of|"
    r"was designed by|is the leader of|has the currency of|"
    r"was manufactured by|is associated with|has the owner of|"
    r"is the owner of|was owned by|is owned by)\s+(.+)$",
    re.IGNORECASE,
)
fallback_pattern = re.compile(
    r"^(.+?)\s+(was|is|has|had)\s+(.+?)\s+(?:of|in|by|from|at|to)\s+(.+)$",
    re.IGNORECASE,
)

entity_facts = defaultdict(list)
for serial, text in facts:
    m = predicate_pattern.match(text)
    if m:
        entity_facts[m.group(1).strip()].append({"serial": serial, "text": text, "value": m.group(2).strip()})
    else:
        m2 = fallback_pattern.match(text)
        if m2:
            entity_facts[m2.group(1).strip()].append({"serial": serial, "text": text, "value": m2.group(4).strip()})

# Filter for conflicts (>= 2 distinct values)
conflicting = {}
for entity, flist in entity_facts.items():
    distinct = set(f["value"] for f in flist)
    if len(distinct) >= 2:
        sorted_facts = sorted(flist, key=lambda x: x["serial"])
        conflicting[entity] = {
            "facts": sorted_facts,
            "distinct_values": list(distinct),
            "n_distinct": len(distinct),
        }

sorted_all = sorted(conflicting.items(), key=lambda x: (x[1]["n_distinct"], len(x[1]["facts"])), reverse=True)

# Load existing benchmark
benchmark_path = os.path.join(PROJECT_ROOT, "data", "synthetic_benchmark.json")
already_done = set()
if os.path.exists(benchmark_path):
    with open(benchmark_path, "r") as f:
        existing = json.load(f)
    already_done = set(item["entity"] for item in existing)
    print(f"Already processed: {len(already_done)} entities.")

remaining = [(e, info) for e, info in sorted_all if e not in already_done]
print(f"\nTotal conflicting: {len(sorted_all)}")
print(f"Remaining: {len(remaining)}")
print(f"\n--- Next 40 entities ---\n")

batch = remaining[:40]
output = []
for i, (entity, info) in enumerate(batch):
    oldest = info["facts"][0]
    newest = info["facts"][-1]
    lower = oldest["text"].lower()
    if "country" in lower: pred = "country"
    elif "genre" in lower: pred = "genre"
    elif "located" in lower or "capital" in lower: pred = "location"
    elif "language" in lower: pred = "language"
    elif "religion" in lower: pred = "religion"
    elif "nationality" in lower: pred = "nationality"
    elif "ethnicity" in lower: pred = "ethnicity"
    elif "occupation" in lower: pred = "occupation"
    elif "created by" in lower or "directed by" in lower: pred = "creator"
    else: pred = "value"

    entry = {
        "entity": entity,
        "predicate": pred,
        "n_distinct": info["n_distinct"],
        "oldest_serial": oldest["serial"],
        "oldest_value": oldest["value"],
        "newest_serial": newest["serial"],
        "newest_value": newest["value"],
        "all_values": info["distinct_values"],
    }
    output.append(entry)
    print(f"{i+1:2d}. {entity}")
    print(f"    pred={pred}, n_distinct={info['n_distinct']}")
    print(f"    oldest(s{oldest['serial']}): {oldest['value']}")
    print(f"    newest(s{newest['serial']}): {newest['value']}")
    print(f"    all: {info['distinct_values']}")

extract_path = os.path.join(PROJECT_ROOT, "data", "entity_extract_batch1.json")
with open(extract_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved to: {extract_path}")
