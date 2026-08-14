"""Count total conflicting entities in factconsolidation_sh_262k and compute batch math."""
import json, re, math, os
from collections import defaultdict
from datasets import load_dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_PATH = os.path.join(REPO_ROOT, "data", "synthetic_benchmark.json")
BATCH_SIZE = 40

# Already processed
with open(BENCHMARK_PATH, encoding="utf-8") as f:
    existing = json.load(f)
already_done = set(item["entity"] for item in existing)
print(f"Entities already in benchmark : {len(already_done)}")

# Load dataset
print("Loading dataset (cached)...")
ds = load_dataset("ai-hyz/MemoryAgentBench", split="Conflict_Resolution", revision="main")
row = next(s for s in ds if s["metadata"]["source"] == "factconsolidation_sh_262k")
ctx = row["context"]

# Parse facts
pat = re.compile(r"(\d+)\.\s")
matches = list(pat.finditer(ctx))
facts = []
for i, m in enumerate(matches):
    idx = int(m.group(1))
    s, e = m.end(), (matches[i+1].start() if i+1 < len(matches) else len(ctx))
    facts.append((idx, ctx[s:e].strip().rstrip(".")))

print(f"Total facts parsed            : {len(facts):,}")

# Extract entity facts
entity_facts = defaultdict(list)
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

for serial, text in facts:
    m = predicate_pattern.match(text)
    if m:
        entity_facts[m.group(1).strip()].append({"serial": serial, "value": m.group(2).strip()})
    else:
        m2 = fallback_pattern.match(text)
        if m2:
            entity_facts[m2.group(1).strip()].append({"serial": serial, "value": m2.group(4).strip()})

# Count conflicting entities
conflicting = {
    e: fl for e, fl in entity_facts.items()
    if len(set(f["value"] for f in fl)) >= 2 and len(fl) >= 2
}

total_conflicting  = len(conflicting)
already_processed  = len(already_done)
remaining          = total_conflicting - already_processed
full_batches       = remaining // BATCH_SIZE
leftover           = remaining % BATCH_SIZE
total_batches      = math.ceil(remaining / BATCH_SIZE)

print()
print("=" * 50)
print(f"Total conflicting entities    : {total_conflicting:,}")
print(f"Already processed             : {already_processed}")
print(f"Remaining                     : {remaining:,}")
print(f"Batch size                    : {BATCH_SIZE}")
print(f"Full batches of {BATCH_SIZE}          : {full_batches}")
print(f"Final partial batch           : {leftover} entities")
print(f"Total batches needed          : {total_batches}")
print("=" * 50)
print(f"  → {full_batches} full batches × {BATCH_SIZE} = {full_batches * BATCH_SIZE} entities")
if leftover:
    print(f"  → 1 partial batch           = {leftover} entities")
print(f"  → Grand total               = {full_batches * BATCH_SIZE + leftover} entities")
