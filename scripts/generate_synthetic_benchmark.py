"""Generate a 20-entity synthetic benchmark for semantic routing stress-testing.

Reads the MemoryAgentBench SH-262K dataset, identifies entities with multiple
conflicting facts, and generates adversarial historical/aggregation/boolean
questions for each.

Usage:
    python scripts/generate_synthetic_benchmark.py
"""
from __future__ import annotations
import json
import re
import sys
from collections import defaultdict
from datasets import load_dataset

# ── Load dataset ──────────────────────────────────────────────────────────
print("Loading MemoryAgentBench dataset...")
ds = load_dataset("ai-hyz/MemoryAgentBench", split="Conflict_Resolution", revision="main")
row = next(s for s in ds if s["metadata"]["source"] == "factconsolidation_sh_262k")
ctx = row["context"]

# ── Parse facts ───────────────────────────────────────────────────────────
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

# ── Extract entity -> list of (serial, fact_text, value) ──────────────────
# Facts follow patterns like:
#   "X was created in the country of Y"
#   "X has the genre of Y"
#   "X was located in Y"
entity_facts = defaultdict(list)
# Pattern: "<Entity> <predicate> <value>"
# We'll extract the subject entity and the object value
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
    re.IGNORECASE
)

# Broader fallback: anything with "was/is/has ... of/in/by ..."
fallback_pattern = re.compile(
    r"^(.+?)\s+(was|is|has|had)\s+(.+?)\s+(?:of|in|by|from|at|to)\s+(.+)$",
    re.IGNORECASE
)

for serial, text in facts:
    m = predicate_pattern.match(text)
    if m:
        entity = m.group(1).strip()
        value = m.group(2).strip()
        entity_facts[entity].append({"serial": serial, "text": text, "value": value})
    else:
        m2 = fallback_pattern.match(text)
        if m2:
            entity = m2.group(1).strip()
            value = m2.group(4).strip()
            entity_facts[entity].append({"serial": serial, "text": text, "value": value})

# ── Find entities with multiple DISTINCT values (conflicts) ───────────────
conflicting_entities = {}
for entity, fact_list in entity_facts.items():
    distinct_values = set(f["value"] for f in fact_list)
    if len(distinct_values) >= 2 and len(fact_list) >= 2:
        conflicting_entities[entity] = {
            "facts": sorted(fact_list, key=lambda x: x["serial"]),
            "distinct_values": distinct_values,
            "n_distinct": len(distinct_values),
            "n_facts": len(fact_list),
        }

# Sort by number of distinct values (most conflicting first) and take top 20
sorted_entities = sorted(
    conflicting_entities.items(),
    key=lambda x: (x[1]["n_distinct"], x[1]["n_facts"]),
    reverse=True
)[:20]

print(f"\nFound {len(conflicting_entities)} entities with conflicting facts.")
print(f"Selected top 20 entities for benchmark generation.\n")

# ── Print selected entities for inspection ─────────────────────────────────
for i, (entity, info) in enumerate(sorted_entities):
    print(f"  {i+1:2d}. {entity} — {info['n_distinct']} distinct values, {info['n_facts']} facts")
    oldest = info["facts"][0]
    newest = info["facts"][-1]
    print(f"      Oldest (serial {oldest['serial']}): {oldest['value']}")
    print(f"      Newest (serial {newest['serial']}): {newest['value']}")

# ── Generate adversarial questions ─────────────────────────────────────────
benchmark = []

# Question templates — varied syntax to stress-test embeddings
historical_templates = [
    "What was the initial {predicate} recorded for {entity}?",
    "Prior to any updates, what {predicate} was associated with {entity}?",
    "Looking at the earliest entry, which {predicate} did {entity} have?",
    "Regarding {entity}, what was the foundational {predicate} first documented?",
    "In the original records, which {predicate} was attributed to {entity}?",
    "What {predicate} was {entity} first catalogued under?",
    "Tracing back to the earliest data point, what {predicate} applied to {entity}?",
    "What was the primordial {predicate} designation for {entity}?",
    "Which {predicate} was the genesis entry for {entity}?",
    "Before any modifications, what {predicate} corresponded to {entity}?",
    "What archival {predicate} was initially assigned to {entity}?",
    "What {predicate} did the founding record of {entity} specify?",
    "According to the inaugural entry, what {predicate} did {entity} possess?",
    "What was the ancestral {predicate} linked to {entity}?",
    "In the baseline documentation, what {predicate} was tied to {entity}?",
    "What {predicate} marked the debut record of {entity}?",
    "What preliminary {predicate} was first recorded for {entity}?",
    "Which {predicate} constituted the earliest known fact about {entity}?",
    "What was the embryonic {predicate} information for {entity}?",
    "What {predicate} did the seminal record of {entity} contain?",
]

aggregation_templates = [
    "Give me the total count of distinct {predicate}s that have been associated with {entity}.",
    "Across all records, how many unique {predicate}s does {entity} have?",
    "What is the cumulative number of different {predicate}s recorded for {entity}?",
    "Tallying every distinct {predicate}, how many are linked to {entity}?",
    "Calculate the total sum of unique {predicate} entries for {entity}.",
    "How many separate {predicate} designations exist in the records for {entity}?",
    "Enumerate the number of non-duplicate {predicate}s that pertain to {entity}.",
    "What quantity of distinct {predicate}s has {entity} been catalogued under?",
    "Counting only unique values, how many {predicate}s are on file for {entity}?",
    "How many discrete {predicate} classifications apply to {entity}?",
    "Provide the cardinality of the set of {predicate}s associated with {entity}.",
    "What is the total tally of disparate {predicate}s for {entity}?",
    "How many non-redundant {predicate}s have been attributed to {entity}?",
    "Determine the aggregate count of differing {predicate}s for {entity}.",
    "What is the census of unique {predicate} entries under {entity}?",
    "How many individual {predicate}s appear across the dataset for {entity}?",
    "What is the sum total of varied {predicate}s recorded against {entity}?",
    "Quantify the distinct {predicate} assignments for {entity}.",
    "How many heterogeneous {predicate}s has {entity} accumulated?",
    "Provide a count of all unique {predicate}s that reference {entity}.",
]

boolean_true_templates = [
    "Is it accurate that {entity} was at some point recorded under the {predicate} of {value}?",
    "Can you verify whether {value} was ever documented as the {predicate} for {entity}?",
    "In the knowledge base, does {entity} have any association with {value} as a {predicate}?",
    "Confirm or deny: the {predicate} {value} appears in the historical records of {entity}.",
    "Was {value} ever registered as a valid {predicate} for {entity} at any point?",
    "Has {entity} at any recorded instance been linked to the {predicate} {value}?",
    "Does the evidence support that {value} served as the {predicate} for {entity}?",
    "Is there any factual basis for {entity} having the {predicate} {value}?",
    "At any juncture in the records, was {value} the documented {predicate} for {entity}?",
    "Can it be substantiated that {entity} carried the {predicate} designation of {value}?",
]

boolean_false_templates = [
    "Is it the case that {entity} was ever recorded under the {predicate} of {value}?",
    "Was the {predicate} {value} ever a documented fact about {entity}?",
    "Does the database contain any record of {entity} having the {predicate} of {value}?",
    "Can you confirm that {value} appeared as a {predicate} entry for {entity}?",
    "Is there documentation showing {entity} was associated with the {predicate} {value}?",
    "Was {value} at any point logged as the {predicate} for {entity}?",
    "Did {entity} ever possess the {predicate} designation of {value}?",
    "Has {value} been attributed as the {predicate} for {entity} in any record?",
    "According to the facts, was {entity} ever linked to the {predicate} {value}?",
    "Is it verifiable that {value} was the {predicate} for {entity} at any time?",
]

# Fabricated false values per predicate type
false_values = {
    "country": ["Japan", "Brazil", "Iceland", "Madagascar", "Finland", "Peru", "Mongolia", "Norway", "Ecuador", "Thailand",
                 "Fiji", "Bhutan", "Latvia", "Uruguay", "Zambia", "Cyprus", "Bolivia", "Bahrain", "Tonga", "Liechtenstein"],
    "genre": ["Noise Rock", "Vaporwave", "Zydeco", "Afrobeat", "Shoegaze", "Drone Metal", "Glitch Hop", "Cumbia",
              "Krautrock", "Psybient", "Chiptune", "Synthwave", "Bossa Nova", "Ragga", "Bhangra", "J-Core", "Turbofolk",
              "Funaná", "Tropicália", "Fado"],
    "location": ["Reykjavik", "Ulaanbaatar", "Timbuktu", "Vladivostok", "Nairobi", "Quito", "Antananarivo", "Suva",
                  "Thimphu", "Kathmandu", "Ouagadougou", "Belmopan", "Funafuti", "Dili", "Nukuʻalofa", "Tarawa",
                  "Majuro", "Palikir", "Apia", "Roseau"],
    "default": ["Plutonium", "Narwhal Inc.", "Quantum Dynamics", "Hyperion Corp", "Nebula Systems", "Axiom Ltd",
                "Zenith Group", "Prism Analytics", "Vortex Solutions", "Apex Industries", "Cipher Holdings",
                "Nexus Global", "Orion Ventures", "Stratos Capital", "Helix Partners", "Quasar Labs",
                "Pinnacle Tech", "Meridian Corp", "Atlas Collective", "Spectra Holdings"],
}

import random
random.seed(42)

for i, (entity, info) in enumerate(sorted_entities):
    oldest_fact = info["facts"][0]
    newest_fact = info["facts"][-1]
    all_values = list(info["distinct_values"])
    n_distinct = info["n_distinct"]
    
    # Infer predicate type from fact text
    fact_text_lower = oldest_fact["text"].lower()
    if "country" in fact_text_lower:
        predicate = "country"
    elif "genre" in fact_text_lower:
        predicate = "genre"
    elif "located" in fact_text_lower or "capital" in fact_text_lower:
        predicate = "location"
    elif "language" in fact_text_lower:
        predicate = "language"
    elif "religion" in fact_text_lower:
        predicate = "religion"
    elif "nationality" in fact_text_lower:
        predicate = "nationality"
    elif "ethnicity" in fact_text_lower:
        predicate = "ethnicity"
    elif "occupation" in fact_text_lower:
        predicate = "occupation"
    elif "created by" in fact_text_lower or "directed by" in fact_text_lower:
        predicate = "creator"
    else:
        predicate = "value"
    
    # ── Historical Question ──
    h_template = historical_templates[i % len(historical_templates)]
    h_question = h_template.format(entity=entity, predicate=predicate)
    benchmark.append({
        "entity": entity,
        "intent": "historical",
        "question_text": h_question,
        "ground_truth_answer": oldest_fact["value"]
    })
    
    # ── Aggregation Question ──
    a_template = aggregation_templates[i % len(aggregation_templates)]
    a_question = a_template.format(entity=entity, predicate=predicate)
    benchmark.append({
        "entity": entity,
        "intent": "aggregation",
        "question_text": a_question,
        "ground_truth_answer": str(n_distinct)
    })
    
    # ── Boolean Question ──
    # Alternate between True and False to create a balanced set
    if i % 2 == 0:
        # True: pick a value that actually exists
        bool_value = random.choice(all_values)
        b_template = boolean_true_templates[i % len(boolean_true_templates)]
        b_question = b_template.format(entity=entity, predicate=predicate, value=bool_value)
        benchmark.append({
            "entity": entity,
            "intent": "boolean",
            "question_text": b_question,
            "ground_truth_answer": "True"
        })
    else:
        # False: pick a fabricated value that does NOT exist
        false_pool = false_values.get(predicate, false_values["default"])
        fake_value = random.choice([v for v in false_pool if v not in all_values])
        b_template = boolean_false_templates[i % len(boolean_false_templates)]
        b_question = b_template.format(entity=entity, predicate=predicate, value=fake_value)
        benchmark.append({
            "entity": entity,
            "intent": "boolean",
            "question_text": b_question,
            "ground_truth_answer": "False"
        })

# ── Write to disk ──────────────────────────────────────────────────────────
import os
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "synthetic_benchmark.json")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(benchmark, f, indent=2, ensure_ascii=False)

print(f"\n✅ Wrote {len(benchmark)} questions ({len(sorted_entities)} entities × 3 intents)")
print(f"   → {output_path}")
print(f"\nBreakdown:")
print(f"   Historical: {sum(1 for q in benchmark if q['intent'] == 'historical')}")
print(f"   Aggregation: {sum(1 for q in benchmark if q['intent'] == 'aggregation')}")
print(f"   Boolean (True): {sum(1 for q in benchmark if q['intent'] == 'boolean' and q['ground_truth_answer'] == 'True')}")
print(f"   Boolean (False): {sum(1 for q in benchmark if q['intent'] == 'boolean' and q['ground_truth_answer'] == 'False')}")
