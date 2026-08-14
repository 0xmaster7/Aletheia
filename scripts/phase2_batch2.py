"""Phase 2 — Batch 2 (Fixed): 40 new entities with correct pluralization and
predicate-aware boolean distractors.

Fixes applied vs. previous run:
  1. Pluralization: uses PLURAL_MAP so "country" → "countries", not "countrys".
  2. Boolean True: only picks a value whose fact shares the same predicate type as
     the oldest fact, preventing cross-predicate bleed (e.g. Australia ≠ religion).
  3. Boolean False: distractor pool is strictly keyed to the inferred predicate type.

Usage:
    /usr/local/bin/python3.11 scripts/phase2_batch2.py
"""
from __future__ import annotations
import json
import re
import os
import random
from collections import defaultdict
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_PATH = os.path.join(REPO_ROOT, "data", "synthetic_benchmark.json")
BATCH_SIZE = 40
random.seed(42)

# ── 1. Pluralization map ──────────────────────────────────────────────────
PLURAL_MAP: dict[str, str] = {
    "country":      "countries",
    "nationality":  "nationalities",
    "ethnicity":    "ethnicities",
    "language":     "languages",
    "religion":     "religions",
    "genre":        "genres",
    "location":     "locations",
    "occupation":   "occupations",
    "creator":      "creators",
    "value":        "values",
}

def pluralize(pred: str) -> str:
    return PLURAL_MAP.get(pred, pred + "s")

# ── 2. Predicate inference (shared helper) ────────────────────────────────
def infer_predicate(text_lower: str) -> str:
    if "country" in text_lower:
        return "country"
    if "genre" in text_lower:
        return "genre"
    if "located" in text_lower or "capital" in text_lower:
        return "location"
    if "language" in text_lower:
        return "language"
    if "religion" in text_lower:
        return "religion"
    if "nationality" in text_lower:
        return "nationality"
    if "ethnicity" in text_lower:
        return "ethnicity"
    if "occupation" in text_lower:
        return "occupation"
    if "created by" in text_lower or "directed by" in text_lower:
        return "creator"
    return "value"

# ── 3. Predicate-specific false distractor pools ──────────────────────────
# Each key maps to fabricated values that can NEVER be a real value for that predicate.
FALSE_VALUES: dict[str, list[str]] = {
    "country": [
        "Fictoria", "Nuldania", "Zephyria", "Valdoria", "Morthania",
        "Escandor", "Pyranthia", "Duskholm", "Arenovia", "Cresthaven",
        "Lumbria", "Vesperine", "Solmoor", "Grimholt", "Brackenvale",
        "Threnveld", "Dunhaven", "Selgrave", "Ostmark", "Veldtshire",
    ],
    "nationality": [
        "Fictorian", "Nuldanian", "Zephyrian", "Valdorian", "Morthanian",
        "Escandorian", "Pyranthian", "Duskholmian", "Arenovian", "Cresthavian",
        "Lumbrian", "Vesperinian", "Solmoorian", "Grimholtian", "Brackenvian",
        "Threnveldian", "Dunhavian", "Selgravian", "Ostmarkian", "Veldtshirian",
    ],
    "ethnicity": [
        "Thornosi", "Veldrani", "Caluvian", "Mirekai", "Ossivari",
        "Dralthic", "Fenhari", "Korveshi", "Suldrani", "Pelthori",
        "Grauvari", "Nocthari", "Selvani", "Brethosi", "Quelvari",
        "Mirathi", "Drothani", "Fenvari", "Korvathi", "Suldhari",
    ],
    "language": [
        "Thornosi", "Veldrani", "Caluvian", "Mirekai", "Ossivari",
        "Dralthic", "Fenhari", "Korveshi", "Suldrani", "Pelthori",
        "Grauvian", "Nocthari", "Selvanese", "Brethosi", "Quelvarish",
        "Mirathi", "Drothanic", "Fenvari", "Korvathi", "Suldharian",
    ],
    "religion": [
        "Zorvanism", "Thalmorism", "Vekthrism", "Pyranthism", "Duskfaith",
        "Atherionism", "Veldranism", "Solmorism", "Noctharism", "Fenharism",
        "Ossivarianism", "Caluvianism", "Mirethism", "Bractharism", "Quelvarism",
        "Dralthicism", "Suldharism", "Threnveldism", "Grimholtism", "Brackenfaith",
    ],
    "genre": [
        "Noise Rock", "Vaporwave", "Zydeco", "Shoegaze", "Drone Metal",
        "Glitch Hop", "Krautrock", "Psybient", "Chiptune", "Turbofolk",
        "Funanã", "Tropicália", "Fado", "Cumbia", "Afrobeat",
        "Bhangra", "J-Core", "Bossa Nova", "Ragga", "Synthwave",
    ],
    "location": [
        "Reykjavik", "Ulaanbaatar", "Timbuktu", "Vladivostok", "Nairobi",
        "Quito", "Antananarivo", "Suva", "Thimphu", "Kathmandu",
        "Ouagadougou", "Belmopan", "Funafuti", "Dili", "Tarawa",
        "Majuro", "Palikir", "Apia", "Roseau", "Nukualofa",
    ],
    "occupation": [
        "Zookeeper", "Glassblower", "Farrier", "Lepidopterist", "Campanologist",
        "Coopersmith", "Chandler", "Milliner", "Tanner", "Wheelwright",
        "Fletcher", "Thatcher", "Cordwainer", "Hosier", "Haberdasher",
        "Mercer", "Draper", "Ironmonger", "Verderer", "Furbisher",
    ],
    "creator": [
        "Narwhal Inc.", "Quantum Dynamics", "Hyperion Corp", "Nebula Systems",
        "Axiom Ltd", "Zenith Group", "Prism Analytics", "Vortex Solutions",
        "Apex Industries", "Cipher Holdings", "Nexus Global", "Orion Ventures",
        "Stratos Capital", "Helix Partners", "Quasar Labs", "Pinnacle Tech",
        "Meridian Corp", "Atlas Collective", "Spectra Holdings", "Solaris Works",
    ],
    "value": [
        "Narwhal Inc.", "Quantum Dynamics", "Hyperion Corp", "Nebula Systems",
        "Axiom Ltd", "Zenith Group", "Prism Analytics", "Vortex Solutions",
        "Apex Industries", "Cipher Holdings", "Nexus Global", "Orion Ventures",
        "Stratos Capital", "Helix Partners", "Quasar Labs", "Pinnacle Tech",
        "Meridian Corp", "Atlas Collective", "Spectra Holdings", "Solaris Works",
    ],
}

# ── Load existing benchmark ───────────────────────────────────────────────
with open(BENCHMARK_PATH, encoding="utf-8") as f:
    existing = json.load(f)
already_done = set(item["entity"] for item in existing)
print(f"Already processed {len(already_done)} entities.\n")

# ── Load dataset ──────────────────────────────────────────────────────────
print("Loading MemoryAgentBench dataset...")
ds = load_dataset("ai-hyz/MemoryAgentBench", split="Conflict_Resolution", revision="main")
row = next(s for s in ds if s["metadata"]["source"] == "factconsolidation_sh_262k")
ctx = row["context"]
print(f"Loaded context of length {len(ctx):,} chars.")

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
print(f"Parsed {len(facts)} facts.\n")

# ── Extract entity → facts (with per-fact predicate_type stored) ──────────
entity_facts: dict = defaultdict(list)

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
        entity = m.group(1).strip()
        value  = m.group(2).strip()
        ptype  = infer_predicate(text.lower())
        entity_facts[entity].append(
            {"serial": serial, "text": text, "value": value, "predicate_type": ptype}
        )
    else:
        m2 = fallback_pattern.match(text)
        if m2:
            entity = m2.group(1).strip()
            value  = m2.group(4).strip()
            ptype  = infer_predicate(text.lower())
            entity_facts[entity].append(
                {"serial": serial, "text": text, "value": value, "predicate_type": ptype}
            )

# ── Find conflicting entities ─────────────────────────────────────────────
conflicting_entities: dict = {}
for entity, fact_list in entity_facts.items():
    distinct_values = set(f["value"] for f in fact_list)
    if len(distinct_values) >= 2 and len(fact_list) >= 2:
        sorted_facts = sorted(fact_list, key=lambda x: x["serial"])
        conflicting_entities[entity] = {
            "facts": sorted_facts,
            "distinct_values": distinct_values,
            "n_distinct": len(distinct_values),
            "n_facts": len(fact_list),
        }

print(f"Total entities with conflicting facts: {len(conflicting_entities)}")

# Sort deterministically (same key as original script)
sorted_entities = sorted(
    conflicting_entities.items(),
    key=lambda x: (x[1]["n_distinct"], x[1]["n_facts"]),
    reverse=True,
)

# ── Skip processed, pick next batch ──────────────────────────────────────
unprocessed = [(e, info) for e, info in sorted_entities if e not in already_done]
batch = unprocessed[:BATCH_SIZE]
print(f"Selected {len(batch)} new entities.\n")

# ── Question templates ────────────────────────────────────────────────────
# Aggregation templates use {predicate_plural}; boolean/historical use {predicate} singular.
OFFSET = len(already_done)

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
    "Give me the total count of distinct {predicate_plural} that have been associated with {entity}.",
    "Across all records, how many unique {predicate_plural} does {entity} have?",
    "What is the cumulative number of different {predicate_plural} recorded for {entity}?",
    "Tallying every distinct {predicate}, how many are linked to {entity}?",
    "Calculate the total sum of unique {predicate} entries for {entity}.",
    "How many separate {predicate} designations exist in the records for {entity}?",
    "Enumerate the number of non-duplicate {predicate_plural} that pertain to {entity}.",
    "What quantity of distinct {predicate_plural} has {entity} been catalogued under?",
    "Counting only unique values, how many {predicate_plural} are on file for {entity}?",
    "How many discrete {predicate} classifications apply to {entity}?",
    "Provide the cardinality of the set of {predicate_plural} associated with {entity}.",
    "What is the total tally of disparate {predicate_plural} for {entity}?",
    "How many non-redundant {predicate_plural} have been attributed to {entity}?",
    "Determine the aggregate count of differing {predicate_plural} for {entity}.",
    "What is the census of unique {predicate} entries under {entity}?",
    "How many individual {predicate_plural} appear across the dataset for {entity}?",
    "What is the sum total of varied {predicate_plural} recorded against {entity}?",
    "Quantify the distinct {predicate} assignments for {entity}.",
    "How many heterogeneous {predicate_plural} has {entity} accumulated?",
    "Provide a count of all unique {predicate_plural} that reference {entity}.",
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

# ── Generate questions ─────────────────────────────────────────────────────
new_questions = []

for batch_idx, (entity, info) in enumerate(batch):
    global_idx  = OFFSET + batch_idx
    oldest_fact = info["facts"][0]
    predicate   = infer_predicate(oldest_fact["text"].lower())
    pred_plural = pluralize(predicate)
    all_values  = list(info["distinct_values"])

    # Values that genuinely match this predicate type (for True boolean).
    same_pred_values = [
        f["value"] for f in info["facts"]
        if f.get("predicate_type", "value") == predicate
    ]
    if not same_pred_values:
        same_pred_values = all_values  # graceful fallback

    n_distinct = info["n_distinct"]

    # ── Historical ──
    h_tmpl = historical_templates[global_idx % len(historical_templates)]
    h_q    = h_tmpl.format(entity=entity, predicate=predicate)
    new_questions.append({
        "entity": entity,
        "intent": "historical",
        "question_text": h_q,
        "ground_truth_answer": oldest_fact["value"],
    })

    # ── Aggregation ──
    a_tmpl = aggregation_templates[global_idx % len(aggregation_templates)]
    a_q    = a_tmpl.format(
        entity=entity,
        predicate=predicate,
        predicate_plural=pred_plural,
    )
    new_questions.append({
        "entity": entity,
        "intent": "aggregation",
        "question_text": a_q,
        "ground_truth_answer": str(n_distinct),
    })

    # ── Boolean ── (alternate True / False on global_idx)
    if global_idx % 2 == 0:
        # True: pick a real value from same-predicate facts
        bool_value = random.choice(same_pred_values)
        b_tmpl = boolean_true_templates[global_idx % len(boolean_true_templates)]
        b_q    = b_tmpl.format(entity=entity, predicate=predicate, value=bool_value)
        new_questions.append({
            "entity": entity,
            "intent": "boolean",
            "question_text": b_q,
            "ground_truth_answer": "True",
        })
    else:
        # False: pick a fabricated value from the correct predicate category
        false_pool = FALSE_VALUES.get(predicate, FALSE_VALUES["value"])
        fake_value = random.choice([v for v in false_pool if v not in all_values])
        b_tmpl = boolean_false_templates[global_idx % len(boolean_false_templates)]
        b_q    = b_tmpl.format(entity=entity, predicate=predicate, value=fake_value)
        new_questions.append({
            "entity": entity,
            "intent": "boolean",
            "question_text": b_q,
            "ground_truth_answer": "False",
        })

print(f"Generated {len(new_questions)} new questions ({len(batch)} entities × 3 intents).")

# ── Atomic write ──────────────────────────────────────────────────────────
combined = existing + new_questions
with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)

print(f"\n=== SUCCESS ===")
print(f"Written {len(combined)} total questions to {BENCHMARK_PATH}")
print(f"  Previous : {len(existing)}")
print(f"  New      : {len(new_questions)}")
print(f"\nNew question breakdown:")
print(f"  Historical      : {sum(1 for q in new_questions if q['intent'] == 'historical')}")
print(f"  Aggregation     : {sum(1 for q in new_questions if q['intent'] == 'aggregation')}")
print(f"  Boolean (True)  : {sum(1 for q in new_questions if q['intent'] == 'boolean' and q['ground_truth_answer'] == 'True')}")
print(f"  Boolean (False) : {sum(1 for q in new_questions if q['intent'] == 'boolean' and q['ground_truth_answer'] == 'False')}")

print("\n── Spot-check: first 6 entities ──")
for q in new_questions[:18]:
    print(f"  [{q['intent']:12s}] {q['entity']}")
    print(f"    Q: {q['question_text']}")
    print(f"    A: {q['ground_truth_answer']}")
    print()
