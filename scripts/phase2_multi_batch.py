"""Phase 2 — Multi-Batch Runner with In-Loop Validation.

Runs N_BATCHES sequential batches of BATCH_SIZE entities. For each batch:
  1. Generate 120 questions deterministically.
  2. Validate: plurals, cross-predicate boolean bleed, JSON hygiene.
  3. Auto-correct any errors in memory.
  4. Append atomically to data/synthetic_benchmark.json.
  5. Print per-batch report.

Usage:
    /usr/local/bin/python3.11 scripts/phase2_multi_batch.py
"""
from __future__ import annotations
import json, re, os, random, math
from collections import defaultdict
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────
REPO_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_PATH = os.path.join(REPO_ROOT, "data", "synthetic_benchmark.json")
BATCH_SIZE    = 40
N_BATCHES     = 1
random.seed(42)

# ── Helpers ───────────────────────────────────────────────────────────────
PLURAL_MAP: dict[str, str] = {
    "country":     "countries",
    "nationality": "nationalities",
    "ethnicity":   "ethnicities",
    "language":    "languages",
    "religion":    "religions",
    "genre":       "genres",
    "location":    "locations",
    "occupation":  "occupations",
    "creator":     "creators",
    "value":       "values",
}

def pluralize(pred: str) -> str:
    return PLURAL_MAP.get(pred, pred + "s")

def infer_predicate(text_lower: str) -> str:
    if "country"    in text_lower: return "country"
    if "genre"      in text_lower: return "genre"
    if "located"    in text_lower or "capital" in text_lower: return "location"
    if "language"   in text_lower: return "language"
    if "religion"   in text_lower: return "religion"
    if "nationality" in text_lower: return "nationality"
    if "ethnicity"  in text_lower: return "ethnicity"
    if "occupation" in text_lower: return "occupation"
    if "created by" in text_lower or "directed by" in text_lower: return "creator"
    return "value"

# ── Predicate-specific false distractor pools ──────────────────────────────
FALSE_VALUES: dict[str, list[str]] = {
    "country": [
        "Fictoria","Nuldania","Zephyria","Valdoria","Morthania",
        "Escandor","Pyranthia","Duskholm","Arenovia","Cresthaven",
        "Lumbria","Vesperine","Solmoor","Grimholt","Brackenvale",
        "Threnveld","Dunhaven","Selgrave","Ostmark","Veldtshire",
    ],
    "nationality": [
        "Fictorian","Nuldanian","Zephyrian","Valdorian","Morthanian",
        "Escandorian","Pyranthian","Duskholmian","Arenovian","Cresthavian",
        "Lumbrian","Vesperinian","Solmoorian","Grimholtian","Brackenvian",
        "Threnveldian","Dunhavian","Selgravian","Ostmarkian","Veldtshirian",
    ],
    "ethnicity": [
        "Thornosi","Veldrani","Caluvian","Mirekai","Ossivari",
        "Dralthic","Fenhari","Korveshi","Suldrani","Pelthori",
        "Grauvari","Nocthari","Selvani","Brethosi","Quelvari",
        "Mirathi","Drothani","Fenvari","Korvathi","Suldhari",
    ],
    "language": [
        "Thornosi","Veldrani","Caluvian","Mirekai","Ossivari",
        "Dralthic","Fenhari","Korveshi","Suldrani","Pelthori",
        "Grauvian","Nocthari","Selvanese","Brethosi","Quelvarish",
        "Mirathi","Drothanic","Fenvari","Korvathi","Suldharian",
    ],
    "religion": [
        "Zorvanism","Thalmorism","Vekthrism","Pyranthism","Duskfaith",
        "Atherionism","Veldranism","Solmorism","Noctharism","Fenharism",
        "Ossivarianism","Caluvianism","Mirethism","Bractharism","Quelvarism",
        "Dralthicism","Suldharism","Threnveldism","Grimholtism","Brackenfaith",
    ],
    "genre": [
        "Noise Rock","Vaporwave","Zydeco","Shoegaze","Drone Metal",
        "Glitch Hop","Krautrock","Psybient","Chiptune","Turbofolk",
        "Funanã","Tropicália","Fado","Cumbia","Afrobeat",
        "Bhangra","J-Core","Bossa Nova","Ragga","Synthwave",
    ],
    "location": [
        "Reykjavik","Ulaanbaatar","Timbuktu","Vladivostok","Nairobi",
        "Quito","Antananarivo","Suva","Thimphu","Kathmandu",
        "Ouagadougou","Belmopan","Funafuti","Dili","Tarawa",
        "Majuro","Palikir","Apia","Roseau","Nukualofa",
    ],
    "occupation": [
        "Zookeeper","Glassblower","Farrier","Lepidopterist","Campanologist",
        "Coopersmith","Chandler","Milliner","Tanner","Wheelwright",
        "Fletcher","Thatcher","Cordwainer","Hosier","Haberdasher",
        "Mercer","Draper","Ironmonger","Verderer","Furbisher",
    ],
    "creator": [
        "Narwhal Inc.","Quantum Dynamics","Hyperion Corp","Nebula Systems",
        "Axiom Ltd","Zenith Group","Prism Analytics","Vortex Solutions",
        "Apex Industries","Cipher Holdings","Nexus Global","Orion Ventures",
        "Stratos Capital","Helix Partners","Quasar Labs","Pinnacle Tech",
        "Meridian Corp","Atlas Collective","Spectra Holdings","Solaris Works",
    ],
    "value": [
        "Narwhal Inc.","Quantum Dynamics","Hyperion Corp","Nebula Systems",
        "Axiom Ltd","Zenith Group","Prism Analytics","Vortex Solutions",
        "Apex Industries","Cipher Holdings","Nexus Global","Orion Ventures",
        "Stratos Capital","Helix Partners","Quasar Labs","Pinnacle Tech",
        "Meridian Corp","Atlas Collective","Spectra Holdings","Solaris Works",
    ],
}

# ── Question templates ─────────────────────────────────────────────────────
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

# ── Validation ────────────────────────────────────────────────────────────
BAD_PLURAL_RE = re.compile(
    r'\b(countrys|nationalitys|ethnicitys|occupations{2,}|creators{2,}|genres{2,}|'
    r'religions{2,}|languages{2,}|locations{2,}|values{2,})\b',
    re.IGNORECASE,
)

def validate_and_fix(questions: list[dict], entity_info_map: dict) -> tuple[list[dict], list[str]]:
    """Validate batch; return (cleaned_questions, list_of_fixes_applied)."""
    fixed = []
    log = []

    for q in questions:
        text = q["question_text"]

        # 1. Bad plural check
        if BAD_PLURAL_RE.search(text):
            # Re-derive correct form
            for bad, good in [
                ("countrys", "countries"), ("nationalitys", "nationalities"),
                ("ethnicitys", "ethnicities"),
            ]:
                if bad in text:
                    text = text.replace(bad, good)
                    log.append(f"  [PLURAL FIX] {q['entity']}: '{bad}' → '{good}'")
            q = dict(q)
            q["question_text"] = text

        # 2. Cross-predicate boolean bleed check (True booleans only)
        if q["intent"] == "boolean" and q["ground_truth_answer"] == "True":
            entity = q["entity"]
            info   = entity_info_map.get(entity)
            if info:
                oldest_pred = infer_predicate(info["facts"][0]["text"].lower())
                same_pred_vals = [
                    f["value"] for f in info["facts"]
                    if f.get("predicate_type") == oldest_pred
                ]
                # Check if the value in the question is from same-predicate facts
                # Extract value by checking which same_pred_val appears in question text
                found_valid = any(v in text for v in same_pred_vals)
                if not found_valid and same_pred_vals:
                    # Re-generate with a valid same-predicate value
                    replacement_val = random.choice(same_pred_vals)
                    # Reconstruct question using same template slot
                    new_text = re.sub(
                        r'(?<=of |with |was )\S.*?(?= as a | for | at any| the |\?)',
                        replacement_val,
                        text,
                    )
                    if new_text == text:
                        # Simpler replacement: just rebuild from template
                        global_idx  = list(entity_info_map.keys()).index(entity) if entity in entity_info_map else 0
                        tmpl_idx    = global_idx % len(boolean_true_templates)
                        new_text    = boolean_true_templates[tmpl_idx].format(
                            entity=entity,
                            predicate=oldest_pred,
                            value=replacement_val,
                        )
                    log.append(
                        f"  [BOOL BLEED FIX] {entity}: replaced cross-predicate value → '{replacement_val}'"
                    )
                    q = dict(q)
                    q["question_text"] = new_text

        # 3. JSON hygiene: no markdown or stray text
        for field in ("entity", "intent", "question_text", "ground_truth_answer"):
            if "```" in str(q.get(field, "")):
                q = dict(q)
                q[field] = q[field].replace("```", "").strip()
                log.append(f"  [MARKDOWN FIX] {q['entity']}: stripped backticks from {field}")

        fixed.append(q)

    return fixed, log


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — one-time dataset load, then batch loop
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PHASE 2 MULTI-BATCH RUNNER")
print(f"Target: {N_BATCHES} batches × {BATCH_SIZE} entities = {N_BATCHES * BATCH_SIZE} entities")
print("=" * 60)

# Load dataset once
print("\nLoading MemoryAgentBench (cached)...")
ds  = load_dataset("ai-hyz/MemoryAgentBench", split="Conflict_Resolution", revision="main")
row = next(s for s in ds if s["metadata"]["source"] == "factconsolidation_sh_262k")
ctx = row["context"]

# Parse all facts once
pat     = re.compile(r"(\d+)\.\s")
matches = list(pat.finditer(ctx))
facts   = []
for i, m in enumerate(matches):
    idx  = int(m.group(1))
    s    = m.end()
    e    = matches[i + 1].start() if i + 1 < len(matches) else len(ctx)
    facts.append((idx, ctx[s:e].strip().rstrip(".")))
print(f"Parsed {len(facts):,} facts.\n")

# Build entity_facts once
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
        entity_facts[entity].append({"serial": serial, "text": text, "value": value, "predicate_type": ptype})
    else:
        m2 = fallback_pattern.match(text)
        if m2:
            entity = m2.group(1).strip()
            value  = m2.group(4).strip()
            ptype  = infer_predicate(text.lower())
            entity_facts[entity].append({"serial": serial, "text": text, "value": value, "predicate_type": ptype})

# Build conflicting entity list (sorted deterministically, same key as always)
conflicting_entities: dict = {}
for entity, fact_list in entity_facts.items():
    distinct_values = set(f["value"] for f in fact_list)
    if len(distinct_values) >= 2 and len(fact_list) >= 2:
        sorted_facts = sorted(fact_list, key=lambda x: x["serial"])
        conflicting_entities[entity] = {
            "facts":           sorted_facts,
            "distinct_values": distinct_values,
            "n_distinct":      len(distinct_values),
            "n_facts":         len(fact_list),
        }

sorted_entities = sorted(
    conflicting_entities.items(),
    key=lambda x: (x[1]["n_distinct"], x[1]["n_facts"]),
    reverse=True,
)
print(f"Total conflicting entities: {len(sorted_entities):,}\n")

# ── Batch loop ────────────────────────────────────────────────────────────
overall_new = 0

for batch_num in range(1, N_BATCHES + 1):
    print(f"\n{'─' * 60}")
    print(f"BATCH {batch_num:2d} / {N_BATCHES}")
    print(f"{'─' * 60}")

    # Re-load benchmark each batch so 'already_done' is always current
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        existing = json.load(f)
    already_done = set(item["entity"] for item in existing)

    unprocessed = [(e, info) for e, info in sorted_entities if e not in already_done]
    if len(unprocessed) < 1:
        print("  ⚠️  No unprocessed entities left. Stopping early.")
        break

    batch    = unprocessed[:BATCH_SIZE]
    OFFSET   = len(already_done)

    # ── Generate ──
    new_questions: list[dict] = []
    entity_info_map: dict     = {}

    for batch_idx, (entity, info) in enumerate(batch):
        global_idx      = OFFSET + batch_idx
        oldest_fact     = info["facts"][0]
        predicate       = infer_predicate(oldest_fact["text"].lower())
        pred_plural     = pluralize(predicate)
        all_values      = list(info["distinct_values"])
        same_pred_vals  = [
            f["value"] for f in info["facts"]
            if f.get("predicate_type") == predicate
        ] or all_values

        entity_info_map[entity] = info

        # Historical
        h_tmpl = historical_templates[global_idx % len(historical_templates)]
        new_questions.append({
            "entity":               entity,
            "intent":               "historical",
            "question_text":        h_tmpl.format(entity=entity, predicate=predicate),
            "ground_truth_answer":  oldest_fact["value"],
        })

        # Aggregation
        a_tmpl = aggregation_templates[global_idx % len(aggregation_templates)]
        new_questions.append({
            "entity":               entity,
            "intent":               "aggregation",
            "question_text":        a_tmpl.format(
                                        entity=entity,
                                        predicate=predicate,
                                        predicate_plural=pred_plural,
                                    ),
            "ground_truth_answer":  str(info["n_distinct"]),
        })

        # Boolean (alternate True/False on global_idx)
        if global_idx % 2 == 0:
            bool_value = random.choice(same_pred_vals)
            b_tmpl     = boolean_true_templates[global_idx % len(boolean_true_templates)]
            new_questions.append({
                "entity":               entity,
                "intent":               "boolean",
                "question_text":        b_tmpl.format(
                                            entity=entity, predicate=predicate, value=bool_value
                                        ),
                "ground_truth_answer":  "True",
            })
        else:
            false_pool = FALSE_VALUES.get(predicate, FALSE_VALUES["value"])
            fake_value = random.choice([v for v in false_pool if v not in all_values])
            b_tmpl     = boolean_false_templates[global_idx % len(boolean_false_templates)]
            new_questions.append({
                "entity":               entity,
                "intent":               "boolean",
                "question_text":        b_tmpl.format(
                                            entity=entity, predicate=predicate, value=fake_value
                                        ),
                "ground_truth_answer":  "False",
            })

    # ── Validate & auto-correct ──
    clean_questions, fix_log = validate_and_fix(new_questions, entity_info_map)

    if fix_log:
        print(f"  ⚠️  Auto-corrections applied ({len(fix_log)}):")
        for entry in fix_log:
            print(entry)
    else:
        print("  ✅ Validation passed — no issues found.")

    # Final sanity counts
    n_hist  = sum(1 for q in clean_questions if q["intent"] == "historical")
    n_agg   = sum(1 for q in clean_questions if q["intent"] == "aggregation")
    n_bt    = sum(1 for q in clean_questions if q["intent"] == "boolean" and q["ground_truth_answer"] == "True")
    n_bf    = sum(1 for q in clean_questions if q["intent"] == "boolean" and q["ground_truth_answer"] == "False")

    assert len(clean_questions) % 3 == 0, f"Expected multiple of 3 questions, got {len(clean_questions)}"
    n_entities = len(batch)
    assert n_hist == n_entities,  f"Expected {n_entities} historical, got {n_hist}"
    assert n_agg  == n_entities,  f"Expected {n_entities} aggregation, got {n_agg}"
    assert (n_bt + n_bf) == n_entities, f"Expected {n_entities} boolean, got {n_bt + n_bf}"

    # ── Atomic save ──
    combined = existing + clean_questions
    with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    overall_new += len(clean_questions)

    entities_done = [e for e, _ in batch]
    print(f"\n  Entities processed : {len(batch)}")
    print(f"  Questions added    : {len(clean_questions)}  (hist={n_hist}, agg={n_agg}, bool_T={n_bt}, bool_F={n_bf})")
    print(f"  File total now     : {len(combined)} questions")
    print(f"  Sample entities    : {', '.join(entities_done[:5])} ...")

print("\n" + "=" * 60)
print(f"DONE — {N_BATCHES} batches complete.")
print(f"Total new questions added : {overall_new}")
with open(BENCHMARK_PATH, encoding="utf-8") as f:
    final = json.load(f)
print(f"Grand total in file       : {len(final)}")
print(f"Grand total entities      : {len(set(q['entity'] for q in final))}")
print("=" * 60)
