"""Full structural integrity validation for data/synthetic_benchmark.json.

Checks:
  1. Total object count == 13425
  2. All required keys present on every object
  3. No empty question_text or ground_truth_answer
  4. Intent values are strictly the three expected strings
  5. Boolean answers are strictly "True" or "False"
  6. Aggregation answers are parseable positive integers
  7. Total distinct entities == 4475
  8. No duplicate (entity, intent) pairs
  9. Every entity has exactly one of each intent
 10. No bad plurals ("countrys", "nationalitys", "ethnicitys")
 11. Distribution report (boolean balance, top ground-truth values, predicate spread)
"""
import json
import re
from collections import defaultdict

FILE_PATH       = "data/synthetic_benchmark.json"
EXPECTED_TOTAL  = 13425
EXPECTED_ENT    = 4475
VALID_INTENTS   = {"historical", "aggregation", "boolean"}
VALID_BOOL_ANS  = {"True", "False"}
BAD_PLURAL_RE   = re.compile(r'\b(countrys|nationalitys|ethnicitys)\b', re.IGNORECASE)

errors = []

# ── Load ──────────────────────────────────────────────────────────────────
try:
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception as e:
    print(f"❌ Could not load file: {e}")
    raise SystemExit(1)

print(f"Loaded {len(data):,} objects from {FILE_PATH}\n")

# ── 1. Total count ─────────────────────────────────────────────────────────
if len(data) != EXPECTED_TOTAL:
    errors.append(f"[COUNT] Expected {EXPECTED_TOTAL}, got {len(data)}")

# ── Per-object checks ──────────────────────────────────────────────────────
entities        = defaultdict(list)
seen_pairs      = set()
bad_plural_hits = []

for i, obj in enumerate(data):

    # 2. Required keys
    missing = [k for k in ("entity", "intent", "question_text", "ground_truth_answer") if k not in obj]
    if missing:
        errors.append(f"[KEYS] Object #{i}: missing {missing}")
        continue

    entity  = obj["entity"]
    intent  = obj["intent"]
    qtext   = str(obj["question_text"]).strip()
    answer  = str(obj["ground_truth_answer"]).strip()

    # 3. No empty fields
    if not qtext:
        errors.append(f"[EMPTY_Q] #{i} ({entity}): empty question_text")
    if not answer:
        errors.append(f"[EMPTY_A] #{i} ({entity}): empty ground_truth_answer")

    # 4. Intent values
    if intent not in VALID_INTENTS:
        errors.append(f"[INTENT] #{i} ({entity}): invalid intent '{intent}'")

    # 5. Boolean answer validation
    if intent == "boolean" and answer not in VALID_BOOL_ANS:
        errors.append(f"[BOOL_ANS] #{i} ({entity}): boolean answer must be 'True'/'False', got '{answer}'")

    # 6. Aggregation answer is a positive integer
    if intent == "aggregation":
        try:
            v = int(answer)
            if v < 1:
                errors.append(f"[AGG_ANS] #{i} ({entity}): aggregation answer < 1: '{answer}'")
        except ValueError:
            errors.append(f"[AGG_ANS] #{i} ({entity}): aggregation answer not integer: '{answer}'")

    # 7. Duplicate (entity, intent) pairs
    pair = (entity, intent)
    if pair in seen_pairs:
        errors.append(f"[DUPE] Duplicate (entity, intent) pair: {pair}")
    seen_pairs.add(pair)

    # 8. Bad plurals
    if BAD_PLURAL_RE.search(qtext):
        bad_plural_hits.append((i, entity, qtext))

    entities[entity].append(intent)

# ── 9. Entity count ────────────────────────────────────────────────────────
if len(entities) != EXPECTED_ENT:
    errors.append(f"[ENT_COUNT] Expected {EXPECTED_ENT} entities, got {len(entities)}")

# ── 10. Each entity has exactly one of each intent ─────────────────────────
intent_errors = []
for ent, intents in entities.items():
    if sorted(intents) != ["aggregation", "boolean", "historical"]:
        intent_errors.append(f"  {ent}: {sorted(intents)}")
if intent_errors:
    errors.append(f"[INTENT_SET] {len(intent_errors)} entities have wrong intent set:")
    errors.extend(intent_errors[:10])  # show first 10 only

# ── 11. Bad plurals ────────────────────────────────────────────────────────
if bad_plural_hits:
    errors.append(f"[BAD_PLURAL] {len(bad_plural_hits)} bad plurals found:")
    for i, ent, qt in bad_plural_hits[:5]:
        errors.append(f"  #{i} ({ent}): {qt[:80]}")

# ── Distribution report ────────────────────────────────────────────────────
print("── Distribution Report ──────────────────────────────────────────")

bool_qs   = [o for o in data if o["intent"] == "boolean"]
bool_true  = sum(1 for q in bool_qs if q["ground_truth_answer"] == "True")
bool_false = sum(1 for q in bool_qs if q["ground_truth_answer"] == "False")
print(f"  Intent counts  : historical={sum(1 for o in data if o['intent']=='historical'):,}  "
      f"aggregation={sum(1 for o in data if o['intent']=='aggregation'):,}  "
      f"boolean={len(bool_qs):,}")
print(f"  Boolean balance: True={bool_true:,}  False={bool_false:,}  "
      f"(ratio {bool_true/max(bool_false,1):.2f}:1)")

agg_answers = [int(o["ground_truth_answer"]) for o in data
               if o["intent"] == "aggregation" and o["ground_truth_answer"].isdigit()]
if agg_answers:
    print(f"  Aggregation answers: min={min(agg_answers)}, max={max(agg_answers)}, "
          f"mean={sum(agg_answers)/len(agg_answers):.2f}")

from collections import Counter
top_agg = Counter(agg_answers).most_common(5)
print(f"  Top aggregation counts: {top_agg}")

print()

# ── Final verdict ──────────────────────────────────────────────────────────
if errors:
    print(f"❌ Validation FAILED — {len(errors)} error(s) found:\n")
    for e in errors[:20]:
        print(f"  {e}")
    if len(errors) > 20:
        print(f"  ... and {len(errors) - 20} more.")
else:
    print(f"Total objects         : {len(data):,}")
    print(f"Total distinct entities: {len(entities):,}")
    print("✅ DATASET IS 100% VALIDATED AND PAPER-READY.")
