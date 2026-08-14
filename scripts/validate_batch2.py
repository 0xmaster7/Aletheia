"""Validate the new 120 questions in data/synthetic_benchmark.json."""
import json
import re
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(REPO_ROOT, "data", "synthetic_benchmark.json")

with open(path, encoding="utf-8") as f:
    data = json.load(f)

new_qs = data[60:]  # only the 120 new ones

# Check 1: bad plurals
bad_plural = [q for q in new_qs if re.search(r'\b(countrys|nationalitys|ethnicitys|occupations?s|locations?s)\b', q["question_text"])]
print(f"Bad plurals found : {len(bad_plural)}")
for q in bad_plural:
    print(f"  [{q['entity']}] {q['question_text']}")

# Check 2: cross-predicate bleed — religion questions containing country values
KNOWN_COUNTRIES = {
    "United States of America", "Australia", "India", "United Kingdom",
    "Germany", "France", "Canada", "China", "Japan", "Russia",
    "Brazil", "Denmark", "New Zealand", "Fiji", "Iceland", "Mexico",
    "Norway", "Sweden", "Poland", "Argentina",
}
cross_pred = []
for q in new_qs:
    if q["intent"] == "boolean" and "as a religion" in q["question_text"]:
        for country in KNOWN_COUNTRIES:
            if country in q["question_text"]:
                cross_pred.append((q["entity"], q["question_text"]))
                break

print(f"\nCross-predicate religion/country bleed: {len(cross_pred)}")
for e, qt in cross_pred:
    print(f"  [{e}] {qt}")

# Print all boolean True/False for manual review
print("\n── Boolean TRUE questions (new batch) ──")
for q in new_qs:
    if q["intent"] == "boolean" and q["ground_truth_answer"] == "True":
        print(f"  {q['entity'][:30]:<30} | {q['question_text'][:80]}")

print("\n── Boolean FALSE questions (new batch) ──")
for q in new_qs:
    if q["intent"] == "boolean" and q["ground_truth_answer"] == "False":
        print(f"  {q['entity'][:30]:<30} | {q['question_text'][:80]}")

# Final verdict
is_clean = not bad_plural and not cross_pred
print(f"\nTotal new questions : {len(new_qs)}")
print(f"Total in file       : {len(data)}")
print(f"Status              : {'CLEAN ✅' if is_clean else 'ISSUES FOUND ❌'}")
