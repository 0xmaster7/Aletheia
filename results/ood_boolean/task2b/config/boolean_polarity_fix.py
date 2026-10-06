"""One frozen Boolean polarity correction for Task 2b."""
from __future__ import annotations
from dataclasses import replace
import re
from scripts.lib.prof_feedback import QueryPlan

_EXPLICIT_NEGATION = re.compile(r"\b(?:not|never|no)\b", re.I)

def apply_boolean_polarity_fix(question: str, plan: QueryPlan) -> QueryPlan:
    if plan.intent != "boolean":
        return plan
    if plan.negated and not _EXPLICIT_NEGATION.search(question):
        return replace(plan, negated=False)
    return plan
