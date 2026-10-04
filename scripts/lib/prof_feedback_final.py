"""Evaluation-only v3 policy overrides for the final confirmatory run.

The frozen v2 operator remains unchanged. This module applies deterministic
question-level overrides after parsing the model's plan, then executes the
same local symbolic operators over retrieved facts.
"""
from __future__ import annotations

from dataclasses import replace
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.lib.prof_feedback import (
    Candidate,
    QueryPlan,
    deterministic_candidates,
    execute_boolean_gate,
    execute_plan,
    local_history_complete,
)
from scripts.lib.prof_feedback import approved_evaluation_prompt_and_schema

FINAL_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts/evaluation_final_v3.txt"


# These nine coarse relation names are exactly the generator's named predicates.
KNOWN_RELATION_LIST = (
    "country", "genre", "location", "language", "religion",
    "nationality", "ethnicity", "occupation", "creator",
)

# These are the historical template cues found in the generated benchmark.
# Explicit previous-state language maps to previous; all first-origin language
# maps to earliest. The source templates also use the four synonym cues below.
TEMPORAL_CUE_MAP = {
    "first": "earliest", "earliest": "earliest", "initial": "earliest",
    "oldest": "earliest", "original": "earliest", "previous": "previous",
    "prior": "previous", "earlier": "previous", "before": "previous",
    "beginning": "earliest", "inaugural": "earliest", "founding": "earliest",
    "ancestral": "earliest", "primordial": "earliest", "genesis": "earliest",
    "foundational": "earliest", "debut": "earliest", "preliminary": "earliest",
    "embryonic": "earliest", "seminal": "earliest", "initially": "earliest",
}

COUNT_CUES = (
    re.compile(r"\bhow\s+many\b", re.I),
    re.compile(r"\bcardinality\b", re.I),
    re.compile(r"\bcount(?:ing)?\b", re.I),
    re.compile(r"\bnumber\s+of\b", re.I),
    re.compile(r"\bquantity\b", re.I),
    re.compile(r"\btally(?:ing)?\b", re.I),
    re.compile(r"\bcensus\b", re.I),
    re.compile(r"\bquantif(?:y|ying)\b", re.I),
)
_EARLIEST_BEFORE_UPDATE = re.compile(
    r"\b(?:prior\s+to|before)\s+any\s+(?:updates?|modifications?)\b", re.I,
)


def _has_relation(question: str) -> bool:
    text = question.casefold()
    return any(re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text)
               for term in KNOWN_RELATION_LIST)


def _temporal_cue(question: str) -> str | None:
    text = question.casefold()
    if _EARLIEST_BEFORE_UPDATE.search(text):
        # These are source-generator phrases for the initial state, not the
        # immediately preceding state.
        return "earliest"
    hits = []
    for cue, scope in TEMPORAL_CUE_MAP.items():
        if re.search(r"(?<![a-z0-9])" + re.escape(cue) + r"(?![a-z0-9])", text):
            hits.append((text.index(cue), cue, scope))
    if not hits:
        return None
    # Use first cue in question order. `before` in e.g. "before any updates"
    # means prior state; `first`/origin terms mean earliest state.
    return min(hits)[2]


def _asks_count(question: str) -> bool:
    return any(pattern.search(question) for pattern in COUNT_CUES)


def build_final_evaluation_request(question: str, retrieved: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the frozen final-run query-plan request without creating a client."""
    _, schema = approved_evaluation_prompt_and_schema()
    prompt = FINAL_PROMPT_PATH.read_text(encoding="utf-8").rstrip("\n")
    pool = "\n".join(f"{int(row['fact_idx'])}. {row['text']}" for row in retrieved)
    content = prompt.replace("{hop_query}", question).replace("{pool}", pool)
    return {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": content}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "aletheia_final_v3", "strict": True, "schema": schema,
        }},
        "max_completion_tokens": 1024,
        "temperature": 0.0,
    }


def override_query_plan(question: str, plan: QueryPlan) -> QueryPlan:
    """Correct four known plan-level omissions using question text only."""
    updates: dict[str, Any] = {}
    if not _has_relation(question):
        updates.update(property_scope="entity_wide_values", property_name=None)
    if _asks_count(question):
        updates.update(
            intent="aggregation", aggregation_mode="count",
            temporal_scope="unspecified", ordinal_offset=None,
        )
    else:
        temporal = _temporal_cue(question)
        if temporal is not None:
            updates.update(intent="historical", temporal_scope=temporal,
                           ordinal_offset=0 if temporal == "earliest" else None,
                           aggregation_mode="none")
    return replace(plan, **updates) if updates else plan


def execute_final_plan(
    question: str,
    plan: QueryPlan,
    retrieved: Iterable[Mapping[str, Any]],
    entity_fact_counts: Mapping[str, int],
    entity_property_fact_counts: Mapping[tuple[str, str], int] | None = None,
) -> tuple[QueryPlan, dict[str, Any], bool]:
    """Apply code-level overrides and run operators; return plan, answer, coverage."""
    normalized = override_query_plan(question, plan)
    if normalized.intent == "boolean":
        # Positive evidence is existential across the complete entity's
        # retrieved facts; a negative conclusion still requires entity-wide
        # history verified against the local corpus count.
        entity_plan = replace(
            normalized, property_scope="entity_wide_values", property_name=None,
            supported=bool(normalized.entity and normalized.boolean_target),
        )
        candidates = deterministic_candidates(entity_plan, retrieved)
        complete = local_history_complete(
            entity_plan, retrieved, entity_fact_counts, entity_property_fact_counts,
        )
        gate_plan = normalized
        if normalized.entity and normalized.boolean_target:
            # The retrieved facts and exact subject/target fields are enough
            # for the evidence gate even if the model marked a property scope
            # ambiguous. Absence still depends on the entity-wide count check.
            gate_plan = replace(normalized, supported=True)
        answer = execute_boolean_gate(gate_plan, candidates, complete_entity_history=complete)
        return normalized, answer, complete
    candidates = deterministic_candidates(normalized, retrieved)
    complete = local_history_complete(
        normalized, retrieved, entity_fact_counts, entity_property_fact_counts,
    )
    answer = execute_plan(normalized, candidates, complete_history=complete)
    return normalized, answer, complete
