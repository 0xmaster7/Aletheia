"""Offline-safe primitives for the professor-feedback revision campaign.

This module deliberately has no OpenAI or Langfuse dependency. A future paid
runner can consume its validated plans, candidates, operators, scoring, and
cache keys through a single guarded client.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


INTENTS = {"current_value", "historical", "aggregation", "boolean"}
PROPERTY_SCOPES = {"specific", "entity_wide_values", "ambiguous"}
TEMPORAL_SCOPES = {"latest", "earliest", "previous", "ordinal", "any", "unspecified"}
AGGREGATION_MODES = {"count", "list", "none"}
EVALUATION_MAX_COMPLETION_TOKENS = 1024
EVALUATION_V2_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts/evaluation_v2.txt"
)

DIRECT_CONTROL_SYSTEM = (
    "Answer questions using only the supplied knowledge-pool facts. "
    "Respond concisely with the answer and do not add unsupported information."
)
DIRECT_CONTROL_USER = "[Knowledge Pool]\n{pool}\n\nQuestion: {question}\nAnswer:"
CHAIN_OF_THOUGHT_SYSTEM = (
    "Answer questions using only the supplied knowledge-pool facts. "
    "Reason through the relevant evidence before answering, then give a short rationale "
    "and a concise final answer. Do not add unsupported information."
)
CHAIN_OF_THOUGHT_USER = "[Knowledge Pool]\n{pool}\n\nQuestion: {question}"


def approved_evaluation_prompt_and_schema() -> tuple[str, dict[str, Any]]:
    """Load the frozen evaluation-only v2 prompt and plan-only schema.

    The production prompt in ``scripts/lib/_pipeline.py`` is intentionally not
    imported or changed by this evaluation path.
    """
    prompt = EVALUATION_V2_PROMPT_PATH.read_text(encoding="utf-8").rstrip("\n")
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["query_plan"],
        "properties": {
            "query_plan": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "intent", "entity", "property_scope", "property_name", "temporal_scope",
                    "ordinal_offset", "aggregation_mode", "boolean_target", "negated", "supported", "reason",
                ],
                "properties": {
                    "intent": {"type": "string", "enum": ["current_value", "historical", "aggregation", "boolean"]},
                    "entity": {"type": ["string", "null"]},
                    "property_scope": {"type": "string", "enum": ["specific", "entity_wide_values", "ambiguous"]},
                    "property_name": {"type": ["string", "null"]},
                    "temporal_scope": {"type": "string", "enum": ["latest", "earliest", "previous", "ordinal", "any", "unspecified"]},
                    "ordinal_offset": {"type": ["integer", "null"], "minimum": 0},
                    "aggregation_mode": {"type": "string", "enum": ["count", "list", "none"]},
                    "boolean_target": {"type": ["string", "null"]},
                    "negated": {"type": "boolean"},
                    "supported": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
            },
        },
    }
    return prompt, schema


def build_approved_evaluation_request(
    question: str, retrieved: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a no-client request payload for the approved evaluation path."""
    prompt, schema = approved_evaluation_prompt_and_schema()
    pool = "\n".join(f"{int(row['fact_idx'])}. {row['text']}" for row in retrieved)
    content = prompt.replace("{hop_query}", question).replace("{pool}", pool)
    return {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": content}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "aletheia_evaluation_v2", "strict": True, "schema": schema,
        }},
        "max_completion_tokens": EVALUATION_MAX_COMPLETION_TOKENS,
        "temperature": 0.0,
    }


def build_control_request(
    arm: str, question: str, retrieved: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build direct-answer or CoT control calls, without constructing a client."""
    if arm not in {"direct", "cot"}:
        raise ValueError("control arm must be 'direct' or 'cot'")
    pool = "\n".join(f"{int(row['fact_idx'])}. {row['text']}" for row in retrieved)
    system, user_template = (
        (DIRECT_CONTROL_SYSTEM, DIRECT_CONTROL_USER)
        if arm == "direct"
        else (CHAIN_OF_THOUGHT_SYSTEM, CHAIN_OF_THOUGHT_USER)
    )
    return {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_template.format(pool=pool, question=question)},
        ],
        "max_completion_tokens": 1024 if arm == "cot" else 512,
        "temperature": 0.0,
    }


@dataclass(frozen=True)
class QueryPlan:
    intent: str
    entity: str | None
    property_scope: str
    property_name: str | None
    temporal_scope: str
    ordinal_offset: int | None
    aggregation_mode: str
    boolean_target: str | None
    negated: bool
    supported: bool
    reason: str

    @classmethod
    def from_object(cls, obj: Any) -> "QueryPlan":
        if not isinstance(obj, Mapping):
            raise ValueError("query_plan must be a JSON object")
        required = {
            "intent", "entity", "property_scope", "property_name", "temporal_scope",
            "ordinal_offset", "aggregation_mode", "boolean_target", "negated",
            "supported", "reason",
        }
        if set(obj) != required:
            raise ValueError(f"query_plan keys must be exactly {sorted(required)}")
        if obj["intent"] not in INTENTS:
            raise ValueError("query_plan intent is unsupported")
        if obj["property_scope"] not in PROPERTY_SCOPES:
            raise ValueError("query_plan property_scope is invalid")
        if obj["temporal_scope"] not in TEMPORAL_SCOPES:
            raise ValueError("query_plan temporal_scope is invalid")
        if obj["aggregation_mode"] not in AGGREGATION_MODES:
            raise ValueError("query_plan aggregation_mode is invalid")
        for key in ("entity", "property_name", "boolean_target"):
            if obj[key] is not None and not isinstance(obj[key], str):
                raise ValueError(f"query_plan {key} must be string or null")
        if obj["ordinal_offset"] is not None and (
            isinstance(obj["ordinal_offset"], bool) or not isinstance(obj["ordinal_offset"], int)
            or obj["ordinal_offset"] < 0
        ):
            raise ValueError("ordinal_offset must be a nonnegative integer or null")
        if not isinstance(obj["negated"], bool) or not isinstance(obj["supported"], bool):
            raise ValueError("negated and supported must be booleans")
        if not isinstance(obj["reason"], str):
            raise ValueError("reason must be a string")
        plan = cls(**dict(obj))
        if plan.property_scope == "specific" and not plan.property_name:
            raise ValueError("specific property scope requires property_name")
        if plan.property_scope == "ambiguous" and plan.supported:
            raise ValueError("an ambiguous property plan cannot be supported")
        if plan.intent == "historical" and plan.temporal_scope not in {
            "latest", "earliest", "previous", "ordinal", "unspecified",
        }:
            raise ValueError("historical temporal_scope is inconsistent")
        if plan.intent == "aggregation" and plan.aggregation_mode not in {"count", "list"}:
            raise ValueError("aggregation requires count or list mode")
        if plan.intent == "boolean" and plan.boolean_target is None and plan.supported:
            raise ValueError("supported boolean plan requires a target")
        return plan

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Candidate:
    serial: int
    fact_text: str
    answer_entity: str


def parse_and_validate_candidates(
    obj: Any, retrieved: Iterable[Mapping[str, Any]],
) -> list[Candidate]:
    """Validate serial/text provenance and deduplicate identical returned facts."""
    if not isinstance(obj, Mapping) or set(obj) != {"candidates"}:
        raise ValueError("response must be an object with only a candidates key")
    raw = obj["candidates"]
    if not isinstance(raw, list):
        raise ValueError("candidates must be a list")
    allowed: dict[tuple[int, str], None] = {}
    for fact in retrieved:
        try:
            allowed[(int(fact["fact_idx"]), str(fact["text"]))] = None
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("retrieved fact is missing a valid fact_idx/text pair") from exc
    seen: set[tuple[int, str]] = set()
    result: list[Candidate] = []
    for i, item in enumerate(raw):
        if not isinstance(item, Mapping) or set(item) != {"serial", "fact_text", "answer_entity"}:
            raise ValueError(f"candidate {i} has an invalid schema")
        serial = item["serial"]
        text = item["fact_text"]
        answer = item["answer_entity"]
        if isinstance(serial, bool) or not isinstance(serial, int):
            raise ValueError(f"candidate {i} serial must be an integer")
        if not isinstance(text, str) or not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"candidate {i} fact_text/answer_entity is invalid")
        identity = (serial, text)
        if identity not in allowed:
            raise ValueError(f"candidate {i} does not match retrieved serial and fact_text")
        normalized_answer = normalize_value(answer)
        normalized_fact = normalize_value(text)
        if not normalized_answer or f" {normalized_answer} " not in f" {normalized_fact} ":
            raise ValueError(f"candidate {i} answer_entity is not stated in fact_text")
        if identity not in seen:
            seen.add(identity)
            result.append(Candidate(serial, text, answer.strip()))
    return result


def normalize_value(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def property_type_from_fact(text: str) -> str:
    """Mirror the released data generator's coarse property labels."""
    lowered = text.casefold()
    if "country" in lowered or "citizen of" in lowered or "citizenship" in lowered:
        return "country"
    if "genre" in lowered:
        return "genre"
    if any(cue in lowered for cue in ("located", "capital", "born in", "died in", "worked in")):
        return "location"
    if "language" in lowered or "speaks" in lowered:
        return "language"
    if "religion" in lowered:
        return "religion"
    if "nationality" in lowered:
        return "nationality"
    if "ethnicity" in lowered:
        return "ethnicity"
    if "occupation" in lowered:
        return "occupation"
    if "created by" in lowered or "directed by" in lowered:
        return "creator"
    return "value"


_FACT_VALUE_PATTERNS = (
    re.compile(r"^(.+?)\s+speaks the language of\s+(.+)$", re.I),
    re.compile(
        r"^(.+?)\s+(?:was created in the country of|has the genre of|was located in|"
        r"is located in|was born in|has the nationality of|was created by|has the capital of|"
        r"was written in|has the official language of|is the capital of|was founded in|"
        r"has the population of|was produced by|has the religion of|was directed by|"
        r"was published in|was released in|has the ethnicity of|was performed by|"
        r"is a member of|has the occupation of|was invented by|is the currency of|"
        r"was composed by|was discovered by|is in the continent of|has the language of|"
        r"was designed by|is the leader of|has the currency of|was manufactured by|"
        r"is associated with|has the owner of|is the owner of|was owned by|is owned by)\s+(.+)$",
        re.I,
    ),
)
_FACT_FALLBACK_PATTERN = re.compile(
    r"^(.+?)\s+(was|is|has|had)\s+(.+?)\s+(?:of|in|by|from|at|to)\s+(.+)$", re.I,
)


def parse_fact_template(fact_text: str) -> tuple[str, str] | None:
    """Return (subject, value) using the released benchmark fact templates."""
    text = str(fact_text).strip().rstrip(".")
    for pattern in _FACT_VALUE_PATTERNS:
        match = pattern.match(text)
        if match:
            return match.group(1).strip(), match.group(2).strip()
    match = _FACT_FALLBACK_PATTERN.match(text)
    if match:
        return match.group(1).strip(), match.group(4).strip()
    return None


def _canonical_property(name: str | None) -> str | None:
    if name is None:
        return None
    value = normalize_value(name)
    aliases = {
        "citizenship": "country", "nationality": "nationality", "language": "language",
        "religion": "religion", "location": "location", "city": "location",
        "genre": "genre", "occupation": "occupation", "creator": "creator",
        "ethnicity": "ethnicity", "country": "country",
    }
    return aliases.get(value, value)


def deterministic_candidates(
    plan: QueryPlan, retrieved: Iterable[Mapping[str, Any]],
) -> list[Candidate]:
    """Select exact subject/property facts and derive value spans locally."""
    if not plan.supported or not plan.entity or plan.property_scope == "ambiguous":
        return []
    selected: list[Candidate] = []
    seen: set[tuple[int, str]] = set()
    wanted_property = _canonical_property(plan.property_name)
    for row in retrieved:
        serial = row.get("fact_idx", row.get("serial"))
        text = row.get("text", row.get("fact_text"))
        if isinstance(serial, bool) or not isinstance(serial, int) or not isinstance(text, str):
            continue
        parsed = parse_fact_template(text)
        if parsed is None:
            continue
        subject, value = parsed
        if subject != plan.entity or not value:
            continue
        if plan.property_scope == "specific" and _canonical_property(property_type_from_fact(text)) != wanted_property:
            continue
        identity = (serial, text)
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(Candidate(serial, text, value))
    return selected


def local_history_complete(
    plan: QueryPlan,
    retrieved: Iterable[Mapping[str, Any]],
    entity_fact_counts: Mapping[str, int],
    entity_property_fact_counts: Mapping[tuple[str, str], int] | None = None,
) -> bool:
    """Verify completeness by comparing retrieved local counts to corpus counts.

    Boolean negatives require the complete entity history, even for a named
    property. Temporal and aggregation operators compare the requested scope.
    """
    if not plan.entity or plan.entity not in entity_fact_counts:
        return False
    all_rows = deterministic_candidates(
        QueryPlan(**{**plan.as_dict(), "property_scope": "entity_wide_values", "property_name": None}),
        retrieved,
    )
    if plan.intent == "boolean":
        expected = entity_fact_counts[plan.entity]
        return len({fact.serial for fact in all_rows}) == expected
    scoped = deterministic_candidates(plan, retrieved)
    if plan.property_scope == "specific" and entity_property_fact_counts is not None:
        key = (plan.entity, _canonical_property(plan.property_name) or "value")
        expected = entity_property_fact_counts.get(key)
        if expected is None:
            return False
    else:
        expected = entity_fact_counts[plan.entity]
    return len({fact.serial for fact in scoped}) == expected


def execute_boolean_gate(
    plan: QueryPlan, candidates: Iterable[Candidate], *, complete_entity_history: bool,
) -> dict[str, Any]:
    """Apply the evaluation-set existential gate: evidence proves True; only a
    verified complete local entity history can prove absence/False.
    """
    if plan.intent != "boolean" or not plan.supported or not plan.boolean_target:
        return {"answer": None, "status": "unsupported", "reason": plan.reason or "boolean plan unsupported"}
    target = normalize_value(plan.boolean_target)
    matches = any(normalize_value(fact.answer_entity) == target for fact in candidates)
    if matches:
        return {"answer": not plan.negated, "status": "answered"}
    if not complete_entity_history:
        return {"answer": None, "status": "insufficient_evidence", "reason": "absence cannot be established from partial retrieval"}
    return {"answer": plan.negated, "status": "answered"}


def filter_candidates(plan: QueryPlan, candidates: Iterable[Candidate]) -> list[Candidate]:
    """Keep only subject/property evidence the plan can verify locally."""
    if not plan.supported or not plan.entity:
        return []
    kept: list[Candidate] = []
    for candidate in candidates:
        text = candidate.fact_text.strip()
        parsed_fact = parse_fact_template(text)
        if not parsed_fact or parsed_fact[0] != plan.entity.strip():
            continue
        if plan.property_scope == "ambiguous":
            continue
        if plan.property_scope == "specific":
            if _canonical_property(property_type_from_fact(text)) != _canonical_property(plan.property_name):
                continue
        kept.append(candidate)
    return kept


def execute_plan(
    plan: QueryPlan, candidates: Iterable[Candidate], *, complete_history: bool = False,
) -> dict[str, Any]:
    """Execute supported plans; require explicit corpus coverage for exhaustive claims."""
    facts = filter_candidates(plan, candidates)
    if not plan.supported:
        return {"answer": None, "status": "unsupported", "reason": plan.reason or "plan unsupported"}
    if plan.property_scope == "ambiguous":
        return {"answer": None, "status": "unsupported", "reason": "ambiguous property scope"}
    if not facts and not (plan.intent == "boolean" and complete_history):
        return {"answer": None, "status": "insufficient_evidence", "reason": "no validated subject/property candidates"}
    ordered = sorted(facts, key=lambda item: item.serial)
    if plan.intent == "current_value":
        if not complete_history:
            return {"answer": None, "status": "insufficient_evidence", "reason": "latest value requires complete history"}
        return {"answer": max(ordered, key=lambda item: item.serial).answer_entity, "status": "answered"}
    if plan.intent == "historical":
        if not complete_history:
            return {"answer": None, "status": "insufficient_history", "reason": "temporal selection requires complete history"}
        if plan.temporal_scope == "latest":
            selected = ordered[-1]
        elif plan.temporal_scope == "earliest" or plan.temporal_scope == "unspecified":
            selected = ordered[0]
        else:
            # Collapse consecutive same-value states before interpreting offset.
            timeline: list[Candidate] = []
            last: str | None = None
            for item in reversed(ordered):
                value = normalize_value(item.answer_entity)
                if value and value != last:
                    timeline.append(item)
                    last = value
            offset = 1 if plan.temporal_scope == "previous" else plan.ordinal_offset
            if offset is None or offset < 1 or offset >= len(timeline):
                return {"answer": None, "status": "insufficient_history", "reason": "requested prior state is unavailable"}
            selected = timeline[offset]
        return {"answer": selected.answer_entity, "status": "answered", "serial": selected.serial}
    if plan.intent == "aggregation":
        if not complete_history:
            return {"answer": None, "status": "insufficient_evidence", "reason": "aggregation requires complete history"}
        values: list[str] = []
        seen: set[str] = set()
        for item in ordered:
            norm = normalize_value(item.answer_entity)
            if norm and norm not in seen:
                seen.add(norm)
                values.append(item.answer_entity)
        if plan.aggregation_mode == "count":
            return {"answer": len(values), "status": "answered", "values": values}
        return {"answer": values, "status": "answered"}
    if plan.intent == "boolean":
        return execute_boolean_gate(
            plan, ordered, complete_entity_history=complete_history,
        )
    return {"answer": None, "status": "unsupported", "reason": "unhandled intent"}


def subem(predicted: Any, gold: str) -> bool:
    """Case-insensitive substring match used by the paper's SubEM protocol."""
    return str(gold).casefold() in str(predicted or "").casefold()


def strict_integer(predicted: Any, gold: str) -> bool:
    match = re.fullmatch(r"\s*([+-]?\d+)\s*", str(predicted))
    return bool(match and match.group(1) == str(gold).strip())


def canonical_boolean(predicted: Any, gold: str) -> bool:
    return str(predicted).strip().casefold() == str(gold).strip().casefold() and str(gold).strip().casefold() in {"true", "false"}


def normalized_historical(predicted: Any, gold: str, aliases: Iterable[str] = ()) -> bool:
    target = normalize_value(gold)
    allowed = {target, *(normalize_value(alias) for alias in aliases)}
    return normalize_value(str(predicted)) in allowed


def request_cache_key(
    *, model: str, messages: list[dict[str, Any]], temperature: float,
    seed: int | None, max_output_tokens: int, response_format: Mapping[str, Any],
    adapter_version: str,
) -> str:
    payload = {
        "model": model, "messages": messages, "temperature": temperature,
        "seed": seed, "max_output_tokens": max_output_tokens,
        "response_format": response_format, "adapter_version": adapter_version,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
