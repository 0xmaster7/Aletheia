"""BM25 baseline + CAR v2 pipelines, fully instrumented with Langfuse.

Every operation gets its own span:

  bm25_retrieval = [ tokenize_query → bm25_score → rank_topk ]
  car_v2 hop  = [ substitute_template → bm25_retrieval → format_pool →
                  extract_candidates_llm → parse_candidates_json → freshness_pick ]
"""
from __future__ import annotations
from functools import lru_cache
import json
import math
import re
import time
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from _lf import OpenAI, observe, get_client

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None

from semantic_router import Route, SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder

encoder = HuggingFaceEncoder(name="sentence-transformers/all-MiniLM-L6-v2")

route_current_value = Route(
    name="current_value",
    utterances=[
        "What is the current value?",
        "Who is the current person?",
        "What is the latest state?",
        "Which answer is true now?",
        "What is the newest recorded fact?"
    ]
)

route_historical = Route(
    name="historical",
    utterances=[
        "What was the previous value before the current one?",
        "Who held the role before now?",
        "What was the earlier state?",
        "What was the original recorded value?",
        "Show the state before the latest update."
    ]
)

route_boolean = Route(
    name="boolean",
    utterances=[
        "Is the current state still true?",
        "Does the latest record match this statement?",
        "Was this property true in the newest fact?",
        "Can you verify this yes or no?",
        "Is this claim correct right now?"
    ]
)

route_aggregation = Route(
    name="aggregation",
    utterances=[
        "How many unique values have appeared?",
        "List all historical values.",
        "Count the distinct entries in memory.",
        "Show every previous state.",
        "What are all recorded versions?"
    ]
)

routes = [route_current_value, route_historical, route_boolean, route_aggregation]
rl = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")

# ────────────────────────────────────────────────────────────────────────────
# Shared constants — matching MemoryAgentBench's RAG/FactConsolidation config
# ────────────────────────────────────────────────────────────────────────────
import os as _os
MODEL = _os.environ.get("PIPELINE_MODEL", "gpt-4o-mini")
ROUTER_MODEL = _os.environ.get("ROUTER_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SYSTEM_MESSAGE = "You are a helpful assistant that can read the context and memorize it for future retrieval."
TOP_K = 10
KNOWN_RELEVANT_Q1 = {6954, 10906, 3070, 7164, 7774}  # ground-truth chain facts for Q1

ROUTE_UTTERANCES = {
    "freshness": [
        "What is the current value?",
        "Who is the current person?",
        "What is the latest state?",
        "Which answer is true now?",
        "What is the newest recorded fact?",
    ],
    "temporal": [
        "What was the previous value before the current one?",
        "Who held the role before now?",
        "What was the earlier state?",
        "What was the original recorded value?",
        "Show the state before the latest update.",
    ],
    "boolean": [
        "Is the current state still true?",
        "Does the latest record match this statement?",
        "Was this property true in the newest fact?",
        "Can you verify this yes or no?",
        "Is this claim correct right now?",
    ],
    "aggregation": [
        "How many unique values have appeared?",
        "List all historical values.",
        "Count the distinct entries in memory.",
        "Show every previous state.",
        "What are all recorded versions?",
    ],
}
ROUTE_TOP_K = {"freshness": TOP_K, "temporal": 25, "boolean": TOP_K, "aggregation": 40}
ROUTE_HINT_PATTERNS = {
    "freshness": [
        r"\bcurrent\b", r"\blatest\b", r"\bnewest\b", r"\bnow\b",
        r"\bcurrently\b", r"\bpresent\b", r"\bstill\b",
    ],
    "temporal": [
        r"\bprevious\b", r"\bbefore\b", r"\bearlier\b", r"\bprior\b",
        r"\bformer\b", r"\bused to\b", r"\boriginal\b", r"\binitial(?:ly)?\b",
        r"\bfirst\b", r"\bold(?:est)?\b",
    ],
    "boolean": [
        r"^(?:is|are|was|were|am|do|does|did|has|have|had|can|could|will|would|should)\b",
    ],
    "aggregation": [
        r"^(?:how many|count|list|show all|name all)\b",
        r"\bnumber of\b", r"\btotal\b", r"\ball previous\b",
        r"\bhistory of\b", r"\ball recorded\b",
    ],
}
TEMPORAL_OLDEST_PATTERNS = [
    r"\boriginal\b", r"\binitial(?:ly)?\b", r"\bfirst\b", r"\bearliest\b", r"\boldest\b",
]
ORDINAL_WORD_TO_OFFSET = {
    "first": 0,
    "second": 1,
    "third": 2,
    "fourth": 3,
    "fifth": 4,
    "1st": 0,
    "2nd": 1,
    "3rd": 2,
    "4th": 3,
    "5th": 4,
}
NEGATION_HINTS = (" not ", " never ", " no longer ")

QUERY_TEMPLATE_BM25 = (
    "Pretend you are a knowledge management system. Each fact in the knowledge pool is "
    "provided with a serial number at the beginning, and the newer fact has larger serial number. "
    "\n You need to solve the conflicts of facts in the knowledge pool by finding the newest "
    "fact with larger serial number. You need to answer a question based on this rule. "
    "You should give a very concise answer without saying other words for the question "
    "**only** from the knowledge pool you have memorized rather than the real facts in real "
    "world. \n\nFor example:\n\n [Knowledge Pool] \n\n Question: Based on the provided "
    "Knowledge Pool, what is the name of the current president of Russia? \n"
    "Answer: Donald Trump \n\n Now Answer the Question: Based on the provided Knowledge "
    "Pool, {question} \nAnswer:"
)


def tokenize(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", s.lower())


def _normalize_text(s: str) -> str:
    return " ".join(tokenize(s))


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _lexical_similarity(query: str, utterance: str) -> float:
    q_tokens = tokenize(query)
    u_tokens = tokenize(utterance)
    if not q_tokens or not u_tokens:
        return 0.0
    q_set = set(q_tokens)
    u_set = set(u_tokens)
    overlap = len(q_set & u_set)
    if overlap == 0:
        return 0.0
    return overlap / math.sqrt(len(q_set) * len(u_set))


@lru_cache(maxsize=1)
def _get_sentence_encoder() -> Any | None:
    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer(ROUTER_MODEL)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _get_route_embeddings() -> dict[str, np.ndarray] | None:
    encoder = _get_sentence_encoder()
    if encoder is None:
        return None

    flattened: list[str] = []
    route_slices: dict[str, tuple[int, int]] = {}
    start = 0
    for route_name, utterances in ROUTE_UTTERANCES.items():
        flattened.extend(utterances)
        route_slices[route_name] = (start, start + len(utterances))
        start += len(utterances)

    try:
        embeddings = np.asarray(
            encoder.encode(flattened, normalize_embeddings=True),
            dtype=float,
        )
    except Exception:
        return None

    return {
        route_name: embeddings[s:e]
        for route_name, (s, e) in route_slices.items()
    }


def _route_bonus(question: str, route_name: str) -> tuple[float, list[str]]:
    lowered = _normalize_text(question)
    matched = [
        pattern for pattern in ROUTE_HINT_PATTERNS[route_name]
        if re.search(pattern, lowered, flags=re.IGNORECASE)
    ]
    if route_name == "aggregation":
        return 0.9 * len(matched), matched
    if route_name == "boolean":
        return 0.8 * len(matched), matched
    if route_name == "temporal":
        return 0.75 * len(matched), matched
    if matched:
        return 0.35 * len(matched), matched
    return 0.1, []


@observe(name="route_query", as_type=None)
def route_query(question: str) -> dict[str, Any]:
    route_embeddings = _get_route_embeddings()
    embedding_mode = route_embeddings is not None
    query_embedding = None
    if embedding_mode:
        encoder = _get_sentence_encoder()
        try:
            query_embedding = np.asarray(
                encoder.encode([question], normalize_embeddings=True)[0],
                dtype=float,
            )
        except Exception:
            embedding_mode = False
            query_embedding = None

    scores: dict[str, float] = {}
    components: dict[str, dict[str, Any]] = {}
    for route_name, utterances in ROUTE_UTTERANCES.items():
        lexical_score = max(_lexical_similarity(question, utterance) for utterance in utterances)
        embedding_score = 0.0
        if embedding_mode and query_embedding is not None and route_embeddings is not None:
            embedding_score = max(
                _cosine_similarity(query_embedding, utterance_embedding)
                for utterance_embedding in route_embeddings[route_name]
            )
        bonus, matched_patterns = _route_bonus(question, route_name)
        score = bonus + (0.55 * lexical_score) + (0.45 * embedding_score)
        scores[route_name] = round(score, 6)
        components[route_name] = {
            "lexical": round(lexical_score, 6),
            "embedding": round(embedding_score, 6),
            "bonus": round(bonus, 6),
            "matched_patterns": matched_patterns,
        }

    selected_route = max(scores, key=scores.get)
    result = {
        "route": selected_route,
        "scores": scores,
        "components": components,
        "method": "embeddings+heuristics" if embedding_mode else "heuristics-only",
        "router_model": ROUTER_MODEL if embedding_mode else None,
    }
    get_client().update_current_span(
        input={"question": question},
        output=result,
    )
    return result


# ────────────────────────────────────────────────────────────────────────────
# Sub-spans for BM25 retrieval (used by BOTH baselines and CAR hops)
# ────────────────────────────────────────────────────────────────────────────
@observe(name="tokenize_query", as_type=None)
def _tokenize_query(query: str) -> list[str]:
    tokens = tokenize(query)
    get_client().update_current_span(
        input={"query": query, "tokenizer": "lowercase + re.findall(r'[A-Za-z0-9]+')"},
        output={"tokens": tokens, "n_tokens": len(tokens)},
    )
    return tokens


@observe(name="bm25_score", as_type=None)
def _bm25_score(bm25: BM25Okapi, query_tokens: list[str]) -> np.ndarray:
    t0 = time.time()
    scores = bm25.get_scores(query_tokens)
    elapsed_ms = int((time.time() - t0) * 1000)
    get_client().update_current_span(
        input={"n_query_tokens": len(query_tokens), "n_documents_in_index": len(scores)},
        output={
            "max_score": float(scores.max()),
            "mean_score": float(scores.mean()),
            "n_nonzero_scores": int((scores > 0).sum()),
            "elapsed_ms": elapsed_ms,
        },
    )
    return scores


@observe(name="rank_topk", as_type=None)
def _rank_topk(scores: np.ndarray, fact_indices: list[int], fact_texts: list[str],
               top_k: int) -> list[dict[str, Any]]:
    top_idx = np.argsort(scores)[::-1][:top_k]
    retrieved = [
        {"rank": r + 1, "fact_idx": int(fact_indices[i]), "score": float(scores[i]),
         "text": fact_texts[i]}
        for r, i in enumerate(top_idx)
    ]
    get_client().update_current_span(
        input={"top_k": top_k, "n_documents_ranked": len(scores)},
        output={
            "top_score": retrieved[0]["score"] if retrieved else 0.0,
            "bottom_score": retrieved[-1]["score"] if retrieved else 0.0,
            "n_returned": len(retrieved),
        },
    )
    return retrieved


@observe(name="bm25_retrieval", as_type=None)
def bm25_retrieve(bm25: BM25Okapi, query: str, fact_indices: list[int],
                  fact_texts: list[str], top_k: int = TOP_K) -> list[dict[str, Any]]:
    """Top-level BM25 retrieval — orchestrates tokenize → score → rank."""
    query_tokens = _tokenize_query(query)
    scores = _bm25_score(bm25, query_tokens)
    retrieved = _rank_topk(scores, fact_indices, fact_texts, top_k)
    retrieved_indices = {r["fact_idx"] for r in retrieved}
    get_client().update_current_span(
        input={"query": query, "top_k": top_k},
        output={
            "n_retrieved": len(retrieved),
            "retrieved": retrieved,
            "recall_known_relevant_q1": f"{len(KNOWN_RELEVANT_Q1 & retrieved_indices)}/{len(KNOWN_RELEVANT_Q1)}",
        },
    )
    return retrieved


# ────────────────────────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────────────────────────
@observe(name="evaluation", as_type=None)
def evaluate_answer(predicted: str, ground_truth_list: list[str]) -> dict[str, Any]:
    is_correct = any(gt.lower() in (predicted or "").lower() for gt in ground_truth_list)
    get_client().update_current_span(
        input={"predicted": predicted, "ground_truth": ground_truth_list, "method": "SubEM"},
        output={"is_correct_subem": is_correct},
    )
    get_client().score_current_trace(
        name="correctness", value=1.0 if is_correct else 0.0,
        comment=f"predicted={predicted!r}, ground_truth={ground_truth_list}",
    )
    return {"is_correct_subem": is_correct, "predicted": predicted, "ground_truth": ground_truth_list}


# ────────────────────────────────────────────────────────────────────────────
# BM25 baseline pipeline
# ────────────────────────────────────────────────────────────────────────────
@observe(name="format_prompt", as_type=None)
def _format_bm25_prompt(question: str, retrieved: list[dict[str, Any]]) -> str:
    knowledge_pool = "\n".join(f"{r['fact_idx']}. {r['text']}." for r in retrieved)
    user_msg = f"[Knowledge Pool]\n{knowledge_pool}\n\n{QUERY_TEMPLATE_BM25.format(question=question)}"
    get_client().update_current_span(
        input={"question": question, "n_retrieved": len(retrieved)},
        output={"prompt_chars": len(user_msg), "knowledge_pool_chars": len(knowledge_pool)},
    )
    return user_msg


@observe(name="bm25_baseline")
def run_bm25_baseline(question: str, question_index: int, ground_truth: list[str],
                      bm25: BM25Okapi, fact_indices: list[int], fact_texts: list[str],
                      client: OpenAI, temperature: float = 0.7,
                      dataset_name: str = "factconsolidation_mh_262k",
                      competency: str = "Conflict_Resolution") -> dict[str, Any]:
    # Append dataset + question identifier to the trace name for filtering
    get_client().update_current_span(
        name=f"bm25_baseline_{dataset_name}_q{question_index + 1}",
        metadata={
            "system": "BM25 + gpt-4o-mini",
            "model": MODEL, "temperature": temperature, "top_k": TOP_K,
            "question_index": question_index, "ground_truth": ground_truth,
            "experiment": "bm25_baseline",
            "dataset": dataset_name,
            "competency": competency,
        },
        input={"question": question},
    )

    retrieved = bm25_retrieve(bm25, question, fact_indices, fact_texts, TOP_K)
    user_msg = _format_bm25_prompt(question, retrieved)
    resp = client.chat.completions.create(
        model=MODEL, temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_msg},
        ],
        name="bm25_llm_answer",
    )
    answer = resp.choices[0].message.content or ""
    eval_result = evaluate_answer(answer, ground_truth)
    get_client().update_current_span(output={"answer": answer, "is_correct": eval_result["is_correct_subem"]})

    return {
        "answer": answer, "is_correct": eval_result["is_correct_subem"],
        "retrieved": retrieved,
        "usage": {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens},
    }


# ────────────────────────────────────────────────────────────────────────────
# CAR v2 pipeline
# ────────────────────────────────────────────────────────────────────────────
DECOMP_PROMPT = """<TASK>
Decompose this multi-hop question into a chain of atomic single-hop queries. Each hop must ask about ONE relationship only.
</TASK>

<HARD_CONSTRAINT>
A single hop query may contain AT MOST ONE relationship word ("of", "by", "from", "in", "where", "associated with", "related to"). If a hop has TWO or more such words connecting entity descriptions, it is INVALID — split it into multiple hops.
</HARD_CONSTRAINT>

<RULES>
1. Each hop asks ONE question about ONE specific entity. The entity must be named in the original question OR be the answer of a previous hop, referenced via {hop_N_answer}.
2. Chain INSIDE-OUT: start from the innermost named entity in the question, then move outward one relationship at a time.
3. The hop count equals the number of relationship words in the question — never fewer.
4. No padding hops. Every hop's answer must feed a later hop or be the final answer.
</RULES>

<EXAMPLE>
<QUESTION>"What is the country of the company where the manager of the supervisor of Alice works?"</QUESTION>

<RELATIONSHIP_COUNT>4 — "supervisor of", "manager of", "company where … works", "country of"</RELATIONSHIP_COUNT>

<CORRECT>
{"hops": [
  {"id": 1, "query": "Who is the supervisor of Alice?"},
  {"id": 2, "query": "Who is the manager of {hop_1_answer}?"},
  {"id": 3, "query": "What company does {hop_2_answer} work at?"},
  {"id": 4, "query": "What country is {hop_3_answer} based in?"}
]}
</CORRECT>

<WRONG reason="hop 1 has 2 'of' words; hop 2 has 2 relationships — both violate HARD_CONSTRAINT">
{"hops": [
  {"id": 1, "query": "Who is the manager of the supervisor of Alice?"},
  {"id": 2, "query": "What country is the company where {hop_1_answer} works based in?"}
]}
</WRONG>

<WRONG reason="single hop contains 4 'of' words; entire chain compressed into one query">
{"hops": [
  {"id": 1, "query": "What country is the company where the manager of the supervisor of Alice works based in?"}
]}
</WRONG>
</EXAMPLE>

<QUESTION>
{question}
</QUESTION>

Return ONLY valid JSON: {"hops": [{"id": 1, "query": "..."}, ...]}"""


@observe(name="decompose")
def _decompose(client: OpenAI, question: str) -> list[dict[str, Any]]:
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system",
             "content": "You are a question decomposition assistant. NEVER combine multiple relationship words into a single hop query."},
            {"role": "user", "content": DECOMP_PROMPT.replace("{question}", question)},
        ],
        name="decompose_llm",
    )
    raw = resp.choices[0].message.content
    hops = json.loads(raw)["hops"]
    get_client().update_current_span(
        input={"question": question},
        output={"n_hops": len(hops), "hops": hops, "raw_llm_output_chars": len(raw)},
    )
    return hops


@observe(name="substitute_template", as_type=None)
def _substitute_template(raw_query: str, prior_results: list[dict[str, Any]]) -> str:
    """Replace {hop_N_answer} placeholders with the actual answers from prior hops."""
    resolved = raw_query
    substitutions = {}
    for prior in prior_results:
        placeholder = f"{{hop_{prior['hop_id']}_answer}}"
        if placeholder in resolved:
            substitutions[placeholder] = prior["answer"] or ""
            resolved = resolved.replace(placeholder, prior["answer"] or "")
    get_client().update_current_span(
        input={"raw_query": raw_query, "n_prior_hops": len(prior_results)},
        output={"resolved_query": resolved, "substitutions_applied": substitutions},
    )
    return resolved


CANDIDATE_PROMPT = """You are given retrieved items from a knowledge pool. Each item has a FRESHNESS marker (the prefix integer) — higher marker = newer version.

Your job: identify EVERY item that DIRECTLY answers the question, and extract the answer entity from each.

Do NOT compare freshness markers. Do NOT pick a "best" one. Include ALL items that match.

Rules:
1. An item directly answers the question only if BOTH its subject AND its predicate exactly match what the question asks about. The predicate noun used in the question (e.g., "spouse", "sport", "profession", "religion") must be the same noun used in the item. Related-but-different predicates do NOT match.
2. The subject named in the question must appear verbatim in the matching item. A different entity with a similar name does NOT match.
3. If a subject has multiple conflicting values (e.g., the same person with two different values at different freshness markers), INCLUDE BOTH as separate candidates. Do not pick.
4. If no item answers the question, return an empty list.
5. Copy the item's text verbatim into `fact_text`.

Question: {hop_query}

Items:
{pool}

Return ONLY valid JSON: {"candidates": [{"serial": <int>, "fact_text": "<verbatim>", "answer_entity": "<extracted>"}, ...]}"""


@observe(name="format_pool", as_type=None)
def _format_pool(retrieved: list[dict[str, Any]]) -> str:
    pool_str = "\n".join(f"{r['fact_idx']}. {r['text']}" for r in retrieved)
    get_client().update_current_span(
        input={"n_retrieved": len(retrieved)},
        output={"pool_chars": len(pool_str), "pool_preview": pool_str[:300]},
    )
    return pool_str


@observe(name="parse_candidates_json", as_type=None)
def _parse_candidates(raw_response: str) -> list[dict[str, Any]]:
    try:
        candidates = json.loads(raw_response).get("candidates", [])
        get_client().update_current_span(
            input={"raw_chars": len(raw_response)},
            output={"n_candidates": len(candidates), "candidates": candidates, "parse_ok": True},
        )
        return candidates
    except json.JSONDecodeError as e:
        get_client().update_current_span(
            input={"raw_chars": len(raw_response)},
            output={"parse_ok": False, "error": str(e), "raw_preview": raw_response[:200]},
            level="ERROR",
        )
        return []


@observe(name="extract_candidates")
def _extract_candidates(client: OpenAI, hop_query: str, retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pool_str = _format_pool(retrieved)
    # No system prompt — response_format={"type":"json_object"} enforces JSON
    prompt_content = CANDIDATE_PROMPT.replace("{hop_query}", hop_query).replace("{pool}", pool_str)
    if "The Unit" in hop_query:
        print("\n========== DEBUG EXTRACT CANDIDATES ==========")
        print("=== PROMPT ===")
        print(prompt_content)

    resp = client.chat.completions.create(
        model=MODEL, temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "user", "content": prompt_content},
        ],
        name="extract_candidates_llm",
    )
    raw = resp.choices[0].message.content or "{}"
    if "The Unit" in hop_query:
        print("\n=== RAW LLM RESPONSE ===")
        print(repr(raw))

    candidates = _parse_candidates(raw)
    if "The Unit" in hop_query:
        print("\n=== PARSED CANDIDATES ===")
        print(candidates)
        print("==============================================\n")
        
    get_client().update_current_span(
        input={"hop_query": hop_query, "n_retrieved": len(retrieved)},
        output={"n_candidates": len(candidates), "candidates": candidates},
    )
    return candidates


@observe(name="freshness_pick", as_type=None)
def _freshness_pick(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        get_client().update_current_span(
            input={"candidates": []}, output={"chosen": None, "note": "no candidates"},
        )
        return None
    chosen = max(candidates, key=lambda c: c["serial"])
    rejected = [c for c in candidates if c["serial"] != chosen["serial"]]
    get_client().update_current_span(
        input={
            "n_candidates": len(candidates),
            "candidates": candidates,
            "all_serials": sorted(c["serial"] for c in candidates),
        },
        output={
            "chosen_serial": chosen["serial"],
            "chosen_answer": chosen["answer_entity"],
            "chosen_fact_text": chosen["fact_text"],
            "rejected_candidates": [{"serial": r["serial"], "answer": r["answer_entity"]} for r in rejected],
            "method": "max(serial) — deterministic Python",
        },
    )
    return chosen


def _collapse_state_timeline(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse repeated consecutive values so "previous state" means a state change."""
    timeline: list[dict[str, Any]] = []
    last_value = None
    for candidate in sorted(candidates, key=lambda c: c["serial"], reverse=True):
        value = _normalize_text(candidate.get("answer_entity", ""))
        if value and value != last_value:
            timeline.append(candidate)
            last_value = value
    return timeline


def _temporal_offset(question: str) -> int | None:
    lowered = _normalize_text(question)
    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in TEMPORAL_OLDEST_PATTERNS):
        return None
    for ordinal_word, offset in ORDINAL_WORD_TO_OFFSET.items():
        if re.search(rf"\b{re.escape(ordinal_word)}\b", lowered) and re.search(
            r"\b(previous|before|earlier|prior|former|latest|newest|recent)\b", lowered
        ):
            return offset
    if re.search(r"\b(previous|before|earlier|prior|former|last)\b", lowered):
        return 1
    return 0


@observe(name="historical_pick", as_type=None)
def _historical_pick(candidates: list[dict[str, Any]], question: str) -> dict[str, Any] | None:
    timeline = _collapse_state_timeline(candidates)
    if not timeline:
        get_client().update_current_span(
            input={"question": question, "n_candidates": 0},
            output={"chosen": None, "timeline_length": 0},
        )
        return None

    offset = _temporal_offset(question)
    if offset is None:
        chosen = min(timeline, key=lambda c: c["serial"])
        selection_reason = "oldest-state"
    else:
        chosen = timeline[min(offset, len(timeline) - 1)]
        selection_reason = f"state_offset={min(offset, len(timeline) - 1)}"
    get_client().update_current_span(
        input={"question": question, "n_candidates": len(candidates)},
        output={
            "chosen": chosen,
            "timeline_length": len(timeline),
            "selection_reason": selection_reason,
            "timeline_serials": [candidate["serial"] for candidate in timeline],
        },
    )
    return chosen


def _question_mentions_phrase(question: str, phrase: str) -> bool:
    norm_question = f" {_normalize_text(question)} "
    norm_phrase = _normalize_text(phrase)
    if not norm_phrase:
        return False
    if f" {norm_phrase} " in norm_question:
        return True
    phrase_tokens = set(norm_phrase.split())
    question_tokens = set(norm_question.split())
    return bool(phrase_tokens) and phrase_tokens.issubset(question_tokens)


@observe(name="boolean_gate", as_type=None)
def _boolean_gate(question: str, chosen: dict[str, Any] | None) -> bool:
    if not chosen:
        get_client().update_current_span(
            input={"question": question, "chosen": None},
            output={"result": False, "reason": "no-latest-state"},
        )
        return False

    target_matches = _question_mentions_phrase(question, chosen.get("answer_entity", ""))
    if not target_matches:
        target_matches = _question_mentions_phrase(question, chosen.get("fact_text", ""))
    lowered = f" {_normalize_text(question)} "
    negated = any(hint in lowered for hint in NEGATION_HINTS)
    result = (not target_matches) if negated else target_matches
    get_client().update_current_span(
        input={"question": question, "chosen": chosen},
        output={
            "result": result,
            "target_matches_latest_state": target_matches,
            "negated_question": negated,
        },
    )
    return result


@observe(name="aggregate_answers", as_type=None)
def _aggregate_answers(candidates: list[dict[str, Any]], question: str) -> dict[str, Any]:
    ordered_unique: list[str] = []
    seen: set[str] = set()
    for candidate in sorted(candidates, key=lambda c: c["serial"]):
        answer = candidate.get("answer_entity", "").strip()
        normalized = _normalize_text(answer)
        if answer and normalized and normalized not in seen:
            seen.add(normalized)
            ordered_unique.append(answer)

    lowered = _normalize_text(question)
    is_count_query = bool(re.search(r"\b(how many|count|number of|total)\b", lowered))
    answer = str(len(ordered_unique)) if is_count_query else json.dumps(ordered_unique)
    result = {
        "answer": answer,
        "unique_answers": ordered_unique,
        "unique_count": len(ordered_unique),
        "mode": "count" if is_count_query else "list",
    }
    get_client().update_current_span(
        input={"question": question, "n_candidates": len(candidates)},
        output=result,
    )
    return result


@observe(name="adaptive_router_pipeline")
def run_adaptive_router_pipeline(question: str, question_index: int, ground_truth: list[str],
                                 bm25: BM25Okapi, fact_indices: list[int], fact_texts: list[str],
                                 client: OpenAI,
                                 dataset_name: str = "factconsolidation_mh_262k",
                                 competency: str = "Conflict_Resolution") -> dict[str, Any]:
    route = rl(question).name

    top_k = min(TOP_K, len(fact_texts))
    if route == "historical":
        top_k = min(25, len(fact_texts))
    elif route == "aggregation":
        top_k = min(40, len(fact_texts))

    get_client().update_current_span(
        name=f"adaptive_router_{dataset_name}_q{question_index + 1}",
        metadata={
            "system": "Semantic routing + adaptive operator layer",
            "model": MODEL,
            "question_index": question_index,
            "ground_truth": ground_truth,
            "experiment": "adaptive_router_pipeline",
            "dataset": dataset_name,
            "competency": competency,
            "route": str(route),
            "route_method": "semantic_router",
            "route_top_k": top_k,
        },
        input={"question": question},
    )

    retrieved = bm25_retrieve(bm25, question, fact_indices, fact_texts, top_k)
    candidates = _extract_candidates(client, question, retrieved)

    chosen = None
    aggregate = None
    if route == "current_value":
        chosen = _freshness_pick(candidates)
        answer = chosen["answer_entity"] if chosen else "(no answer)"
    elif route == "historical":
        chosen = _historical_pick(candidates, question)
        answer = chosen["answer_entity"] if chosen else "(no answer)"
    elif route == "boolean":
        chosen = _freshness_pick(candidates)
        answer = str(_boolean_gate(question, chosen))
    elif route == "aggregation":
        aggregate = _aggregate_answers(candidates, question)
        answer = aggregate["answer"]
    else:
        chosen = _freshness_pick(candidates)
        answer = chosen["answer_entity"] if chosen else "(no answer)"

    eval_result = evaluate_answer(answer, ground_truth)
    get_client().update_current_span(
        output={
            "answer": answer,
            "is_correct": eval_result["is_correct_subem"],
            "route": str(route),
            "n_candidates": len(candidates),
            "chosen_serial": chosen.get("serial") if chosen else None,
            "aggregate_unique_count": aggregate["unique_count"] if aggregate else None,
        },
    )
    return {
        "answer": answer,
        "is_correct": eval_result["is_correct_subem"],
        "route": str(route),
        "route_scores": None,
        "route_components": None,
        "route_method": "semantic_router",
        "retrieved": retrieved,
        "candidates": candidates,
        "chosen": chosen,
        "aggregate": aggregate,
        "n_candidates": len(candidates) if candidates is not None else 0,
        "chosen_serial": chosen.get("serial") if chosen else None,
    }


@observe(name="hop")
def _execute_hop(hop_id: int, raw_query: str, prior_results: list[dict[str, Any]],
                 bm25: BM25Okapi, fact_indices: list[int], fact_texts: list[str],
                 client: OpenAI) -> dict[str, Any]:
    get_client().update_current_span(input={"hop_id": hop_id, "raw_query": raw_query})
    resolved_query = _substitute_template(raw_query, prior_results)
    retrieved = bm25_retrieve(bm25, resolved_query, fact_indices, fact_texts, TOP_K)
    candidates = _extract_candidates(client, resolved_query, retrieved)
    chosen = _freshness_pick(candidates)
    answer = chosen["answer_entity"] if chosen else None
    result = {
        "hop_id": hop_id, "raw_query": raw_query, "resolved_query": resolved_query,
        "retrieved": retrieved, "candidates": candidates,
        "chosen": chosen, "answer": answer,
    }
    get_client().update_current_span(output={"answer": answer, "n_candidates": len(candidates)})
    return result


def _can_execute_hop(raw_query: str, prior_results: list[dict[str, Any]]) -> tuple[bool, str | None]:
    """Check whether all {hop_N_answer} placeholders in raw_query have non-None answers."""
    needed_ids = re.findall(r"\{hop_(\d+)_answer\}", raw_query)
    for nid in needed_ids:
        n = int(nid)
        prior = next((p for p in prior_results if p.get("hop_id") == n), None)
        if prior is None or prior.get("answer") is None:
            return False, nid
    return True, None


@observe(name="car_v2")
def run_car_v2(question: str, question_index: int, ground_truth: list[str],
               bm25: BM25Okapi, fact_indices: list[int], fact_texts: list[str],
               client: OpenAI,
               dataset_name: str = "factconsolidation_mh_262k",
               competency: str = "Conflict_Resolution") -> dict[str, Any]:
    get_client().update_current_span(
        name=f"car_v2_{dataset_name}_q{question_index + 1}",
        metadata={
            "system": "CAR v2 (deterministic freshness, minimal-guided decompose, skip-on-failure)",
            "model": MODEL, "temperature": 0.0, "top_k_per_hop": TOP_K,
            "question_index": question_index, "ground_truth": ground_truth,
            "experiment": "car_v2",
            "dataset": dataset_name,
            "competency": competency,
        },
        input={"question": question},
    )

    hops = _decompose(client, question)
    trace: list[dict[str, Any]] = []
    for hop in hops:
        # Skip-on-failure: if this hop needs a {hop_N_answer} that is None, skip it.
        can_run, missing = _can_execute_hop(hop["query"], trace)
        if not can_run:
            trace.append({
                "hop_id": hop["id"], "raw_query": hop["query"], "resolved_query": None,
                "skipped": True, "skip_reason": f"depends on hop_{missing}_answer which is None",
                "answer": None, "retrieved": [], "candidates": [], "chosen": None,
            })
            continue
        result = _execute_hop(hop["id"], hop["query"], trace, bm25, fact_indices, fact_texts, client)
        result["skipped"] = False
        trace.append(result)

    # Final answer = last hop's answer that is non-None.
    final_answer = next(
        (t["answer"] for t in reversed(trace) if t.get("answer")),
        ""
    ) or ""
    eval_result = evaluate_answer(final_answer or "(no answer)", ground_truth)
    n_executed = sum(1 for t in trace if not t.get("skipped"))
    n_skipped = sum(1 for t in trace if t.get("skipped"))
    get_client().update_current_span(
        output={
            "final_answer": final_answer or "(no answer)",
            "is_correct": eval_result["is_correct_subem"],
            "n_hops_planned": len(hops),
            "n_hops_executed": n_executed,
            "n_hops_skipped": n_skipped,
        },
    )
    return {
        "final_answer": final_answer or "(no answer)",
        "is_correct": eval_result["is_correct_subem"],
        "decomposition": hops, "trace": trace,
        "n_hops_planned": len(hops),
        "n_hops_executed": n_executed,
        "n_hops_skipped": n_skipped,
    }
