"""Prepare offline evidence and manifests for the professor-feedback campaign.

This command loads only the pinned local MemoryAgentBench cache and repository
files. It never constructs an OpenAI client. A fresh ``--run-id`` directory is
required so preparation cannot overwrite a prior campaign.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
INTENTS = ("historical", "aggregation", "boolean")
SERIAL_RE = re.compile(r"(?m)^[ \t]*(\d+)\.\s")
LEGACY_SERIAL_RE = re.compile(r"(\d+)\.\s")
PREDICATE_RE = re.compile(
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
FALLBACK_RE = re.compile(
    r"^(.+?)\s+(was|is|has|had)\s+(.+?)\s+(?:of|in|by|from|at|to)\s+(.+)$",
    re.IGNORECASE,
)
QUESTION_PROPERTY_TYPES = {
    "country": "country", "countries": "country",
    "genre": "genre", "genres": "genre",
    "location": "location", "locations": "location",
    "language": "language", "languages": "language",
    "religion": "religion", "religions": "religion",
    "nationality": "nationality", "nationalities": "nationality",
    "ethnicity": "ethnicity", "ethnicities": "ethnicity",
    "occupation": "occupation", "occupations": "occupation",
    "creator": "creator", "creators": "creator",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(raw.encode("utf-8"))


def normalize(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def infer_property_type(text: str) -> str:
    lowered = text.casefold()
    if "country" in lowered:
        return "country"
    if "genre" in lowered:
        return "genre"
    if "located" in lowered or "capital" in lowered:
        return "location"
    if "language" in lowered:
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


def question_property_type(question: str) -> str:
    lowered = question.casefold()
    for noun, kind in QUESTION_PROPERTY_TYPES.items():
        if re.search(rf"\b{re.escape(noun)}\b", lowered):
            return kind
    return "value"


def parse_facts(context: str, *, line_start: bool = True) -> list[dict[str, Any]]:
    pattern = SERIAL_RE if line_start else LEGACY_SERIAL_RE
    matches = list(pattern.finditer(context))
    facts: list[dict[str, Any]] = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(context)
        facts.append({
            "serial": int(match.group(1)),
            "text": context[match.end():end].strip().rstrip("."),
            "context_order": i,
            "offset": match.start(),
        })
    return facts


def parse_generated_fact(fact: dict[str, Any]) -> dict[str, Any] | None:
    text = fact["text"]
    match = PREDICATE_RE.match(text)
    parse_kind = "explicit"
    if match:
        entity, value = match.group(1).strip(), match.group(2).strip()
    else:
        match = FALLBACK_RE.match(text)
        if not match:
            return None
        entity, value = match.group(1).strip(), match.group(4).strip()
        parse_kind = "fallback"
    return {
        "entity": entity,
        "serial": fact["serial"],
        "context_order": fact["context_order"],
        "fact_text": text,
        "answer_entity": value,
        "property_type": infer_property_type(text),
        "parse_kind": parse_kind,
    }


def load_false_value_catalog() -> list[str]:
    """Read literal false-value arrays without importing/executing generators."""
    values: set[str] = set()
    for path in (
        ROOT / "scripts/data_generation/generate_synthetic_benchmark.py",
        ROOT / "scripts/data_generation/phase2_multi_batch.py",
        ROOT / "scripts/data_generation/phase2_batch2.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if not any(name.casefold() in {"false_values", "false_pool"} for name in names):
                continue
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict):
                for entries in value.values():
                    if isinstance(entries, list):
                        values.update(item for item in entries if isinstance(item, str))
    return sorted(values, key=lambda item: (-len(normalize(item)), normalize(item)))


def find_boolean_target(question: str, entity: str, value_catalog: list[str]) -> tuple[str | None, str]:
    qnorm = f" {normalize(question)} "
    enorm = normalize(entity)
    matches: list[str] = []
    for value in value_catalog:
        vnorm = normalize(value)
        if not vnorm or vnorm == enorm or len(vnorm) < 3:
            continue
        if f" {vnorm} " in qnorm:
            matches.append(value)
    if not matches:
        return None, "not_found"
    max_len = max(len(normalize(item)) for item in matches)
    longest = sorted({item for item in matches if len(normalize(item)) == max_len})
    if len(longest) != 1:
        return None, "multiple_equal_length_matches"
    return longest[0], "matched_catalog"


def support_for_item(
    item: dict[str, Any], entity_facts: dict[str, list[dict[str, Any]]],
    value_catalog: list[str],
) -> dict[str, Any]:
    entity = item["entity"]
    question = item["question_text"]
    intent = item["intent"]
    records = entity_facts.get(entity, [])
    qproperty = question_property_type(question)
    selected = records if qproperty == "value" else [r for r in records if r["property_type"] == qproperty]
    all_values = {normalize(r["answer_entity"]): r["answer_entity"] for r in records if normalize(r["answer_entity"])}
    selected_values = {normalize(r["answer_entity"]): r["answer_entity"] for r in selected if normalize(r["answer_entity"])}
    generated_gt = item["ground_truth_answer"]
    issue_codes: list[str] = []
    status = "verified"
    support: list[dict[str, Any]] = []
    generated_semantics = "all parsed subject values" if qproperty == "value" else f"values inferred as {qproperty}"

    if not records:
        status = "unresolved"
        issue_codes.append("entity_not_found_in_source_parse")
    elif intent == "historical":
        if not selected:
            status = "unresolved"
            issue_codes.append("no_source_facts_for_question_property")
        else:
            chosen = min(selected, key=lambda rec: (rec["serial"], rec["context_order"]))
            support = [chosen]
            if normalize(chosen["answer_entity"]) != normalize(generated_gt):
                status = "invalid_gold_candidate"
                issue_codes.append("historical_gold_differs_from_line_start_source")
            if len(records) != len({r["serial"] for r in records}):
                issue_codes.append("duplicate_serial_inside_entity_support")
    elif intent == "aggregation":
        raw_distinct = {r["answer_entity"] for r in selected}
        norm_distinct = set(selected_values)
        generated_count = len({r["answer_entity"] for r in records})
        intended_count = len(norm_distinct)
        if qproperty != "value" and len({r["property_type"] for r in records}) > 1:
            if generated_count != len({r["answer_entity"] for r in selected}):
                status = "ambiguous_gold_candidate"
                issue_codes.append("generated_count_spans_multiple_properties")
        if not selected:
            status = "unresolved"
            issue_codes.append("no_source_facts_for_question_property")
        try:
            expected_count = int(generated_gt)
        except (TypeError, ValueError):
            expected_count = None
            status = "invalid_gold_candidate"
            issue_codes.append("aggregation_gold_not_integer")
        if expected_count is not None and expected_count != len(raw_distinct):
            status = "invalid_gold_candidate"
            issue_codes.append("aggregation_gold_differs_from_generated_scope")
        if len(raw_distinct) != len(norm_distinct):
            issue_codes.append("normalization_collision_changes_count")
        representatives: dict[str, dict[str, Any]] = {}
        for rec in sorted(selected, key=lambda r: (r["serial"], r["context_order"])):
            norm = normalize(rec["answer_entity"])
            if norm and norm not in representatives:
                representatives[norm] = rec
        support = list(representatives.values())
        generated_semantics = (
            "all parsed subject values, regardless of relation" if qproperty == "value"
            else f"generator count across all relations; question names {qproperty}"
        )
    elif intent == "boolean":
        target_catalog = sorted(
            {*value_catalog, *(r["answer_entity"] for r in records)},
            key=lambda value: (-len(normalize(value)), normalize(value)),
        )
        target, target_status = find_boolean_target(question, entity, target_catalog)
        if not target:
            status = "ambiguous_gold_candidate"
            issue_codes.append(f"boolean_target_{target_status}")
        else:
            matching = [r for r in selected if normalize(r["answer_entity"]) == normalize(target)]
            source_truth = bool(matching)
            labeled_truth = str(generated_gt).strip().casefold() == "true"
            if source_truth != labeled_truth:
                status = "invalid_gold_candidate"
                issue_codes.append("boolean_label_disagrees_with_source_support")
            support = matching
            if not source_truth:
                issue_codes.append("negative_query_absence_needs_complete_history")
    else:
        status = "unresolved"
        issue_codes.append("unknown_intent")

    return {
        "entity": entity,
        "intent": intent,
        "question": question,
        "ground_truth_answer": generated_gt,
        "question_property_type": qproperty,
        "source_property_types": sorted({r["property_type"] for r in records}),
        "generation_semantics": generated_semantics,
        "source_fact_count": len(records),
        "all_distinct_value_count_raw": len({r["answer_entity"] for r in records}),
        "all_distinct_value_count_normalized": len(all_values),
        "property_distinct_value_count_raw": len({r["answer_entity"] for r in selected}),
        "property_distinct_value_count_normalized": len(selected_values),
        "boolean_target": target if intent == "boolean" else None,
        "support_serials": [r["serial"] for r in support],
        "support_facts": [
            {"serial": r["serial"], "fact_text": r["fact_text"],
             "answer_entity": r["answer_entity"], "property_type": r["property_type"]}
            for r in support
        ],
        "status": status,
        "issue_codes": sorted(set(issue_codes)),
        "evaluation_only_metadata": True,
    }


def parse_saved_result_error(row: dict[str, Any], row_number: int) -> dict[str, Any]:
    intent, route = row["intent"], row["route"]
    answer, gold = str(row["answer"]), str(row["ground_truth"])
    correct = bool(row["qa_correct"])
    if answer.startswith("<error:"):
        primary = "execution_error"
    elif route != intent:
        primary = "misroute" if not correct else "correct"
    elif correct:
        primary = "correct"
    elif answer == "(no answer)":
        primary = "no_answer"
    elif intent == "aggregation" and answer.strip().isdigit() and gold.strip().isdigit():
        primary = "aggregation_undercount" if int(answer) < int(gold) else "aggregation_overcount"
    elif intent == "boolean":
        primary = "boolean_mismatch"
    elif intent == "historical":
        primary = "historical_answer_mismatch"
    else:
        primary = "other_answer_mismatch"
    secondary = []
    if route != intent:
        secondary.append("misrouted")
    if answer == "(no answer)":
        secondary.append("no_answer")
    return {
        "row_id": row_number,
        "entity": row["entity"],
        "intent": intent,
        "route": route,
        "question": row["question"],
        "answer": answer,
        "ground_truth": gold,
        "legacy_score": bool(row["qa_correct"]),
        "legacy_route_correct": bool(row["route_correct"]),
        "primary_category": primary,
        "secondary_flags": ";".join(secondary),
        "observed_symptom": primary,
        "evidence": "saved result row; this file has no candidates or retrieval trace",
        "inference_limit": "cannot attribute answer mismatch to retrieval, extraction, or operator from saved output alone",
    }


def package_versions() -> dict[str, str]:
    packages = (
        "datasets", "huggingface_hub", "langfuse", "numpy", "openai",
        "rank-bm25", "semantic-router", "sentence-transformers", "torch", "transformers",
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def hash_inputs(submitted_docx: Path | None) -> dict[str, Any]:
    tracked_inputs: dict[str, str] = {}
    for relative in ("data/synthetic_benchmark.json", "results/poc_results"):
        path = ROOT / relative
        paths = [path] if path.is_file() else sorted(path.glob("*.json"))
        for file_path in paths:
            tracked_inputs[str(file_path.relative_to(ROOT))] = file_hash(file_path)
    for relative in ("logs/rerun_2026-10-03.log",):
        path = ROOT / relative
        if path.exists():
            tracked_inputs[relative] = file_hash(path)
    return {
        "repo_inputs_sha256": tracked_inputs,
        "submitted_docx_sha256": file_hash(submitted_docx) if submitted_docx and submitted_docx.exists() else None,
    }


def make_sample(benchmark: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    first_entity_order = list(dict.fromkeys(item["entity"] for item in benchmark))
    dev_entities = first_entity_order[:20]
    remaining = first_entity_order[20:]
    rng = random.Random(42)
    heldout_entities = rng.sample(remaining, 80)
    heldout_set = set(heldout_entities)
    available_ood = [entity for entity in remaining if entity not in heldout_set]
    ood_entities = rng.sample(available_ood, 20)
    selected_entities = set(dev_entities) | heldout_set
    selected: list[dict[str, Any]] = []
    for source_index, item in enumerate(benchmark):
        if item["entity"] not in selected_entities:
            continue
        split = "development" if item["entity"] in dev_entities else "held_out"
        row = dict(item)
        row.update({
            "source_index": source_index,
            "question_id": f"syn-{source_index:05d}-{canonical_hash(item)[:12]}",
            "question_sha256": canonical_hash(item),
            "split": split,
            "entity_sample_seed": 42,
        })
        selected.append(row)
    expected = Counter({"historical": 100, "aggregation": 100, "boolean": 100})
    actual = Counter(item["intent"] for item in selected)
    if len(selected) != 300 or actual != expected:
        raise RuntimeError(f"sample construction did not yield 300 balanced rows: {len(selected)}, {actual}")
    entity_count = Counter(item["entity"] for item in selected)
    if len(entity_count) != 100 or any(count != 3 for count in entity_count.values()):
        raise RuntimeError("sample must contain 100 entities with one row per intent")
    return selected, dev_entities, ood_entities


def build_source_support(context: str) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    facts = parse_facts(context, line_start=True)
    legacy_facts = parse_facts(context, line_start=False)
    anchored_offsets = {match.start() for match in SERIAL_RE.finditer(context)}
    legacy_matches = list(LEGACY_SERIAL_RE.finditer(context))
    spurious = [
        {"pseudo_serial": int(match.group(1)), "offset": match.start(),
         "context_snippet": context[max(0, match.start() - 48):match.start() + 80]}
        for match in legacy_matches if match.start() not in anchored_offsets
    ]
    repeated = Counter(fact["serial"] for fact in facts)
    records_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unparsed: list[dict[str, Any]] = []
    for fact in facts:
        parsed = parse_generated_fact(fact)
        if parsed is None:
            unparsed.append({"serial": fact["serial"], "fact_text": fact["text"]})
            continue
        records_by_entity[parsed["entity"]].append(parsed)
    for records in records_by_entity.values():
        records.sort(key=lambda rec: (rec["serial"], rec["context_order"]))
    stats = {
        "line_start_fact_count": len(facts),
        "line_start_unique_serial_count": len(repeated),
        "line_start_duplicate_serial_ids": {str(k): v for k, v in repeated.items() if v > 1},
        "legacy_regex_fact_count": len(legacy_facts),
        "legacy_regex_empty_chunks": sum(not fact["text"] for fact in legacy_facts),
        "legacy_regex_false_boundaries": len(spurious),
        "legacy_regex_false_boundary_examples": spurious,
        "generator_parser_unparsed_line_start_facts": len(unparsed),
        "generator_parser_unparsed_examples": unparsed[:25],
        "entity_groups": len(records_by_entity),
        "facts_parser_note": "Line-start markers are the audited source of fact boundaries; generator-compatible subject/value parsing is retained separately.",
    }
    return facts, records_by_entity, stats


def build_fc_manifest(ds: Any) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for length in ("6k", "32k", "64k", "262k"):
        source = f"factconsolidation_sh_{length}"
        row = next((item for item in ds if item["metadata"].get("source") == source), None)
        if row is None:
            raise RuntimeError(f"pinned dataset is missing {source}")
        questions = row["questions"][:100]
        answers = row["answers"][:100]
        if len(questions) != 100 or len(answers) != 100:
            raise RuntimeError(f"{source} does not have the required first 100 question/answer pairs")
        rows[source] = {
            "split": "Conflict_Resolution",
            "n": 100,
            "context_sha256": sha256_bytes(row["context"].encode("utf-8")),
            "context_chars": len(row["context"]),
            "ordered_question_answer_pairs": [
                {"source_index": i, "question": q, "answer": a,
                 "question_sha256": sha256_bytes(str(q).encode("utf-8")),
                 "answer_sha256": sha256_bytes(json.dumps(a, ensure_ascii=False, sort_keys=True).encode("utf-8"))}
                for i, (q, a) in enumerate(zip(questions, answers))
            ],
        }
    return rows


def query_for_retrieval(question: str, route: str) -> str:
    if route not in {"historical", "aggregation"}:
        return question
    noise = r"\b(how many|count|number of|total|unique|distinct|different|list all|sum|what|was|is|the|initial|earliest|first|previous|original|value|values|recorded|associated|with|for|give|me)\b"
    result = re.sub(noise, "", question, flags=re.IGNORECASE).strip()
    return result or question


def rank_retrieval(scores: Any, limit: int) -> list[int]:
    # Diagnostics use stable descending ties by original source order.
    import numpy as np
    return np.argsort(-np.asarray(scores), kind="stable")[:limit].tolist()


def rank_pipeline(scores: Any, limit: int) -> list[int]:
    """Reproduce the repository pipeline's historical reverse-argsort tie behavior."""
    import numpy as np
    return np.argsort(scores)[::-1][:limit].tolist()


def retrieval_audit(
    sampled: list[dict[str, Any]], supports: list[dict[str, Any]],
    facts: list[dict[str, Any]], output_dir: Path, *, include_dense: bool,
    reuse_cache_from: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from rank_bm25 import BM25Okapi
    from scripts.lib._pipeline import tokenize

    fact_texts = [fact["text"] for fact in facts]
    serial_to_position = {fact["serial"]: i for i, fact in enumerate(facts)}
    bm25_index_start = time.perf_counter()
    bm25 = BM25Okapi([tokenize(text) for text in fact_texts])
    bm25_index_seconds = time.perf_counter() - bm25_index_start
    support_by_index = {row["source_index"]: row for row in supports}
    support_rows = [support_by_index[row["source_index"]] for row in sampled]
    dense_matrix = None
    if include_dense:
        from scripts.lib._pipeline import encoder
        vectors_path = output_dir / "private_cache" / "source_fact_embeddings.npy"
        cache_meta = output_dir / "private_cache" / "source_fact_embeddings.sha256"
        source_key = canonical_hash({
            "model": "all-MiniLM-L6-v2",
            "fact_texts_sha256": sha256_bytes("\n".join(fact_texts).encode("utf-8")),
        })
        import numpy as np
        if vectors_path.exists() and cache_meta.exists() and cache_meta.read_text().strip() == source_key:
            dense_matrix = np.load(vectors_path, mmap_mode="r")
            if dense_matrix.shape[0] != len(facts):
                raise RuntimeError("cached dense vectors do not match the fact corpus")
        elif reuse_cache_from is not None:
            reusable_vectors = reuse_cache_from / "private_cache" / "source_fact_embeddings.npy"
            reusable_meta = reuse_cache_from / "private_cache" / "source_fact_embeddings.sha256"
            if not reusable_vectors.exists() or not reusable_meta.exists() or reusable_meta.read_text().strip() != source_key:
                raise RuntimeError("requested reusable dense cache is missing or does not match the source corpus/model")
            import shutil
            shutil.copy2(reusable_vectors, vectors_path)
            shutil.copy2(reusable_meta, cache_meta)
            dense_matrix = np.load(vectors_path, mmap_mode="r")
        else:
            dense_matrix = encoder.encode(
                fact_texts, batch_size=64, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=True,
            )
            np.save(vectors_path, dense_matrix)
            cache_meta.write_text(source_key + "\n", encoding="utf-8")

    ks = (10, 25, 40, 60, 80)
    per_query: list[dict[str, Any]] = []
    for item, support in zip(sampled, support_rows):
        from scripts.lib._pipeline import native_route
        actual_route = native_route(item["question_text"])
        query = query_for_retrieval(item["question_text"], actual_route)
        relevant_serials = sorted(set(support["support_serials"]))
        relevant_positions = [serial_to_position[s] for s in relevant_serials if s in serial_to_position]
        tokens = tokenize(query)
        retrieval_start = time.perf_counter()
        sparse_scores = bm25.get_scores(tokens)
        sparse_rank = rank_retrieval(sparse_scores, len(facts))
        pipeline_sparse_rank = rank_pipeline(sparse_scores, len(facts))
        bm25_ms = (time.perf_counter() - retrieval_start) * 1000
        dense_rank: list[int] | None = None
        fused_rank: list[int] | None = None
        dense_ms: float | None = None
        rrf_ms: float | None = None
        if dense_matrix is not None:
            dense_start = time.perf_counter()
            q_vec = encoder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
            dense_scores = dense_matrix @ q_vec
            dense_rank = rank_retrieval(dense_scores, len(facts))
            dense_ms = (time.perf_counter() - dense_start) * 1000
            rrf_start = time.perf_counter()
            sparse_ranks = {position: rank for rank, position in enumerate(sparse_rank, start=1)}
            dense_ranks = {position: rank for rank, position in enumerate(dense_rank, start=1)}
            rrf_scores = [
                1.0 / (60 + sparse_ranks[i]) + 1.0 / (60 + dense_ranks[i])
                for i in range(len(facts))
            ]
            fused_rank = rank_retrieval(rrf_scores, len(facts))
            rrf_ms = (time.perf_counter() - rrf_start) * 1000
        eligible = bool(relevant_positions)
        support_values = sorted({normalize(rec["answer_entity"]) for rec in support["support_facts"] if normalize(rec["answer_entity"])})
        records: dict[str, Any] = {
            "source_index": item["source_index"],
            "question_id": item["question_id"],
            "split": item["split"],
            "entity": item["entity"],
            "intent": item["intent"],
            "native_route": actual_route,
            "question": item["question_text"],
            "support_status": support["status"],
            "support_issue_codes": support["issue_codes"],
            "support_serials": relevant_serials,
            "evaluation_only_source_support_values": support_values,
            "retrieval_query": query,
            "retrieval_latency_ms": {"bm25_score_and_rank": bm25_ms, "dense_encode_score_and_rank": dense_ms, "rrf_fusion_and_rank": rrf_ms},
            "retrieval_eligible": eligible,
            "methods": {},
        }
        for method, ranking in (("bm25", sparse_rank), ("pipeline_bm25", pipeline_sparse_rank), ("dense", dense_rank), ("rrf", fused_rank)):
            if ranking is None:
                continue
            method_data = {}
            for k in ks:
                retrieved = {facts[position]["serial"] for position in ranking[:k]}
                hits = sorted(set(relevant_serials) & retrieved)
                covered_values = sorted({
                    normalize(rec["answer_entity"])
                    for rec in support["support_facts"]
                    if normalize(rec["answer_entity"]) and rec["serial"] in retrieved
                })
                method_data[str(k)] = {
                    "relevant_retrieved": len(hits),
                    "relevant_total": len(relevant_serials),
                    "recall": (len(hits) / len(relevant_serials)) if relevant_serials else None,
                    "all_support_retrieved": bool(relevant_serials) and len(hits) == len(relevant_serials),
                    "hit_serials": hits,
                    "missing_serials": sorted(set(relevant_serials) - retrieved),
                    "source_support_value_coverage": len(covered_values) / len(support_values) if support_values else None,
                    "source_support_values_covered": len(covered_values),
                    "source_support_values_total": len(support_values),
                    "retrieved_fact_chars": sum(len(facts[position]["text"]) for position in ranking[:k]),
                }
            records["methods"][method] = method_data
        per_query.append(records)

    summary: dict[str, Any] = {
        "fact_count": len(facts),
        "evaluated_rows": len(per_query),
        "support_eligible_rows": sum(row["retrieval_eligible"] for row in per_query),
        "support_flagged_rows": sum(row["support_status"] != "verified" for row in per_query),
        "index_cost": {
            "bm25_build_seconds_this_run": bm25_index_seconds,
            "source_fact_count": len(facts),
            "dense_embedding_dimensions": int(dense_matrix.shape[1]) if dense_matrix is not None else None,
            "dense_embedding_nbytes": int(dense_matrix.nbytes) if dense_matrix is not None else None,
            "dense_embedding_cache_hit_or_reused": bool(include_dense and reuse_cache_from is not None),
            "dense_embedding_cache_path": str(output_dir / "private_cache" / "source_fact_embeddings.npy") if dense_matrix is not None else None,
            "exact_llm_token_count": None,
            "token_count_note": "GPT-4o-mini o200k_base tokenizer asset is absent from the local tiktoken cache; token counts were not guessed and no network download was attempted by the audit.",
        },
        "k_values": list(ks),
        "rrf_definition": "equal-weight reciprocal rank fusion, 1/(60+BM25_rank) + 1/(60+dense_rank); all comparison ranks use stable descending score with original source order as tie-break",
        "historical_pipeline_tie_note": "pipeline_bm25 separately reproduces the repository's original numpy.argsort(scores)[::-1] ordering; bm25 is the predeclared stable-tie comparison",
        "metrics": {},
    }
    for method in sorted({name for row in per_query for name in row["methods"]}):
        summary["metrics"][method] = {"all_source_supported": {}, "verified_gold_only": {}}
        for k in ks:
            for scope, predicate in (
                ("all_source_supported", lambda row: row["retrieval_eligible"]),
                ("verified_gold_only", lambda row: row["retrieval_eligible"] and row["support_status"] == "verified"),
            ):
                eligible = [row["methods"][method][str(k)] for row in per_query if predicate(row)]
                total_relevant = sum(result["relevant_total"] for result in eligible)
                total_hits = sum(result["relevant_retrieved"] for result in eligible)
                selected_rows = [row for row in per_query if predicate(row)]
                total_values = sum(len(row["evaluation_only_source_support_values"]) for row in selected_rows)
                covered_values = sum(row["methods"][method][str(k)]["source_support_values_covered"] for row in selected_rows)
                latencies = [row["retrieval_latency_ms"].get({"bm25": "bm25_score_and_rank", "pipeline_bm25": "bm25_score_and_rank", "dense": "dense_encode_score_and_rank", "rrf": "rrf_fusion_and_rank"}.get(method, "bm25_score_and_rank")) for row in per_query if predicate(row)]
                latencies = [value for value in latencies if value is not None]
                summary["metrics"][method][scope][str(k)] = {
                    "micro_recall": total_hits / total_relevant if total_relevant else None,
                    "support_facts_retrieved": total_hits,
                    "support_facts_total": total_relevant,
                    "questions_with_all_support_retrieved": sum(result["all_support_retrieved"] for result in eligible),
                    "eligible_questions": len(eligible),
                    "source_support_value_coverage": covered_values / total_values if total_values else None,
                    "source_support_values_covered": covered_values,
                    "source_support_values_total": total_values,
                    "mean_retrieved_fact_chars": sum(result["retrieved_fact_chars"] for result in eligible) / len(eligible) if eligible else None,
                    "mean_retrieval_latency_ms": sum(latencies) / len(latencies) if latencies else None,
                    "p95_retrieval_latency_ms": sorted(latencies)[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else None,
                }
    adaptive = {"all_source_supported": {}, "verified_gold_only": {}}
    for scope, predicate in (
        ("all_source_supported", lambda row: row["retrieval_eligible"]),
        ("verified_gold_only", lambda row: row["retrieval_eligible"] and row["support_status"] == "verified"),
    ):
        selected = []
        for row in per_query:
            if not predicate(row):
                continue
            k = 25 if row["native_route"] == "historical" else 40 if row["native_route"] == "aggregation" else 10
            selected.append(row["methods"]["pipeline_bm25"][str(k)])
        total_relevant = sum(result["relevant_total"] for result in selected)
        total_hits = sum(result["relevant_retrieved"] for result in selected)
        adaptive[scope] = {
            "micro_recall": total_hits / total_relevant if total_relevant else None,
            "support_facts_retrieved": total_hits,
            "support_facts_total": total_relevant,
            "questions_with_all_support_retrieved": sum(result["all_support_retrieved"] for result in selected),
            "eligible_questions": len(selected),
            "policy": "native_route then BM25 top-25 historical, top-40 aggregation, otherwise top-10",
        }
    summary["metrics"]["legacy_adaptive_k_pipeline_bm25"] = adaptive

    groups = {
        "development_first_60": lambda row: row["split"] == "development",
        "development_historical": lambda row, intent="historical": row["split"] == "development" and row["intent"] == intent,
        "development_aggregation": lambda row, intent="aggregation": row["split"] == "development" and row["intent"] == intent,
        "development_boolean": lambda row, intent="boolean": row["split"] == "development" and row["intent"] == intent,
        "pooled_retrieval_rows": lambda row: True,
    }
    summary["metrics_by_group"] = {}
    for group_name, group_predicate in groups.items():
        selected_group = [row for row in per_query if group_predicate(row)]
        summary["metrics_by_group"][group_name] = {}
        for method in sorted({name for row in per_query for name in row["methods"]}):
            summary["metrics_by_group"][group_name][method] = {}
            for k in ks:
                selected = [
                    row for row in selected_group
                    if row["retrieval_eligible"] and row["support_status"] == "verified"
                ]
                results = [row["methods"][method][str(k)] for row in selected]
                support_total = sum(result["relevant_total"] for result in results)
                support_hit = sum(result["relevant_retrieved"] for result in results)
                values_total = sum(len(row["evaluation_only_source_support_values"]) for row in selected)
                values_hit = sum(row["methods"][method][str(k)]["source_support_values_covered"] for row in selected)
                summary["metrics_by_group"][group_name][method][str(k)] = {
                    "micro_recall": support_hit / support_total if support_total else None,
                    "support_facts_retrieved": support_hit,
                    "support_facts_total": support_total,
                    "questions_with_all_support_retrieved": sum(result["all_support_retrieved"] for result in results),
                    "eligible_questions": len(results),
                    "source_support_value_coverage": values_hit / values_total if values_total else None,
                    "source_support_values_covered": values_hit,
                    "source_support_values_total": values_total,
                    "mean_retrieved_fact_chars": sum(result["retrieved_fact_chars"] for result in results) / len(results) if results else None,
                }
    return per_query, summary


def render_retrieval_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Retrieval recall: original 60-question development set",
        "",
        "Metrics use source-supported serials reconstructed from the pinned local context. The verified-gold subset excludes invalid/ambiguous support; no held-out question was used in this retrieval-selection report. Negative boolean questions have no positive support serial and are excluded, so this metric does not prove negative-answer coverage.",
        "",
        "## Overall verified support",
        "",
        "| Method | Eligible | Recall@10 | Recall@25 | Recall@40 | Recall@60 | Recall@80 | Value coverage@80 | Complete support@80 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    group = summary["metrics_by_group"]["development_first_60"]
    for method in ("bm25", "pipeline_bm25", "dense", "rrf"):
        values = group[method]
        n = values["80"]["eligible_questions"]
        recalls = ["n/a" if values[str(k)]["micro_recall"] is None else f"{values[str(k)]['micro_recall']:.1%}" for k in (10, 25, 40, 60, 80)]
        k80 = values["80"]
        value_cov = "n/a" if k80["source_support_value_coverage"] is None else f"{k80['source_support_value_coverage']:.1%}"
        lines.append(f"| {method} | {n} | " + " | ".join(recalls) + f" | {value_cov} | {k80['questions_with_all_support_retrieved']}/{n} |")
    lines += [
        "",
        "## Per-intent verified support",
        "",
        "| Intent | Method | Eligible | Recall@10 | Recall@25 | Recall@40 | Recall@60 | Recall@80 | Value coverage@80 | Complete support@80 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for intent in INTENTS:
        for method in ("bm25", "dense", "rrf"):
            values = summary["metrics_by_group"][f"development_{intent}"][method]
            n = values["80"]["eligible_questions"]
            recalls = ["n/a" if values[str(k)]["micro_recall"] is None else f"{values[str(k)]['micro_recall']:.1%}" for k in (10, 25, 40, 60, 80)]
            k80 = values["80"]
            value_cov = "n/a" if k80["source_support_value_coverage"] is None else f"{k80['source_support_value_coverage']:.1%}"
            lines.append(f"| {intent} | {method} | {n} | " + " | ".join(recalls) + f" | {value_cov} | {k80['questions_with_all_support_retrieved']}/{n} |")
    lines += [
        "",
        "## Retrieval/index costs observed locally",
        "",
        f"- Facts indexed: {summary['index_cost']['source_fact_count']:,}.",
        f"- BM25 index construction: {summary['index_cost']['bm25_build_seconds_this_run']:.3f} seconds for this run.",
        f"- MiniLM embedding matrix: {summary['index_cost']['dense_embedding_dimensions']} dimensions, {summary['index_cost']['dense_embedding_nbytes']:,} bytes on disk; embeddings came from a hash-validated local cache.",
        "- Per-question BM25/dense/RRF latency and retrieved-fact character counts are recorded in `retrieval_recall_development_60.json`; they exclude LLM time.",
        f"- Exact API tokenizer count unavailable: {summary['index_cost']['token_count_note']}",
        "- `pipeline_bm25` reproduces the existing NumPy reverse-argsort tie order; `bm25` uses stable descending score with source order as the tie break. The retrieval configuration has not been frozen pending review of flagged gold rows and the token budget.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="New output directory name under results/prof_feedback and logs/prof_feedback.")
    parser.add_argument("--submitted-docx", type=Path, help="Optional path to the submitted feedback DOCX.")
    parser.add_argument("--skip-dense", action="store_true", help="Skip local dense retrieval preparation; no API requests are made either way.")
    parser.add_argument("--reuse-cache-from", type=Path, help="Reuse an existing matching local MiniLM fact-vector cache from a prior audit run.")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,80}", args.run_id):
        parser.error("--run-id must be a simple unique directory name")
    output_dir = ROOT / "results" / "prof_feedback" / args.run_id
    log_dir = ROOT / "logs" / "prof_feedback" / args.run_id
    if output_dir.exists() or log_dir.exists():
        parser.error(f"run-id already exists; refusing to overwrite: {args.run_id}")
    if git_value("branch", "--show-current") != "revision/prof-feedback":
        parser.error("run preparation must be done on revision/prof-feedback")

    os.environ.setdefault("ALETHEIA_DISABLE_TRACING", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from datasets import load_dataset
    from scripts.lib.config import DATASET_REVISION

    benchmark_path = ROOT / "data/synthetic_benchmark.json"
    saved_results_path = ROOT / "results/poc_results/synthetic_benchmark_results.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    saved_results = json.loads(saved_results_path.read_text(encoding="utf-8"))
    if len(benchmark) != 13_425 or len(saved_results) != 60:
        raise RuntimeError("benchmark/result row counts differ from the reviewed paper inputs")
    first60 = benchmark[:60]
    for i, (question, result) in enumerate(zip(first60, saved_results), start=1):
        left = (question["entity"], question["intent"], question["question_text"], question["ground_truth_answer"])
        right = (result["entity"], result["intent"], result["question"], result["ground_truth"])
        if left != right:
            raise RuntimeError(f"saved results differ from benchmark row {i}")
    if Counter(item["intent"] for item in first60) != {"historical": 20, "aggregation": 20, "boolean": 20}:
        raise RuntimeError("first 60 intent grouping does not match the paper")

    ds = load_dataset("ai-hyz/MemoryAgentBench", split="Conflict_Resolution", revision=DATASET_REVISION)
    fc_row = next((row for row in ds if row["metadata"].get("source") == "factconsolidation_sh_262k"), None)
    if fc_row is None:
        raise RuntimeError("pinned dataset is missing factconsolidation_sh_262k")
    source_facts, entity_facts, parser_stats = build_source_support(fc_row["context"])
    value_catalog = load_false_value_catalog()
    sampled, dev_entities, ood_entities = make_sample(benchmark)

    # The old synthetic router uses native_route and initializes embeddings from
    # the first ROUTE_UTTERANCES dictionary in _pipeline.py. Import that exact
    # implementation only after offline/tracing guards are active.
    from scripts.lib._pipeline import native_route
    router_rows = []
    for i, item in enumerate(first60):
        current_route = native_route(item["question_text"])
        router_rows.append({
            "row_id": i + 1,
            "entity": item["entity"],
            "intent": item["intent"],
            "question": item["question_text"],
            "native_route": current_route,
            "saved_legacy_route": saved_results[i]["route"],
            "matches_saved_route": current_route == saved_results[i]["route"],
        })

    supports: list[dict[str, Any]] = []
    issue_counts: Counter[str] = Counter()
    for source_index, item in enumerate(benchmark):
        result = support_for_item(item, entity_facts, value_catalog)
        result.update({
            "source_index": source_index,
            "question_id": f"syn-{source_index:05d}-{canonical_hash(item)[:12]}",
            "question_sha256": canonical_hash(item),
        })
        supports.append(result)
        issue_counts.update(result["issue_codes"])

    output_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "private_cache").mkdir()
    (output_dir / ".gitignore").write_text("private_cache/\n", encoding="utf-8")
    (log_dir / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")

    development_sample = [row for row in sampled if row["split"] == "development"]
    retrieval_rows, retrieval_summary = retrieval_audit(
        development_sample, supports, source_facts, output_dir,
        include_dense=not args.skip_dense,
        reuse_cache_from=args.reuse_cache_from,
    )

    input_hashes = hash_inputs(args.submitted_docx)
    packages = package_versions()
    manifest = {
        "run_id": args.run_id,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "branch": git_value("branch", "--show-current"),
        "base_sha": git_value("rev-parse", "HEAD"),
        "source_main_sha": git_value("rev-parse", "origin/main"),
        "python": platform.python_version(),
        "dataset": {"repo": "ai-hyz/MemoryAgentBench", "split": "Conflict_Resolution", "revision": DATASET_REVISION, "loaded_offline": True},
        "models": {"router_encoder": "all-MiniLM-L6-v2", "router_setting": "sentence-transformers/all-MiniLM-L6-v2", "llm_default": "gpt-4o-mini", "gpt4o_ablation_legacy": "gpt-4o"},
        "packages": packages,
        "inputs": input_hashes,
        "benchmark_rows": len(benchmark),
        "saved_legacy_result_rows": len(saved_results),
        "samples": {"seed": 42, "development_entities": len(dev_entities), "heldout_entities": 80, "synthetic_rows": len(sampled), "ood_entities_reserved": len(ood_entities)},
        "retrieval_evaluated_rows": len(development_sample),
        "source_parser_audit": parser_stats,
        "paid_calls_made": 0,
        "manuscript_edited": False,
        "prompt_modified": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "synthetic_300.json").write_text(json.dumps(sampled, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "gold_support.json").write_text(json.dumps(supports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "factconsolidation_manifest.json").write_text(json.dumps(build_fc_manifest(ds), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "router_legacy_60.json").write_text(json.dumps(router_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "retrieval_recall_development_60.json").write_text(json.dumps(retrieval_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "retrieval_recall_summary.json").write_text(json.dumps(retrieval_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "retrieval_recall_development_60.md").write_text(render_retrieval_markdown(retrieval_summary), encoding="utf-8")

    failures = [parse_saved_result_error(row, i) for i, row in enumerate(saved_results, start=1)]
    failure_path = output_dir / "legacy_failure_breakdown.csv"
    with failure_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(failures[0]))
        writer.writeheader()
        writer.writerows(failures)
    failure_md = [
        "# Legacy 60 per-question observed failure breakdown",
        "",
        "Categories describe saved symptoms only. The results file has no retrieved candidates, so the evidence column must not be read as a causal attribution.",
        "",
        "| ID | Entity | Intent | Route | Answer | Gold | QA correct | Primary symptom | Secondary flags | Evidence / inference limit |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in failures:
        cells = [
            str(row["row_id"]), row["entity"], row["intent"], row["route"], row["answer"],
            row["ground_truth"], str(row["legacy_score"]), row["primary_category"],
            row["secondary_flags"] or "—", row["inference_limit"],
        ]
        failure_md.append("| " + " | ".join(str(cell).replace("|", "\\|").replace("\n", " ") for cell in cells) + " |")
    (output_dir / "legacy_failure_breakdown.md").write_text("\n".join(failure_md) + "\n", encoding="utf-8")

    result_hashes = input_hashes["repo_inputs_sha256"]
    inventory = [
        "# Source inventory",
        "",
        f"- Run: `{args.run_id}`",
        f"- Branch/base: `{manifest['branch']}` / `{manifest['base_sha']}`",
        f"- Pinned dataset: `ai-hyz/MemoryAgentBench`, split `Conflict_Resolution`, revision `{DATASET_REVISION}` (loaded from local cache with offline mode).",
        f"- Synthetic input: {len(benchmark):,} questions; {len(dev_entities)} calibration entities plus 80 sampled entities for the 300-row package.",
        f"- Legacy result input: {len(saved_results)} rows; the first 60 question/entity/intent/gold tuples match exactly and are balanced 20/20/20.",
        f"- Submitted DOCX SHA-256: `{input_hashes['submitted_docx_sha256']}`.",
        "- No OpenAI requests were made; no manuscript was edited.",
        "",
        "## Environment packages",
        "",
        *[f"- `{name}=={version}`" for name, version in packages.items()],
        "",
        "## Initial protected input hashes",
        "",
        *[f"- `{name}` — `{digest}`" for name, digest in sorted(result_hashes.items())],
        "",
        "Full manifest and dataset/context hashes are in `manifest.json` and `factconsolidation_manifest.json`.",
    ]
    (output_dir / "source_inventory.md").write_text("\n".join(inventory) + "\n", encoding="utf-8")

    primary_counts = Counter(row["primary_category"] for row in failures)
    unresolved_rows = [s for s in supports if s["status"] != "verified"]
    invalid_by_intent = Counter(s["intent"] for s in unresolved_rows)
    dev_ids = set(dev_entities)
    dev_issues = [s for s in unresolved_rows if s["entity"] in dev_ids]
    multi_property = sum(len(s["source_property_types"]) > 1 for s in supports[1::3])
    dev_metrics = retrieval_summary["metrics_by_group"]["development_first_60"]
    retrieval_lines = [
        "| Method | Recall@10 | Recall@25 | Recall@40 | Recall@60 | Recall@80 | Value coverage@K80 | Full support@K80 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in ("bm25", "pipeline_bm25", "dense", "rrf"):
        values = dev_metrics[method]
        fmt = lambda metric, key: "n/a" if metric[key] is None else f"{metric[key]:.1%}"
        recall = [fmt(values[str(k)], "micro_recall") for k in (10, 25, 40, 60, 80)]
        k80 = values["80"]
        retrieval_lines.append(
            f"| {method} | " + " | ".join(recall) +
            f" | {fmt(k80, 'source_support_value_coverage')} | "
            f"{k80['questions_with_all_support_retrieved']}/{k80['eligible_questions']} |"
        )
    retrieval_lines += [
        "",
        "The rows above use the first 60 released records only and verified support rows. The summary JSON also reports intent-specific results, all-source-supported flagged rows, missing serials, latency, and retrieved-fact character counts. `pipeline_bm25` reproduces legacy NumPy tie order; `bm25` uses the predeclared stable source-order tie break.",
        "",
        "| Intent | Method | Verified questions | Recall@10 | Recall@25 | Recall@40 | Recall@60 | Recall@80 | Value coverage@80 | Full support@80 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for intent in INTENTS:
        group = retrieval_summary["metrics_by_group"][f"development_{intent}"]
        for method in ("bm25", "dense", "rrf"):
            values = group[method]
            counts = values["80"]["eligible_questions"]
            recalls = [fmt(values[str(k)], "micro_recall") for k in (10, 25, 40, 60, 80)]
            end = values["80"]
            retrieval_lines.append(
                f"| {intent} | {method} | {counts} | " + " | ".join(recalls) +
                f" | {fmt(end, 'source_support_value_coverage')} | "
                f"{end['questions_with_all_support_retrieved']}/{end['eligible_questions']} |"
            )
    retrieval_route_by_index = {row["source_index"]: row["native_route"] for row in retrieval_rows}
    repeated_route_count = sum(
        row["native_route"] == retrieval_route_by_index.get(row["row_id"] - 1)
        for row in router_rows
    )
    route_mismatches = [
        f"{row['row_id']} ({row['intent']}: saved `{row['saved_legacy_route']}`, replay `{row['native_route']}`)"
        for row in router_rows if not row["matches_saved_route"]
    ]
    report = [
        "# Benchmark and legacy-output audit",
        "",
        "This is a source-based offline audit. It does not change the committed benchmark or interpret missing saved candidates as extraction failures.",
        "",
        "## Source parser and support findings",
        "",
        f"- Line-start fact markers: {parser_stats['line_start_fact_count']:,}; duplicate serial IDs after anchoring: {len(parser_stats['line_start_duplicate_serial_ids'])}.",
        f"- The legacy unanchored parser sees {parser_stats['legacy_regex_false_boundaries']} additional non-line-start markers ({parser_stats['legacy_regex_empty_chunks']} empty chunks). It can split strings such as `U2.` inside a fact.",
        f"- Generator-compatible parser entity groups: {parser_stats['entity_groups']:,}; unparsed line-start facts: {parser_stats['generator_parser_unparsed_line_start_facts']:,}.",
        f"- Synthetic support rows flagged for review: {len(unresolved_rows):,} ({dict(invalid_by_intent)} by intent). These labels are audit flags, not automatic exclusions or corrections.",
        f"- Development rows flagged: {len(dev_issues)}. See `gold_support.json` for exact question IDs, evidence, and issue codes.",
        f"- Entities whose three generated questions include more than one coarse property type: {multi_property:,}; this count alone does not establish gold invalidity.",
        f"- Route-path check: the adaptive pipeline calls `native_route`; its embeddings are built from the first `ROUTE_UTTERANCES` dictionary before a later dictionary rebind. Two same-process passes agree on {repeated_route_count}/60 questions, but current local replay agrees with the saved routes on {sum(row['matches_saved_route'] for row in router_rows)}/60. Mismatches: {', '.join(route_mismatches)}. The saved run does not include the router checkpoint hash needed to explain this discrepancy.",
        "- Property-specific aggregation counts are checked against both the generator's entity-wide rule and the natural-language property scope. Generic `value` questions are treated as entity-wide only because that is how the source generator computes them.",
        "- Negative boolean items are marked with a complete-history caveat; absence from retrieved top-K is not proof of falsity.",
        *[
            f"- First-60 review: row {row['source_index'] + 1}, {row['entity']} ({row['intent']}), gold `{row['ground_truth_answer']}`, codes `{', '.join(row['issue_codes'])}`, support serials `{row['support_serials']}`. Question: {row['question']}"
            for row in dev_issues
        ],
        "",
        "## Legacy 60 observed outcomes",
        "",
        f"- Routing: {sum(bool(row['route_correct']) for row in saved_results)}/60; saved QA score: {sum(bool(row['qa_correct']) for row in saved_results)}/60 using the runner's whitespace-normalized SubEM.",
        f"- Primary row categories: `{dict(sorted(primary_counts.items()))}` (categories sum to 60).",
        *[
            f"- `{category}`: {count}; rows {[row['row_id'] for row in failures if row['primary_category'] == category]}"
            for category, count in sorted(primary_counts.items())
        ],
        "- The saved result JSON contains only route, answer, gold, and correctness fields. It has no retrieved serials or candidates, so it cannot measure retrieval recall or extractor recall for those historical runs.",
        "- Replayed source retrieval can establish a retrieval miss under the reconstructed ranking; it cannot establish which candidates the old LLM actually returned.",
        "",
        "## Reconstructed retrieval recall on the original 60 development questions",
        "",
        *retrieval_lines,
        "",
        f"- Source-supported development rows: {retrieval_summary['support_eligible_rows']}; rows with flagged/unverified gold: {retrieval_summary['support_flagged_rows']}. Negative boolean questions have no positive support serial and are excluded from recall.",
        "- See `retrieval_recall_summary.json` for BM25 and dense recall at K=10/25/40/60/80, equal-weight RRF recall, and the reconstructed legacy adaptive-K BM25 policy. `retrieval_recall_development_60.json` contains per-question support serials, value coverage, missing serials, retrieved character counts and latency.",
        "- Recall is computed against source-fact support under the audited parser, not the historical LLM candidate output. Report verified-gold-only and all-source-supported denominators separately; flagged rows are not silently corrected or removed. Held-out rows were not included in this selection comparison.",
        "",
        "## Sample construction",
        "",
        f"- Development: first 60 released records (20 entities × 3 intents), with no reassignment.",
        "- Held-out: 80 additional distinct entities sampled with seed 42, preserving all three rows per entity; 240 rows.",
        f"- OOD entity reserve: 20 additional entities sampled disjointly with the same RNG stream; {len(ood_entities)} entities reserved. The planned rewrites will be linguistic-shift-only over the same source corpus, not a new-domain OOD claim.",
        "- Source indices and canonical row hashes are stored in `synthetic_300.json`.",
        "",
        "## Limits",
        "",
        "- No paid requests were made. BM25, dense retrieval, and the specified RRF fusion are offline diagnostics only; no method is selected as the final paid-run configuration. No answers from the new systems exist, and no accuracy claim is made for the sampled 300 rows.",
        "- Any flagged or unresolved support item must be reviewed before it can enter a paid manifest. Do not relabel, drop, or repair it silently.",
    ]
    (output_dir / "benchmark_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (log_dir / "preparation.log").write_text(
        f"run_id={args.run_id}\nbranch={manifest['branch']}\nbase_sha={manifest['base_sha']}\n"
        f"dataset_revision={DATASET_REVISION}\npaid_calls=0\ntracing=disabled\n"
        f"retrieval_facts={len(source_facts)}\nretrieval_rows={len(retrieval_rows)}\n",
        encoding="utf-8",
    )

    print(f"Prepared {output_dir}")
    print(f"Base SHA: {manifest['base_sha']}")
    print(f"Dataset revision: {DATASET_REVISION} (offline cache)")
    print(f"Question sample: {len(sampled)} rows; 20 dev entities + 80 held-out entities")
    print(f"Support rows flagged for review: {len(unresolved_rows)}; development flagged: {len(dev_issues)}")
    print(f"Legacy false fact boundaries: {parser_stats['legacy_regex_false_boundaries']}")
    print(f"Development retrieval rows: {len(retrieval_rows)}; support eligible: {retrieval_summary['support_eligible_rows']}; dense={not args.skip_dense}")
    print(json.dumps(retrieval_summary["metrics"], indent=2))
    print("OpenAI calls: 0; manuscript edits: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
