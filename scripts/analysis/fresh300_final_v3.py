"""Freeze, execute once, and score the evaluation-only final-v3 fresh-300 run.

Manifest preparation and scoring are offline. API requests are sent only with
``--run`` after a frozen configuration manifest exists. The durable request
ledger is resumable and each call ID is sent at most once.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_ID = "pf-20261005-final300-v3-01"
PHASE_A = ROOT / "results/prof_feedback/pf-20261004-phaseA-05"
PREVIOUS_RUN = ROOT / "results/prof_feedback/pf-20261004-smoke-01"
PREVIOUS_LOG = ROOT / "logs/prof_feedback/pf-20261004-smoke-01"
OUT = ROOT / "results/prof_feedback" / RUN_ID
LOG = ROOT / "logs/prof_feedback" / RUN_ID
PROMPT_PATH = ROOT / "scripts/prompts/evaluation_final_v3.txt"
OPERATOR_PATH = ROOT / "scripts/lib/prof_feedback_final.py"
OPERATOR_BASE_PATH = ROOT / "scripts/lib/prof_feedback.py"
POLICY_PATH = ROOT / "scripts/lib/prof_feedback_final.py"
SCORER_PATH = ROOT / "scripts/lib/evaluation_scorer.py"
RUNNER_PATH = Path(__file__).resolve()
BENCHMARK_PATH = ROOT / "data/synthetic_benchmark.json"
SEED = 20261005
SAMPLE_N = 300
K = 80
CAPS = {"aletheia": 1024, "direct": 512, "cot": 1024}
MODEL = "gpt-4o-mini"
INPUT_USD_PER_M = 0.15
OUTPUT_USD_PER_M = 0.60
HARD_STOP_USD = 1.00
ARROW_PATH = (
    Path.home() / ".cache/huggingface/datasets/ai-hyz___memory_agent_bench/default/0.0.0"
    / "7ea066982b140a19337e17e60d45d4076e042faf/memory_agent_bench-Conflict_Resolution.arrow"
)
CALLS_PATH = OUT / "requests.jsonl"
FROZEN_PATH = OUT / "frozen_manifest.json"
LEDGER_PATH = LOG / "calls.jsonl"
CHECKPOINT_PATH = LOG / "checkpoint.json"

SERIAL_RE = re.compile(r"(\d+)\.\s")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(raw.encode("utf-8"))


def parse_corpus_facts(context: str) -> list[dict[str, Any]]:
    matches = list(SERIAL_RE.finditer(context))
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, match in enumerate(matches):
        serial = int(match.group(1))
        if serial in seen:
            continue
        seen.add(serial)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(context)
        rows.append({"fact_idx": serial, "text": context[match.end():end].strip().rstrip(".")})
    return rows


def load_cached_split() -> list[dict[str, Any]]:
    """Read the locally cached pinned Arrow snapshot without writing HF locks."""
    from scripts.lib.config import DATASET_REVISION
    if not ARROW_PATH.is_file() or DATASET_REVISION not in str(ARROW_PATH):
        raise RuntimeError(f"pinned local Arrow cache is unavailable: {ARROW_PATH}")
    import pyarrow as pa
    import pyarrow.ipc as ipc
    with pa.memory_map(str(ARROW_PATH), "r") as source:
        table = ipc.open_stream(source).read_all()
    return table.to_pylist()


def local_token_count(messages: list[dict[str, str]], schema: dict[str, Any], encoder: Any) -> int:
    total = 3
    for message in messages:
        total += 4 + len(encoder.encode(message["role"])) + len(encoder.encode(message["content"]))
    schema_json = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    return total + len(encoder.encode(schema_json))


def request_for(arm: str, question: str, retrieved: list[dict[str, Any]]) -> dict[str, Any]:
    from scripts.lib.prof_feedback_final import build_final_evaluation_request
    from scripts.lib.prof_feedback import build_control_request
    if arm == "aletheia":
        return build_final_evaluation_request(question, retrieved)
    return build_control_request(arm, question, retrieved)


def prior_used_indices() -> set[int]:
    old_synthetic = json.loads((PHASE_A / "synthetic_300.json").read_text(encoding="utf-8"))
    old_ood = json.loads((PREVIOUS_RUN / "ood_60.json").read_text(encoding="utf-8"))
    previous_calls = [
        json.loads(line) for line in (PREVIOUS_LOG / "full_calls.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    v2_requests = [
        json.loads(line) for line in (ROOT / "results/prof_feedback/pf-20261004-fresh300-v2-01/requests.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indices = {int(row["source_index"]) for row in old_synthetic + old_ood + previous_calls + v2_requests}
    if len(old_synthetic) != 300 or len(old_ood) != 60 or len(indices) != 700:
        raise RuntimeError(f"expected exactly 700 distinct previously used source questions; found {len(indices)}")
    return indices


def frozen_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from scripts.lib.config import DATASET_REVISION
    import numpy as np
    from rank_bm25 import BM25Okapi
    from scripts.lib.prof_feedback import (
        approved_evaluation_prompt_and_schema, parse_fact_template, property_type_from_fact,
    )
    import tiktoken

    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    source_audit = json.loads((PHASE_A / "gold_support.json").read_text(encoding="utf-8"))
    if len(benchmark) != 13_425 or len(source_audit) != 13_425:
        raise RuntimeError("benchmark or source-audit row count changed")
    support_by_index = {int(row["source_index"]): row for row in source_audit}
    used = prior_used_indices()
    eligible_indices = [index for index in range(len(benchmark)) if index not in used]
    rng = random.Random(SEED)
    selected_indices = rng.sample(eligible_indices, SAMPLE_N)
    selected = []
    for source_index in selected_indices:
        item = dict(benchmark[source_index])
        audit = support_by_index[source_index]
        item.update({
            "source_index": source_index,
            "question_id": f"syn-{source_index:05d}-{audit['question_sha256'][:12]}",
            "source_gold_flagged": audit["status"] != "verified",
            "source_gold_issue_codes": audit["issue_codes"],
            "source_support_status": audit["status"],
        })
        selected.append(item)

    datasets = load_cached_split()
    source_row = next((row for row in datasets if row["metadata"].get("source") == "factconsolidation_sh_262k"), None)
    if source_row is None:
        raise RuntimeError("pinned cached FactConsolidation SH-262K row is missing")
    corpus = parse_corpus_facts(source_row["context"])
    corpus_texts = [row["text"] for row in corpus]
    bm25 = BM25Okapi([re.findall(r"[A-Za-z0-9]+", text.lower()) for text in corpus_texts])

    # Count locally parseable subject histories. If an entity has unparseable
    # subject facts, mark it incomplete so a negative Boolean cannot be inferred.
    selected_entities = {row["entity"] for row in selected}
    entity_fact_counts: Counter[str] = Counter()
    entity_property_counts: Counter[tuple[str, str]] = Counter()
    entity_raw_subject_counts: Counter[str] = Counter()
    for fact in corpus:
        for entity in selected_entities:
            if fact["text"].startswith(entity + " "):
                entity_raw_subject_counts[entity] += 1
                parsed = parse_fact_template(fact["text"])
                if parsed and parsed[0] == entity:
                    entity_fact_counts[entity] += 1
                    entity_property_counts[(entity, property_type_from_fact(fact["text"]))] += 1
    reliable_entity_counts = {
        entity: entity_fact_counts[entity]
        for entity in selected_entities
        if entity_fact_counts[entity] == entity_raw_subject_counts[entity]
    }

    prompt, schema = approved_evaluation_prompt_and_schema()
    encoder = tiktoken.get_encoding("o200k_base")
    calls: list[dict[str, Any]] = []
    for index, item in enumerate(selected, start=1):
        question = item["question_text"]
        tokens = re.findall(r"[A-Za-z0-9]+", question.lower())
        scores = bm25.get_scores(tokens)
        positions = np.argsort(scores)[::-1][:K]
        retrieved = [{"fact_idx": corpus[pos]["fact_idx"], "text": corpus[pos]["text"]} for pos in positions]
        for arm in ("aletheia", "direct", "cot"):
            request = request_for(arm, question, retrieved)
            row = {
                "call_id": f"fresh300-{index:03d}-{arm}",
                "arm": arm,
                "dataset": "fresh300-v2",
                "source_index": item["source_index"],
                "question_id": item["question_id"],
                "entity": item["entity"],
                "intent": item["intent"],
                "question": question,
                "ground_truth_answer": item["ground_truth_answer"],
                "source_gold_flagged": item["source_gold_flagged"],
                "source_gold_issue_codes": item["source_gold_issue_codes"],
                "source_support_status": item["source_support_status"],
                "k": K,
                "model": MODEL,
                "messages": request["messages"],
                "response_format": request.get("response_format"),
                "temperature": 0.0,
                "max_completion_tokens": CAPS[arm],
                "retrieved_items": retrieved,
                "local_input_tokens": local_token_count(
                    request["messages"], request.get("response_format", {}), encoder,
                ),
            }
            calls.append(row)

    if len(calls) != 900 or Counter(row["arm"] for row in calls) != {"aletheia": 300, "direct": 300, "cot": 300}:
        raise RuntimeError("fresh campaign must contain 300 requests per arm")
    if [row["source_index"] for row in calls if row["arm"] == "aletheia"] != selected_indices:
        raise RuntimeError("arm question order diverged")
    context_groups = defaultdict(set)
    for row in calls:
        context_groups[row["source_index"]].add(canonical_hash(row["retrieved_items"]))
    if any(len(values) != 1 for values in context_groups.values()):
        raise RuntimeError("Direct/CoT do not share Aletheia's exact K=80 context")

    stats = {
        "dataset_revision": DATASET_REVISION,
        "dataset_arrow_sha256": sha256_file(ARROW_PATH),
        "benchmark_sha256": sha256_file(BENCHMARK_PATH),
        "sample_seed": SEED,
        "sample_method": "random.Random(20261005).sample(eligible_source_indices, 300)",
        "excluded_prior_question_count": len(used),
        "excluded_prior_source_indices_sha256": canonical_hash(sorted(used)),
        "eligible_question_count": len(eligible_indices),
        "selected_source_indices": selected_indices,
        "selected_source_indices_sha256": canonical_hash(selected_indices),
        "selected_intent_counts": dict(Counter(row["intent"] for row in selected)),
        "selected_flagged_count": sum(row["source_gold_flagged"] for row in selected),
        "entity_count_map": dict(reliable_entity_counts),
        "entity_property_count_map": {f"{entity}\t{prop}": count for (entity, prop), count in entity_property_counts.items()},
        "unreliable_entity_count_names": sorted(selected_entities - set(reliable_entity_counts)),
        "local_corpus_fact_count": len(corpus),
    }
    # Counts are included to make the post-run operator replay independent of
    # any data-generation step. They are covered by the frozen manifest hash.
    return calls, selected, corpus, stats


def freeze() -> dict[str, Any]:
    if OUT.exists() or LOG.exists():
        raise RuntimeError(f"refusing to overwrite existing fresh-run artifacts: {RUN_ID}")
    calls, selected, _corpus, stats = frozen_inputs()
    from scripts.lib.config import DATASET_REVISION
    raw_calls = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in calls).encode("utf-8")
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    (OUT / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
    CALLS_PATH.write_bytes(raw_calls)
    from scripts.lib.prof_feedback_final import FINAL_PROMPT_PATH, TEMPORAL_CUE_MAP, KNOWN_RELATION_LIST
    from scripts.lib.prof_feedback_final import COUNT_CUES
    frozen = {
        "run_id": RUN_ID,
        "frozen_at_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "model": MODEL,
        "temperature": 0.0,
        "k": K,
        "calls_per_arm": 300,
        "arm_count": 3,
        "completion_caps": CAPS,
        "hard_stop_usd_campaign_cumulative": HARD_STOP_USD,
        "prior_campaign_spend_baseline_usd": read_prior_campaign_spend(),
        "input_price_usd_per_million": INPUT_USD_PER_M,
        "output_price_usd_per_million": OUTPUT_USD_PER_M,
        "dataset_revision": DATASET_REVISION,
        "prompt_sha256": sha256_file(FINAL_PROMPT_PATH),
        "operator_code_sha256": sha256_file(OPERATOR_PATH),
        "operator_base_code_sha256": sha256_file(OPERATOR_BASE_PATH),
        "scorer_code_sha256": sha256_file(SCORER_PATH),
        "runner_code_sha256": sha256_file(RUNNER_PATH),
        "sample_seed_sha256": sha256_bytes(str(SEED).encode("ascii")),
        "temporal_cue_list_sha256": canonical_hash(TEMPORAL_CUE_MAP),
        "relation_list_sha256": canonical_hash(KNOWN_RELATION_LIST),
        "count_cue_list_sha256": canonical_hash([pattern.pattern for pattern in COUNT_CUES]),
        "request_manifest_sha256": sha256_bytes(raw_calls),
        "sha256_definitions": {
            "prompt": str(FINAL_PROMPT_PATH.relative_to(ROOT)),
            "operator_code": str(OPERATOR_PATH.relative_to(ROOT)),
            "operator_base_code": str(OPERATOR_BASE_PATH.relative_to(ROOT)),
            "scorer_code": str(SCORER_PATH.relative_to(ROOT)),
            "runner_code": str(RUNNER_PATH.relative_to(ROOT)),
            "request_manifest": str(CALLS_PATH.relative_to(ROOT)),
        },
        **stats,
    }
    frozen["frozen_manifest_sha256"] = canonical_hash(frozen)
    FROZEN_PATH.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    printed = {
        key: frozen[key] for key in (
            "run_id", "model", "k", "calls_per_arm", "completion_caps",
            "hard_stop_usd_campaign_cumulative", "prior_campaign_spend_baseline_usd",
            "sample_seed", "sample_method", "excluded_prior_question_count",
            "excluded_prior_source_indices_sha256", "selected_intent_counts",
            "selected_source_indices_sha256", "prompt_sha256", "operator_code_sha256",
            "operator_base_code_sha256", "scorer_code_sha256", "runner_code_sha256",
            "sample_seed_sha256", "temporal_cue_list_sha256", "relation_list_sha256",
            "count_cue_list_sha256",
            "request_manifest_sha256",
            "frozen_manifest_sha256",
        )
    }
    print(json.dumps(printed, ensure_ascii=False, indent=2))
    return frozen


def read_prior_campaign_spend() -> float:
    checkpoint = PREVIOUS_LOG / "full_checkpoint.json"
    if not checkpoint.is_file():
        raise RuntimeError("prior campaign checkpoint is missing; cannot enforce cumulative $1.00 ceiling")
    value = json.loads(checkpoint.read_text(encoding="utf-8"))
    spent = float(value["cumulative_usage_cost_usd"])
    # Include the already-frozen v2 held-out run in the cumulative ceiling.
    v2_summary = json.loads((ROOT / "results/prof_feedback/pf-20261004-fresh300-v2-01/results_summary.json").read_text(encoding="utf-8"))
    spent += float(v2_summary["fresh300_spend_usd"])
    if spent < 0 or spent >= HARD_STOP_USD:
        raise RuntimeError(f"prior campaign spend leaves no room under $1.00: ${spent:.6f}")
    return spent


def verify_frozen() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    if frozen.get("sample_seed") != SEED or frozen.get("sample_seed_sha256") != sha256_bytes(str(SEED).encode("ascii")):
        raise RuntimeError("frozen sample seed changed; refusing API calls")
    calls_bytes = CALLS_PATH.read_bytes()
    calls_hash = sha256_bytes(calls_bytes)
    if calls_hash != frozen["request_manifest_sha256"]:
        raise RuntimeError("frozen request manifest hash changed; refusing API calls")
    if sha256_file(PROMPT_PATH) != frozen["prompt_sha256"]:
        raise RuntimeError("frozen prompt changed; refusing API calls")
    if sha256_file(OPERATOR_PATH) != frozen["operator_code_sha256"]:
        raise RuntimeError("frozen operator code changed; refusing API calls")
    if sha256_file(OPERATOR_BASE_PATH) != frozen["operator_base_code_sha256"]:
        raise RuntimeError("frozen base operator dependency changed; refusing API calls")
    if sha256_file(SCORER_PATH) != frozen["scorer_code_sha256"]:
        raise RuntimeError("frozen scorer code changed; refusing API calls")
    if sha256_file(RUNNER_PATH) != frozen["runner_code_sha256"]:
        raise RuntimeError("frozen runner code changed; refusing API calls")
    from scripts.lib.prof_feedback_final import TEMPORAL_CUE_MAP, KNOWN_RELATION_LIST, COUNT_CUES
    if canonical_hash(TEMPORAL_CUE_MAP) != frozen["temporal_cue_list_sha256"]:
        raise RuntimeError("frozen temporal cue list changed; refusing API calls")
    if canonical_hash(KNOWN_RELATION_LIST) != frozen["relation_list_sha256"]:
        raise RuntimeError("frozen relation list changed; refusing API calls")
    if canonical_hash([pattern.pattern for pattern in COUNT_CUES]) != frozen["count_cue_list_sha256"]:
        raise RuntimeError("frozen count cue list changed; refusing API calls")
    check = dict(frozen)
    recorded_fingerprint = check.pop("frozen_manifest_sha256")
    if canonical_hash(check) != recorded_fingerprint:
        raise RuntimeError("frozen configuration manifest hash changed")
    calls = [json.loads(line) for line in calls_bytes.decode("utf-8").splitlines() if line.strip()]
    if len(calls) != 900:
        raise RuntimeError("frozen request manifest is not 900 calls")
    return frozen, calls


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def request_body(row: dict[str, Any]) -> dict[str, Any]:
    body = {key: row[key] for key in ("model", "messages", "max_completion_tokens", "temperature")}
    if row["arm"] == "aletheia":
        body["response_format"] = row["response_format"]
    return body


def run() -> dict[str, Any]:
    frozen, calls = verify_frozen()
    env_path = ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY unavailable; no request sent")
    from openai import OpenAI

    prior_spend = float(frozen["prior_campaign_spend_baseline_usd"])
    ledger: dict[str, dict[str, Any]] = {}
    if LEDGER_PATH.exists():
        for line_no, line in enumerate(LEDGER_PATH.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record["call_id"] in ledger:
                raise RuntimeError(f"duplicate ledger call ID at line {line_no}")
            ledger[record["call_id"]] = record
    if any(record["status"] in {"request_error", "usage_missing"} for record in ledger.values()):
        raise RuntimeError("ledger has a call with unknown billed usage; reconcile before resuming")
    fresh_spend = sum(float(row.get("actual_cost_usd", 0) or 0) for row in ledger.values())
    cumulative = prior_spend + fresh_spend
    if cumulative > HARD_STOP_USD:
        raise RuntimeError(f"already above cumulative hard stop: ${cumulative:.6f}")

    LOG.mkdir(parents=True, exist_ok=True)
    client = OpenAI(max_retries=0, timeout=120)
    with LEDGER_PATH.open("a", encoding="utf-8", buffering=1) as stream:
        for row in calls:
            if row["call_id"] in ledger:
                continue
            # The ceiling is based on actual usage from the previous response.
            # Stop before sending if already at the ceiling, then checkpoint
            # immediately after each response that crosses it.
            if cumulative >= HARD_STOP_USD:
                atomic_json(CHECKPOINT_PATH, {
                    "completed_or_terminal_calls": len(ledger), "expected_calls": len(calls),
                    "prior_campaign_spend_usd": round(prior_spend, 10),
                    "fresh300_spend_usd": round(fresh_spend, 10),
                    "cumulative_campaign_spend_usd": round(cumulative, 10),
                    "hard_stop_usd": HARD_STOP_USD, "halt_reason": "cumulative_actual_spend_at_or_above_ceiling",
                    "last_call_id": next(reversed(ledger), None),
                })
                break
            start = time.time()
            try:
                response = client.chat.completions.create(**request_body(row))
            except Exception as exc:
                record = {
                    "call_id": row["call_id"], "arm": row["arm"], "source_index": row["source_index"],
                    "status": "request_error", "error_type": type(exc).__name__,
                    "error": str(exc).replace(os.environ.get("OPENAI_API_KEY", ""), "[redacted]")[:1000],
                    "usage_prompt_tokens": None, "usage_completion_tokens": None,
                    "actual_cost_usd": None, "request_started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush(); os.fsync(stream.fileno())
                ledger[row["call_id"]] = record
                atomic_json(CHECKPOINT_PATH, {
                    "completed_or_terminal_calls": len(ledger), "expected_calls": len(calls),
                    "prior_campaign_spend_usd": round(prior_spend, 10),
                    "fresh300_spend_usd": round(fresh_spend, 10),
                    "cumulative_campaign_spend_usd": round(cumulative, 10),
                    "hard_stop_usd": HARD_STOP_USD, "halt_reason": "request_error_without_usage",
                    "last_call_id": row["call_id"],
                })
                raise RuntimeError(f"halted after one un-retried request error: {row['call_id']}") from exc
            usage = response.usage
            if usage is None or usage.prompt_tokens is None or usage.completion_tokens is None:
                record = {
                    "call_id": row["call_id"], "arm": row["arm"], "source_index": row["source_index"],
                    "status": "usage_missing", "finish_reason": response.choices[0].finish_reason,
                    "request_id": getattr(response, "_request_id", None),
                    "usage_prompt_tokens": None, "usage_completion_tokens": None, "actual_cost_usd": None,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush(); os.fsync(stream.fileno())
                ledger[row["call_id"]] = record
                atomic_json(CHECKPOINT_PATH, {
                    "completed_or_terminal_calls": len(ledger), "expected_calls": len(calls),
                    "prior_campaign_spend_usd": round(prior_spend, 10),
                    "fresh300_spend_usd": round(fresh_spend, 10),
                    "cumulative_campaign_spend_usd": round(cumulative, 10),
                    "hard_stop_usd": HARD_STOP_USD, "halt_reason": "usage_missing",
                    "last_call_id": row["call_id"],
                })
                raise RuntimeError(f"halted: usage missing for {row['call_id']}; no retry")
            amount = usage.prompt_tokens * INPUT_USD_PER_M / 1_000_000 + usage.completion_tokens * OUTPUT_USD_PER_M / 1_000_000
            fresh_spend += amount
            cumulative += amount
            finish = response.choices[0].finish_reason
            record = {
                "call_id": row["call_id"], "arm": row["arm"], "dataset": row["dataset"],
                "source_index": row["source_index"], "question_id": row["question_id"],
                "intent": row["intent"], "ground_truth_answer": row["ground_truth_answer"],
                "source_gold_flagged": row["source_gold_flagged"],
                "source_gold_issue_codes": row["source_gold_issue_codes"],
                "status": "truncated_failure" if finish == "length" else "completed",
                "max_completion_tokens": row["max_completion_tokens"],
                "usage_prompt_tokens": usage.prompt_tokens,
                "usage_completion_tokens": usage.completion_tokens,
                "finish_reason": finish,
                "request_id": getattr(response, "_request_id", None),
                "actual_cost_usd": round(amount, 10),
                "cumulative_fresh300_spend_usd": round(fresh_spend, 10),
                "cumulative_campaign_spend_usd": round(cumulative, 10),
                "elapsed_seconds": round(time.time() - start, 3),
                "response_text": response.choices[0].message.content or "",
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
            ledger[row["call_id"]] = record
            atomic_json(CHECKPOINT_PATH, {
                "completed_or_terminal_calls": len(ledger), "expected_calls": len(calls),
                "prior_campaign_spend_usd": round(prior_spend, 10),
                "fresh300_spend_usd": round(fresh_spend, 10),
                "cumulative_campaign_spend_usd": round(cumulative, 10),
                "hard_stop_usd": HARD_STOP_USD,
                "truncated_failures": sum(item["status"] == "truncated_failure" for item in ledger.values()),
                "halt_reason": "cumulative_actual_spend_above_1.00" if cumulative > HARD_STOP_USD else None,
                "last_call_id": row["call_id"], "updated_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })
            print(json.dumps({key: record[key] for key in (
                "call_id", "arm", "intent", "usage_prompt_tokens", "usage_completion_tokens",
                "finish_reason", "status", "actual_cost_usd", "cumulative_campaign_spend_usd",
            )}), flush=True)
            if cumulative > HARD_STOP_USD:
                print("HARD STOP: cumulative usage spend exceeded $1.00; checkpoint saved.", flush=True)
                break
    final = {
        "terminal_calls": len(ledger), "expected_calls": len(calls),
        "remaining_calls": len(calls) - len(ledger),
        "prior_campaign_spend_usd": round(prior_spend, 6),
        "fresh300_spend_usd": round(fresh_spend, 6),
        "cumulative_campaign_spend_usd": round(cumulative, 6),
        "truncated_failures": sum(record.get("status") == "truncated_failure" for record in ledger.values()),
        "request_errors": sum(record.get("status") in {"request_error", "usage_missing"} for record in ledger.values()),
        "halt_reason": "complete" if len(ledger) == len(calls) else "hard_stop_or_interruption",
    }
    atomic_json(CHECKPOINT_PATH, final)
    return final


def entity_counts_from_frozen(frozen: dict[str, Any]) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
    entity_counts = {key: int(value) for key, value in frozen["entity_count_map"].items()}
    property_counts = {}
    for key, value in frozen["entity_property_count_map"].items():
        entity, prop = key.split("\t", 1)
        property_counts[(entity, prop)] = int(value)
    return entity_counts, property_counts


def score_saved() -> dict[str, Any]:
    frozen, calls = verify_frozen()
    if not LEDGER_PATH.is_file():
        raise RuntimeError("no call ledger to score")
    records = [json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({row["call_id"] for row in records}) != len(records):
        raise RuntimeError("duplicate call ID in ledger")
    by_id = {row["call_id"]: row for row in records}
    by_manifest = {row["call_id"]: row for row in calls}
    from scripts.lib.prof_feedback import QueryPlan
    from scripts.lib.prof_feedback_final import execute_final_plan
    from scripts.lib.evaluation_scorer import score_answer

    entity_counts, property_counts = entity_counts_from_frozen(frozen)
    scored = []
    for call_id, row in by_manifest.items():
        record = by_id.get(call_id, {"status": "not_run"})
        prediction: Any = None
        answer_status = "abstained"
        if record.get("status") == "completed":
            if row["arm"] == "aletheia":
                try:
                    response_obj = json.loads(record.get("response_text", ""))
                    plan = QueryPlan.from_object(response_obj["query_plan"])
                    _final_plan, answer, _complete = execute_final_plan(
                        row["question"], plan, row["retrieved_items"], entity_counts, property_counts,
                    )
                    prediction = answer.get("answer")
                    answer_status = answer.get("status", "abstained")
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    prediction = None
                    answer_status = "invalid_plan"
            else:
                prediction = record.get("response_text", "")
                answer_status = "answered" if prediction else "abstained"
        correct = score_answer(row["intent"], prediction, row["ground_truth_answer"])
        scored.append({
            "call_id": call_id, "arm": row["arm"], "intent": row["intent"],
            "flagged": bool(row["source_gold_flagged"]),
            "answered": answer_status == "answered", "answer_status": answer_status,
            "correct": bool(correct), "prediction": prediction,
            "gold": row["ground_truth_answer"],
        })
    summaries = {}
    for arm in ("aletheia", "direct", "cot"):
        summaries[arm] = {}
        for intent in ("historical", "aggregation", "boolean"):
            rows = [item for item in scored if item["arm"] == arm and item["intent"] == intent]
            summaries[arm][intent] = summarize(rows)
            summaries[arm][f"{intent}_excluding_flagged"] = summarize([row for row in rows if not row["flagged"]])
        arm_rows = [row for row in scored if row["arm"] == arm]
        summaries[arm]["overall"] = summarize(arm_rows)
        summaries[arm]["overall_excluding_flagged"] = summarize([row for row in arm_rows if not row["flagged"]])
    total_prompt = sum(int(row.get("usage_prompt_tokens") or 0) for row in records)
    total_completion = sum(int(row.get("usage_completion_tokens") or 0) for row in records)
    total_spend = sum(float(row.get("actual_cost_usd") or 0) for row in records)
    spend_by_arm = {
        arm: round(sum(float(row.get("actual_cost_usd") or 0) for row in records if row.get("arm") == arm), 8)
        for arm in ("aletheia", "direct", "cot")
    }
    report = {
        "run_id": RUN_ID,
        "frozen_manifest_sha256": frozen["frozen_manifest_sha256"],
        "score_rule": "fixed evaluation_scorer.py; all audit flags retained; truncations and non-completed calls are failures",
        "calls_recorded": len(records),
        "expected_calls": 900,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "fresh300_spend_usd": round(total_spend, 8),
        "spend_by_arm_usd": spend_by_arm,
        "prior_campaign_spend_usd": frozen["prior_campaign_spend_baseline_usd"],
        "cumulative_campaign_spend_usd": round(frozen["prior_campaign_spend_baseline_usd"] + total_spend, 8),
        "truncated_failures": sum(row.get("status") == "truncated_failure" for row in records),
        "summaries": summaries,
    }
    (OUT / "scored_calls.json").write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "results_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    correct = sum(row["correct"] for row in rows)
    answered = sum(row["answered"] for row in rows)
    answered_correct = sum(row["correct"] and row["answered"] for row in rows)
    return {
        "n": count,
        "correct": correct,
        "accuracy": correct / count if count else 0.0,
        "answered": answered,
        "abstained": count - answered,
        "accuracy_when_answered": answered_correct / answered if answered else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true", help="build and print the immutable 900-call configuration offline")
    group.add_argument("--run", action="store_true", help="run/resume the already frozen calls; makes API calls")
    group.add_argument("--score", action="store_true", help="score the saved call ledger offline")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.run:
        print(json.dumps(run(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(score_saved(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
