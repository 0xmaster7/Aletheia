"""Freeze, run once, and score the final 20261006 hybrid confirmation.

Preparation and scoring are offline. API calls occur only with ``--run`` after
the manifest and request list have been frozen. SDK retries are disabled and
actual usage is checkpointed after every call.
"""
from __future__ import annotations

from collections import Counter
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

RUN_ID = "pf-20261006-hybrid300-final-01"
PHASE_A = ROOT / "results/prof_feedback/pf-20261004-phaseA-05"
SMOKE = ROOT / "results/prof_feedback/pf-20261004-smoke-01"
V2 = ROOT / "results/prof_feedback/pf-20261004-fresh300-v2-01/requests.jsonl"
V3 = ROOT / "results/prof_feedback/pf-20261005-final300-v3-01/requests.jsonl"
OUT = ROOT / "results/prof_feedback" / RUN_ID
LOG = ROOT / "logs/prof_feedback" / RUN_ID
BENCHMARK = ROOT / "data/synthetic_benchmark.json"
GOLD_AUDIT = PHASE_A / "gold_support.json"
PROMPT = ROOT / "scripts/prompts/evaluation_final_v3.txt"
OPERATOR = ROOT / "scripts/lib/prof_feedback_final.py"
BASE_OPERATOR = ROOT / "scripts/lib/prof_feedback.py"
SCORER = ROOT / "scripts/lib/evaluation_scorer.py"
RUNNER = Path(__file__).resolve()
REQUESTS = OUT / "requests.jsonl"
POLICY_FILE = OUT / "hybrid_policy.txt"
MANIFEST = OUT / "frozen_manifest.json"
LEDGER = LOG / "calls.jsonl"
CHECKPOINT = LOG / "checkpoint.json"

SEED = 20261006
N = 300
K = 80
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
CAPS = {"aletheia": 1024, "direct": 512, "cot": 1024}
INPUT_USD_PER_M = 0.15
OUTPUT_USD_PER_M = 0.60
HARD_STOP_USD = 0.40
ARROW = (Path.home() / ".cache/huggingface/datasets/ai-hyz___memory_agent_bench/default/0.0.0"
         / "7ea066982b140a19337e17e60d45d4076e042faf/memory_agent_bench-Conflict_Resolution.arrow")
SERIAL_RE = re.compile(r"(\d+)\.\s")

HYBRID_POLICY = """Frozen hybrid policy, final confirmatory run

For each question, use Aletheia's deterministic answer whenever its final-v3
deterministic path answered: a supported non-Boolean plan with verified
complete history, or a Boolean answer accepted by the positive-evidence gate
(with False permitted only when verified complete entity history establishes
absence). Otherwise use the same-question Direct answer. The CoT-fallback
variant applies exactly the same gate and uses that question's CoT answer
instead. Apply this rule uniformly across all intents, with no exceptions.
Score both hybrids offline with the frozen final-v3 scorer. Truncations,
invalid plans, non-Boolean unsupported plans, and operator abstentions make
Aletheia ineligible for selection and invoke the named fallback. For Boolean,
the final-v3 operator's positive-evidence gate may answer from a matching fact
even if the model's original supported flag was false.
"""


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")).encode("utf-8"))


def parse_corpus_facts(context: str) -> list[dict[str, Any]]:
    matches = list(SERIAL_RE.finditer(context))
    rows, seen = [], set()
    for i, match in enumerate(matches):
        serial = int(match.group(1))
        if serial in seen:
            continue
        seen.add(serial)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(context)
        rows.append({"fact_idx": serial, "text": context[match.end():end].strip().rstrip(".")})
    return rows


def question_indices(path: Path, *, arm: str | None = None) -> set[int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {int(r["source_index"]) for r in rows if arm is None or r.get("arm") == arm}


def previously_used_indices() -> tuple[set[int], dict[str, int]]:
    phase = {int(r["source_index"]) for r in json.loads((PHASE_A / "synthetic_300.json").read_text(encoding="utf-8"))}
    ood = {int(r["source_index"]) for r in json.loads((SMOKE / "ood_60.json").read_text(encoding="utf-8"))}
    v2 = question_indices(V2, arm="aletheia")
    v3 = question_indices(V3, arm="aletheia")
    # The smoke/full campaign log also contains four separate FactConsolidation
    # datasets whose local row numbers are unrelated to synthetic source IDs.
    calls = [json.loads(line) for line in (ROOT / "logs/prof_feedback/pf-20261004-smoke-01/full_calls.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    logged_synthetic = {int(r["source_index"]) for r in calls if r.get("dataset") == "synthetic300"}
    logged_ood = {int(r["source_index"]) for r in calls if r.get("dataset") == "ood60"}
    if logged_synthetic != phase or logged_ood != ood:
        raise RuntimeError("saved campaign log and development/OOD question inventories differ")
    if len(phase) != 300 or len(ood) != 60 or len(v2) != 300 or len(v3) != 300:
        raise RuntimeError("expected the four prior synthetic question groups to have 300/60/300/300 unique IDs")
    groups = {"development_synthetic300": phase, "ood60_source_questions": ood,
              "fresh300_v2": v2, "final300_v3": v3}
    union = set().union(*groups.values())
    # Confirm groups are disjoint; all used question IDs must be excluded once.
    if len(union) != sum(len(s) for s in groups.values()):
        raise RuntimeError("previous synthetic source-question groups overlap unexpectedly")
    return union, {name: len(values) for name, values in groups.items()}


def build_requests() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import tiktoken
    from rank_bm25 import BM25Okapi
    from scripts.lib.config import DATASET_REVISION
    from scripts.lib.prof_feedback import parse_fact_template, property_type_from_fact
    from scripts.lib.prof_feedback_final import build_final_evaluation_request
    from scripts.lib.prof_feedback import build_control_request

    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    audit = json.loads(GOLD_AUDIT.read_text(encoding="utf-8"))
    if len(benchmark) != 13_425 or len(audit) != 13_425:
        raise RuntimeError("benchmark/audit row count changed")
    used, used_counts = previously_used_indices()
    eligible = [i for i in range(len(benchmark)) if i not in used]
    rng = random.Random(20261006)
    selected_indices = rng.sample(eligible, N)
    support = {int(r["source_index"]): r for r in audit}
    selected = []
    for source_index in selected_indices:
        item = dict(benchmark[source_index])
        a = support[source_index]
        item.update({"source_index": source_index,
                     "question_id": f"syn-{source_index:05d}-{a['question_sha256'][:12]}",
                     "source_gold_flagged": a["status"] != "verified",
                     "source_gold_issue_codes": a["issue_codes"],
                     "source_support_status": a["status"]})
        selected.append(item)

    if not ARROW.is_file() or DATASET_REVISION not in str(ARROW):
        raise RuntimeError(f"pinned local dataset cache unavailable: {ARROW}")
    with pa.memory_map(str(ARROW), "r") as source:
        dataset = ipc.open_stream(source).read_all().to_pylist()
    source_row = next((r for r in dataset if r["metadata"].get("source") == "factconsolidation_sh_262k"), None)
    if source_row is None:
        raise RuntimeError("pinned FactConsolidation corpus row missing")
    corpus = parse_corpus_facts(source_row["context"])
    corpus_texts = [r["text"] for r in corpus]
    bm25 = BM25Okapi([re.findall(r"[A-Za-z0-9]+", text.lower()) for text in corpus_texts])

    selected_entities = {r["entity"] for r in selected}
    entity_fact_counts: Counter[str] = Counter()
    entity_property_fact_counts: Counter[tuple[str, str]] = Counter()
    raw_subject_counts: Counter[str] = Counter()
    for fact in corpus:
        for entity in selected_entities:
            if fact["text"].startswith(entity + " "):
                raw_subject_counts[entity] += 1
                parsed = parse_fact_template(fact["text"])
                if parsed and parsed[0] == entity:
                    entity_fact_counts[entity] += 1
                    entity_property_fact_counts[(entity, property_type_from_fact(fact["text"]))] += 1
    reliable_counts = {e: entity_fact_counts[e] for e in selected_entities
                       if entity_fact_counts[e] == raw_subject_counts[e]}

    enc = tiktoken.get_encoding("o200k_base")
    calls = []
    for position, item in enumerate(selected, 1):
        tokens = re.findall(r"[A-Za-z0-9]+", item["question_text"].lower())
        scores = bm25.get_scores(tokens)
        fact_positions = np.argsort(scores)[::-1][:K]
        retrieved = [{"fact_idx": corpus[i]["fact_idx"], "text": corpus[i]["text"]} for i in fact_positions]
        for arm in ("aletheia", "direct", "cot"):
            req = (build_final_evaluation_request(item["question_text"], retrieved) if arm == "aletheia"
                   else build_control_request(arm, item["question_text"], retrieved))
            schema = req.get("response_format", {})
            schema_raw = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
            local_tokens = 3 + sum(4 + len(enc.encode(m["role"])) + len(enc.encode(m["content"])) for m in req["messages"])
            if schema:
                local_tokens += len(enc.encode(schema_raw))
            calls.append({
                "call_id": f"hybrid300-{position:03d}-{arm}", "arm": arm,
                "dataset": RUN_ID, "source_index": item["source_index"],
                "question_id": item["question_id"], "entity": item["entity"],
                "intent": item["intent"], "question": item["question_text"],
                "ground_truth_answer": item["ground_truth_answer"],
                "source_gold_flagged": item["source_gold_flagged"],
                "source_gold_issue_codes": item["source_gold_issue_codes"],
                "source_support_status": item["source_support_status"],
                "k": K, "model": MODEL, "temperature": TEMPERATURE,
                "max_completion_tokens": CAPS[arm], "messages": req["messages"],
                "response_format": req.get("response_format"), "retrieved_items": retrieved,
                "local_input_tokens": local_tokens,
            })
    if len(calls) != 900 or Counter(r["arm"] for r in calls) != {"aletheia": 300, "direct": 300, "cot": 300}:
        raise RuntimeError("expected exactly 300 requests per arm")
    contexts: dict[int, set[str]] = {}
    for row in calls:
        contexts.setdefault(row["source_index"], set()).add(canonical_hash(row["retrieved_items"]))
    if any(len(values) != 1 for values in contexts.values()):
        raise RuntimeError("arms do not share the same K=80 contexts")
    stats = {
        "sample_seed": SEED,
        "sample_method": "random.Random(20261006).sample(eligible_source_indices, 300)",
        "prior_source_question_group_counts": used_counts,
        "excluded_prior_question_count": len(used),
        "excluded_prior_source_indices_sha256": canonical_hash(sorted(used)),
        "eligible_question_count": len(eligible),
        "selected_source_indices": selected_indices,
        "selected_source_indices_sha256": canonical_hash(selected_indices),
        "selected_intent_counts": dict(Counter(r["intent"] for r in selected)),
        "selected_flagged_count": sum(r["source_gold_flagged"] for r in selected),
        "entity_count_map": dict(reliable_counts),
        "entity_property_count_map": {f"{e}\t{p}": c for (e, p), c in entity_property_fact_counts.items()},
        "unreliable_entity_count_names": sorted(selected_entities - set(reliable_counts)),
        "local_corpus_fact_count": len(corpus),
        "estimated_prompt_tokens": sum(r["local_input_tokens"] for r in calls),
        "max_output_tokens_at_caps": sum(r["max_completion_tokens"] for r in calls),
    }
    return calls, stats


def freeze() -> dict[str, Any]:
    if OUT.exists() or LOG.exists():
        raise RuntimeError(f"refusing to overwrite existing run artifacts: {RUN_ID}")
    from scripts.lib.prof_feedback_final import TEMPORAL_CUE_MAP, KNOWN_RELATION_LIST, COUNT_CUES
    from scripts.lib.config import DATASET_REVISION
    calls, stats = build_requests()
    OUT.mkdir(parents=True)
    (OUT / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
    POLICY_FILE.write_text(HYBRID_POLICY, encoding="utf-8")
    raw_requests = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in calls).encode("utf-8")
    REQUESTS.write_bytes(raw_requests)
    from scripts.lib.prof_feedback_final import TEMPORAL_CUE_MAP, KNOWN_RELATION_LIST, COUNT_CUES
    frozen = {
        "run_id": RUN_ID, "frozen_at_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "model": MODEL, "temperature": TEMPERATURE, "k": K, "calls_per_arm": N,
        "completion_caps": CAPS, "hard_stop_usd_this_run": HARD_STOP_USD,
        "hybrid_policy": HYBRID_POLICY,
        "policy_application": "uniform across intents; no exceptions",
        "hybrid_policy_sha256": sha_file(POLICY_FILE),
        "final_v3_prompt_sha256": sha_file(PROMPT),
        "final_v3_operator_sha256": sha_file(OPERATOR),
        "base_operator_dependency_sha256": sha_file(BASE_OPERATOR),
        "scorer_sha256": sha_file(SCORER),
        "temporal_cue_list_sha256": canonical_hash(TEMPORAL_CUE_MAP),
        "relation_list_sha256": canonical_hash(KNOWN_RELATION_LIST),
        "count_cue_list_sha256": canonical_hash([p.pattern for p in COUNT_CUES]),
        "seed_expression_sha256": sha_bytes(b"random.Random(20261006)"),
        "runner_sha256": sha_file(RUNNER),
        "request_manifest_sha256": sha_file(REQUESTS),
        "policy_file_sha256": sha_file(POLICY_FILE),
        "benchmark_sha256": sha_file(BENCHMARK),
        "gold_audit_sha256": sha_file(GOLD_AUDIT),
        "dataset_revision": DATASET_REVISION,
        "dataset_arrow_sha256": sha_file(ARROW),
        "factconsolidation_context_sha256": json.loads((PHASE_A / "factconsolidation_manifest.json").read_text(encoding="utf-8"))["factconsolidation_sh_262k"]["context_sha256"],
        "input_usd_per_million": INPUT_USD_PER_M,
        "output_usd_per_million": OUTPUT_USD_PER_M,
        "scorer_rule": "unchanged final-v3 scripts/lib/evaluation_scorer.py; truncations and invalid/noncompleted calls fail",
        "sha256_definitions": {
            "prompt": "scripts/prompts/evaluation_final_v3.txt",
            "operator": "scripts/lib/prof_feedback_final.py",
            "base_operator": "scripts/lib/prof_feedback.py",
            "scorer": "scripts/lib/evaluation_scorer.py",
            "runner": str(RUNNER.relative_to(ROOT)),
            "policy": str(POLICY_FILE.relative_to(ROOT)),
            "requests": str(REQUESTS.relative_to(ROOT)),
        },
        **stats,
    }
    frozen["frozen_manifest_sha256"] = canonical_hash(frozen)
    MANIFEST.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: frozen[k] for k in (
        "run_id", "model", "temperature", "k", "completion_caps", "hard_stop_usd_this_run",
        "sample_seed", "sample_method", "prior_source_question_group_counts",
        "excluded_prior_question_count", "excluded_prior_source_indices_sha256",
        "eligible_question_count", "selected_intent_counts", "selected_flagged_count",
        "selected_source_indices_sha256", "final_v3_prompt_sha256", "final_v3_operator_sha256",
        "base_operator_dependency_sha256", "scorer_sha256", "temporal_cue_list_sha256",
        "relation_list_sha256", "count_cue_list_sha256", "hybrid_policy_sha256",
        "seed_expression_sha256", "runner_sha256", "request_manifest_sha256",
        "estimated_prompt_tokens", "max_output_tokens_at_caps", "frozen_manifest_sha256",
    )}, ensure_ascii=False, indent=2))
    return frozen


def verify_frozen() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from scripts.lib.prof_feedback_final import TEMPORAL_CUE_MAP, KNOWN_RELATION_LIST, COUNT_CUES
    frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))
    checks = {
        "final_v3_prompt_sha256": sha_file(PROMPT),
        "final_v3_operator_sha256": sha_file(OPERATOR),
        "base_operator_dependency_sha256": sha_file(BASE_OPERATOR),
        "scorer_sha256": sha_file(SCORER),
        "hybrid_policy_sha256": sha_file(POLICY_FILE),
        "policy_file_sha256": sha_file(POLICY_FILE),
        "seed_expression_sha256": sha_bytes(b"random.Random(20261006)"),
        "runner_sha256": sha_file(RUNNER),
        "request_manifest_sha256": sha_file(REQUESTS),
        "temporal_cue_list_sha256": canonical_hash(TEMPORAL_CUE_MAP),
        "relation_list_sha256": canonical_hash(KNOWN_RELATION_LIST),
        "count_cue_list_sha256": canonical_hash([p.pattern for p in COUNT_CUES]),
    }
    for key, value in checks.items():
        if frozen[key] != value:
            raise RuntimeError(f"frozen configuration hash mismatch: {key}")
    check = dict(frozen)
    fingerprint = check.pop("frozen_manifest_sha256")
    if canonical_hash(check) != fingerprint:
        raise RuntimeError("manifest fingerprint mismatch")
    calls = [json.loads(line) for line in REQUESTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(calls) != 900:
        raise RuntimeError("frozen request set must have 900 calls")
    return frozen, calls


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def run() -> dict[str, Any]:
    frozen, calls = verify_frozen()
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip("\"'"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY unavailable; no call was sent")
    from openai import OpenAI
    client = OpenAI(max_retries=0, timeout=120)
    ledger = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()] if LEDGER.exists() else []
    by_id = {r["call_id"]: r for r in ledger}
    if len(by_id) != len(ledger):
        raise RuntimeError("duplicate call IDs in ledger")
    if any(r.get("status") in {"request_error", "usage_missing"} for r in ledger):
        raise RuntimeError("unknown billed usage in ledger; refusing to continue")
    spend = sum(float(r.get("actual_cost_usd") or 0) for r in ledger)
    if spend >= HARD_STOP_USD:
        return {"calls": len(ledger), "expected": 900, "spend_usd": round(spend, 8), "halt_reason": "hard_stop_already_reached"}
    LOG.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8", buffering=1) as stream:
        for row in calls:
            if row["call_id"] in by_id:
                continue
            if spend >= HARD_STOP_USD:
                atomic_json(CHECKPOINT, {"terminal_calls": len(by_id), "expected_calls": 900,
                    "spend_usd": round(spend, 10), "hard_stop_usd": HARD_STOP_USD,
                    "halt_reason": "actual_spend_at_or_above_ceiling"})
                break
            request = {k: row[k] for k in ("model", "messages", "temperature", "max_completion_tokens")}
            if row["arm"] == "aletheia":
                request["response_format"] = row["response_format"]
            started = time.time()
            try:
                response = client.chat.completions.create(**request)
            except Exception as exc:
                rec = {"call_id": row["call_id"], "arm": row["arm"], "source_index": row["source_index"],
                       "status": "request_error", "error_type": type(exc).__name__,
                       "error": str(exc).replace(os.environ.get("OPENAI_API_KEY", ""), "[redacted]")[:1000],
                       "usage_prompt_tokens": None, "usage_completion_tokens": None, "actual_cost_usd": None}
                stream.write(json.dumps(rec, ensure_ascii=False) + "\n"); stream.flush(); os.fsync(stream.fileno())
                by_id[row["call_id"]] = rec
                atomic_json(CHECKPOINT, {"terminal_calls": len(by_id), "expected_calls": 900,
                    "spend_usd": round(spend, 10), "hard_stop_usd": HARD_STOP_USD,
                    "halt_reason": "request_error_no_retry", "last_call_id": row["call_id"]})
                raise RuntimeError(f"halted after one request error; no retry: {row['call_id']}") from exc
            usage = response.usage
            if usage is None or usage.prompt_tokens is None or usage.completion_tokens is None:
                rec = {"call_id": row["call_id"], "arm": row["arm"], "status": "usage_missing",
                       "actual_cost_usd": None}
                stream.write(json.dumps(rec, ensure_ascii=False) + "\n"); stream.flush(); os.fsync(stream.fileno())
                by_id[row["call_id"]] = rec
                atomic_json(CHECKPOINT, {"terminal_calls": len(by_id), "expected_calls": 900,
                    "spend_usd": round(spend, 10), "hard_stop_usd": HARD_STOP_USD,
                    "halt_reason": "usage_missing", "last_call_id": row["call_id"]})
                raise RuntimeError(f"halted: usage missing; no retry: {row['call_id']}")
            amount = usage.prompt_tokens * INPUT_USD_PER_M / 1_000_000 + usage.completion_tokens * OUTPUT_USD_PER_M / 1_000_000
            spend += amount
            finish = response.choices[0].finish_reason
            rec = {
                "call_id": row["call_id"], "arm": row["arm"], "dataset": RUN_ID,
                "source_index": row["source_index"], "question_id": row["question_id"],
                "intent": row["intent"], "ground_truth_answer": row["ground_truth_answer"],
                "source_gold_flagged": row["source_gold_flagged"],
                "source_gold_issue_codes": row["source_gold_issue_codes"],
                "status": "truncated_failure" if finish == "length" else "completed",
                "max_completion_tokens": row["max_completion_tokens"],
                "usage_prompt_tokens": usage.prompt_tokens, "usage_completion_tokens": usage.completion_tokens,
                "finish_reason": finish, "request_id": getattr(response, "_request_id", None),
                "actual_cost_usd": round(amount, 10), "cumulative_spend_usd": round(spend, 10),
                "elapsed_seconds": round(time.time() - started, 3),
                "response_text": response.choices[0].message.content or "",
            }
            stream.write(json.dumps(rec, ensure_ascii=False) + "\n"); stream.flush(); os.fsync(stream.fileno())
            by_id[row["call_id"]] = rec
            halt = spend >= HARD_STOP_USD
            atomic_json(CHECKPOINT, {"terminal_calls": len(by_id), "expected_calls": 900,
                "spend_usd": round(spend, 10), "hard_stop_usd": HARD_STOP_USD,
                "halt_reason": "actual_spend_at_or_above_ceiling" if halt else None,
                "truncated_failures": sum(x.get("status") == "truncated_failure" for x in by_id.values()),
                "last_call_id": row["call_id"], "updated_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            print(json.dumps({k: rec[k] for k in ("call_id", "arm", "intent", "usage_prompt_tokens",
                "usage_completion_tokens", "finish_reason", "actual_cost_usd", "cumulative_spend_usd")}), flush=True)
            if halt:
                break
    return {"terminal_calls": len(by_id), "expected_calls": 900, "remaining_calls": 900-len(by_id),
            "spend_usd": round(spend, 8), "hard_stop_usd": HARD_STOP_USD,
            "halt_reason": "complete" if len(by_id) == 900 else "hard_stop_or_interruption"}


def score() -> dict[str, Any]:
    frozen, calls = verify_frozen()
    logs = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({r["call_id"] for r in logs}) != len(logs):
        raise RuntimeError("duplicate calls in saved ledger")
    log_by_id = {r["call_id"]: r for r in logs}
    from scripts.lib.prof_feedback import QueryPlan
    from scripts.lib.prof_feedback_final import execute_final_plan
    from scripts.lib.evaluation_scorer import score_answer
    entity_counts = frozen["entity_count_map"]
    property_counts = {tuple(k.split("\t", 1)): v for k, v in frozen["entity_property_count_map"].items()}
    scored = []
    by_source: dict[int, dict[str, Any]] = {}
    for row in calls:
        rec = log_by_id.get(row["call_id"], {"status": "not_run"})
        prediction = None
        answered = False
        complete = False
        if rec.get("status") == "completed":
            if row["arm"] == "aletheia":
                try:
                    parsed = json.loads(rec.get("response_text", ""))
                    plan = QueryPlan.from_object(parsed["query_plan"])
                    _, result, complete = execute_final_plan(row["question"], plan, row["retrieved_items"], entity_counts, property_counts)
                    prediction = result.get("answer")
                    answered = result.get("status") == "answered"
                    if plan.intent != "boolean":
                        answered = bool(plan.supported and complete and answered)
                    else:
                        # The frozen gate answers True from positive evidence;
                        # False is available only through complete history. The
                        # final-v3 Boolean operator permits positive evidence
                        # despite the model's supported=false flag.
                        answered = bool(plan.entity and plan.boolean_target and answered)
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    prediction, answered, complete = None, False, False
            else:
                prediction = rec.get("response_text", "")
                answered = bool(prediction)
        correct = bool(score_answer(row["intent"], prediction, row["ground_truth_answer"]))
        result = {"call_id": row["call_id"], "source_index": row["source_index"],
                  "arm": row["arm"], "intent": row["intent"], "flagged": row["source_gold_flagged"],
                  "answered": answered, "complete_history": complete, "correct": correct,
                  "prediction": prediction, "gold": row["ground_truth_answer"], "status": rec.get("status")}
        scored.append(result)
        by_source.setdefault(int(row["source_index"]), {})[row["arm"]] = result
    hybrids = []
    for idx, arms in by_source.items():
        a = arms.get("aletheia", {})
        for fallback in ("direct", "cot"):
            used_aletheia = bool(a.get("answered"))
            source = "aletheia" if used_aletheia else fallback
            selected = arms.get(source, {})
            hybrids.append({"source_index": idx, "fallback": fallback, "intent": a.get("intent"),
                "flagged": a.get("flagged", False), "source": source,
                "deterministic": used_aletheia, "prediction": selected.get("prediction"),
                "gold": a.get("gold"), "correct": bool(selected.get("correct", False))})

    def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows); c = sum(bool(r["correct"]) for r in rows)
        return {"n": n, "correct": c, "accuracy": c / n if n else None}

    results: dict[str, Any] = {}
    groups = [(arm, [r for r in scored if r["arm"] == arm]) for arm in ("aletheia", "direct", "cot")]
    for arm, rows in groups:
        results[arm] = {}
        for intent in ("historical", "aggregation", "boolean", "overall"):
            subset = rows if intent == "overall" else [r for r in rows if r["intent"] == intent]
            results[arm][intent] = summary(subset)
            results[arm][intent + "_excluding_flagged"] = summary([r for r in subset if not r["flagged"]])
        results[arm]["answered_count"] = sum(r["answered"] for r in rows)
        results[arm]["abstained_count"] = len(rows) - results[arm]["answered_count"]
    for fallback in ("direct", "cot"):
        rows = [r for r in hybrids if r["fallback"] == fallback]
        results["hybrid_" + fallback + "_fallback"] = {}
        for intent in ("historical", "aggregation", "boolean", "overall"):
            subset = rows if intent == "overall" else [r for r in rows if r["intent"] == intent]
            s = summary(subset)
            deterministic = sum(bool(r["deterministic"]) for r in subset)
            answers = sum(r["prediction"] is not None for r in subset)
            s.update({"deterministic_count": deterministic,
                      "deterministic_share_of_answers": deterministic / answers if answers else None,
                      "answers": answers})
            results["hybrid_" + fallback + "_fallback"][intent] = s
            results["hybrid_" + fallback + "_fallback"][intent + "_excluding_flagged"] = summary([r for r in subset if not r["flagged"]])

    subset_scores = {}
    for a_subset in ("answered", "abstained"):
        for intent in ("historical", "aggregation", "boolean", "overall"):
            chosen = [r for r in scored if r["arm"] == "aletheia" and r["answered"] == (a_subset == "answered")
                      and (intent == "overall" or r["intent"] == intent)]
            indices = {r["source_index"] for r in chosen}
            subset_scores[f"{a_subset}:{intent}"] = {}
            for arm in ("aletheia", "direct", "cot"):
                vals = [r for r in scored if r["arm"] == arm and r["source_index"] in indices]
                subset_scores[f"{a_subset}:{intent}"][arm] = summary(vals)
                subset_scores[f"{a_subset}:{intent}"][arm + "_excluding_flagged"] = summary([r for r in vals if not r["flagged"]])
            subset_scores[f"{a_subset}:{intent}"]["n_retained"] = len(chosen)
            subset_scores[f"{a_subset}:{intent}"]["n_unflagged"] = sum(not r["flagged"] for r in chosen)
    prompt_tokens = sum(int(r.get("usage_prompt_tokens") or 0) for r in logs)
    completion_tokens = sum(int(r.get("usage_completion_tokens") or 0) for r in logs)
    spend = sum(float(r.get("actual_cost_usd") or 0) for r in logs)
    report = {"run_id": RUN_ID, "frozen_manifest_sha256": frozen["frozen_manifest_sha256"],
              "calls_recorded": len(logs), "expected_calls": 900,
              "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
              "spend_usd": round(spend, 8), "spend_by_arm_usd": {
                  arm: round(sum(float(r.get("actual_cost_usd") or 0) for r in logs if r.get("arm") == arm), 8)
                  for arm in ("aletheia", "direct", "cot")},
              "truncated_failures": sum(r.get("status") == "truncated_failure" for r in logs),
              "request_errors": sum(r.get("status") in {"request_error", "usage_missing"} for r in logs),
              "arm_and_hybrid_summaries": results, "aletheia_answered_subset_comparison": subset_scores}
    (OUT / "scored_calls.json").write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "hybrid_scored_calls.json").write_text(json.dumps(hybrids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "results_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true", help="prepare sample and frozen config offline")
    group.add_argument("--run", action="store_true", help="send frozen 900 calls, with no retries")
    group.add_argument("--score", action="store_true", help="score saved calls and hybrids offline")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.run:
        print(json.dumps(run(), ensure_ascii=False, indent=2))
    else:
        score()


if __name__ == "__main__":
    main()
