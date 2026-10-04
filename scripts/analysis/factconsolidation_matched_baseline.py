"""Freeze and run the matched BM25 FactConsolidation baseline comparison.

All preparation and scoring are offline. Calls are made only by ``--run``
after a frozen manifest exists. Each call is sent at most once (SDK retries
disabled), usage is recorded per call, and the run halts after cumulative
observed usage exceeds the $0.10 ceiling.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_ID = "fc262k-bm25-matched-20261004-01"
OUT = ROOT / "results/prof_feedback" / RUN_ID
LOG = ROOT / "logs/prof_feedback" / RUN_ID
PIPELINE = ROOT / "scripts/lib/_pipeline.py"
PREPARE = ROOT / "scripts/analysis/prof_feedback_prepare.py"
SCRIPT = Path(__file__).resolve()
PHASE_A = ROOT / "results/prof_feedback/pf-20261004-phaseA-05/factconsolidation_manifest.json"
ADAPTIVE = ROOT / "results/poc_results/ablation_adaptive_fact_gpt4omini_factconsolidation_sh_262k.json"
EARLIER_BM25 = ROOT / "results/poc_results/paper_sh_conflict_factconsolidation_sh_262k.json"
ARROW = (Path.home() / ".cache/huggingface/datasets/ai-hyz___memory_agent_bench/default/0.0.0"
         / "7ea066982b140a19337e17e60d45d4076e042faf/memory_agent_bench-Conflict_Resolution.arrow")
FROZEN = OUT / "frozen_manifest.json"
REQUESTS = OUT / "requests.jsonl"
LEDGER = LOG / "calls.jsonl"
CHECKPOINT = LOG / "checkpoint.json"
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_COMPLETION_TOKENS = 512
INPUT_USD_PER_M = 0.15
OUTPUT_USD_PER_M = 0.60
HARD_STOP_USD = 0.10
SERIAL_RE = re.compile(r"(\d+)\.\s")


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


def pipeline_constants() -> tuple[str, str]:
    tree = ast.parse(PIPELINE.read_text(encoding="utf-8"))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"SYSTEM_MESSAGE", "QUERY_TEMPLATE_BM25"}:
                found[node.targets[0].id] = ast.literal_eval(node.value)
    return found["SYSTEM_MESSAGE"], found["QUERY_TEMPLATE_BM25"]


def parse_facts(context: str) -> list[dict[str, Any]]:
    matches = list(SERIAL_RE.finditer(context))
    facts = []
    seen = set()
    for i, match in enumerate(matches):
        serial = int(match.group(1))
        if serial in seen:
            continue
        seen.add(serial)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(context)
        facts.append({"fact_idx": serial, "text": context[match.end():end].strip().rstrip(".")})
    return facts


def token_count(messages: list[dict[str, str]], enc: Any) -> int:
    # Same local ChatML framing estimate used by the campaign preparation path.
    return 3 + sum(4 + len(enc.encode(m["role"])) + len(enc.encode(m["content"])) for m in messages)


def build_requests() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import tiktoken
    from rank_bm25 import BM25Okapi
    from scripts.lib.config import DATASET_REVISION

    if not ARROW.is_file() or DATASET_REVISION not in str(ARROW):
        raise RuntimeError(f"pinned FactConsolidation Arrow cache unavailable: {ARROW}")
    with pa.memory_map(str(ARROW), "r") as source:
        ds = ipc.open_stream(source).read_all().to_pylist()
    row = next((r for r in ds if r["metadata"].get("source") == "factconsolidation_sh_262k"), None)
    if row is None:
        raise RuntimeError("FactConsolidation SH-262K dataset row missing")
    facts = parse_facts(row["context"])
    texts = [f["text"] for f in facts]
    ids = [f["fact_idx"] for f in facts]
    bm25 = BM25Okapi([re.findall(r"[A-Za-z0-9]+", text.lower()) for text in texts])
    manifest = json.loads(PHASE_A.read_text(encoding="utf-8"))["factconsolidation_sh_262k"]["ordered_question_answer_pairs"]
    adaptive = json.loads(ADAPTIVE.read_text(encoding="utf-8"))["results"]
    old_bm25 = json.loads(EARLIER_BM25.read_text(encoding="utf-8"))["results"]
    if len(manifest) != 100 or len(adaptive) != 100 or len(old_bm25) != 100:
        raise RuntimeError("expected the same saved 100-question benchmark in each source")
    for i, (q, a, b) in enumerate(zip(manifest, adaptive, old_bm25)):
        if q["question"] != a["question"] or q["question"] != b["question"]:
            raise RuntimeError(f"question ordering differs at row {i}")
        if q["answer"] != a["ground_truth"] or q["answer"] != b["ground_truth"]:
            raise RuntimeError(f"gold ordering differs at row {i}")

    system, query_template = pipeline_constants()
    enc = tiktoken.get_encoding("o200k_base")
    requests = []
    route_counts = Counter()
    for item, prior in zip(manifest, adaptive):
        route = prior.get("route")
        k_routed = 25 if route == "historical" else 10
        route_counts[str(route)] += 1
        for arm, k in (("uniform_k10", 10), ("routed_k10_25", k_routed)):
            scores = bm25.get_scores(re.findall(r"[A-Za-z0-9]+", item["question"].lower()))
            positions = np.argsort(scores)[::-1][:k]
            retrieved = [{"fact_idx": ids[pos], "text": texts[pos]} for pos in positions]
            pool = "\n".join(f"{f['fact_idx']}. {f['text']}." for f in retrieved)
            user = f"[Knowledge Pool]\n{pool}\n\n{query_template.format(question=item['question'])}"
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            requests.append({
                "call_id": f"fc262k-q{item['source_index']:03d}-{arm}",
                "arm": arm, "dataset": "factconsolidation_sh_262k",
                "source_index": item["source_index"], "question": item["question"],
                "gold": item["answer"], "saved_adaptive_route": route, "k": k,
                "model": MODEL, "temperature": TEMPERATURE,
                "max_completion_tokens": MAX_COMPLETION_TOKENS,
                "messages": messages, "retrieved_items": retrieved,
                "local_input_tokens": token_count(messages, enc),
            })
    stats = {
        "dataset_revision": DATASET_REVISION,
        "dataset_arrow_sha256": sha_file(ARROW),
        "context_sha256": json.loads(PHASE_A.read_text(encoding="utf-8"))["factconsolidation_sh_262k"]["context_sha256"],
        "fact_count": len(facts), "questions": 100, "requests": len(requests),
        "saved_route_counts": dict(route_counts),
        "routed_k_counts": dict(Counter(str(r["k"]) for r in requests if r["arm"] == "routed_k10_25")),
        "estimated_prompt_tokens": sum(r["local_input_tokens"] for r in requests),
        "completion_token_cap_total": len(requests) * MAX_COMPLETION_TOKENS,
        "estimated_cost_at_full_completion_cap_usd": round(
            (sum(r["local_input_tokens"] for r in requests) * INPUT_USD_PER_M
             + len(requests) * MAX_COMPLETION_TOKENS * OUTPUT_USD_PER_M) / 1_000_000, 8),
        "ordered_questions_sha256": canonical_hash([(r["source_index"], r["question"], r["gold"]) for r in requests if r["arm"] == "uniform_k10"]),
    }
    return requests, stats


def freeze() -> dict[str, Any]:
    if OUT.exists() or LOG.exists():
        raise RuntimeError(f"refusing to overwrite existing run directory: {OUT} or {LOG}")
    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    requests, stats = build_requests()
    REQUESTS.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in requests), encoding="utf-8")
    manifest = {
        "run_id": RUN_ID, "model": MODEL, "temperature": TEMPERATURE,
        "completion_cap": MAX_COMPLETION_TOKENS,
        "input_usd_per_million": INPUT_USD_PER_M, "output_usd_per_million": OUTPUT_USD_PER_M,
        "hard_stop_usd": HARD_STOP_USD,
        "arms": {"uniform_k10": "BM25 top-10 for each question", "routed_k10_25": "saved Aletheia route: historical K=25; every other route K=10"},
        "system_prompt": "_pipeline.py SYSTEM_MESSAGE", "query_prompt": "_pipeline.py QUERY_TEMPLATE_BM25 (unchanged)",
        "scorer": "SubEM: case-insensitive gold substring in answer; same evaluate_answer rule as saved baseline",
        "sha256": {
            "runner": sha_file(SCRIPT), "pipeline_source": sha_file(PIPELINE),
            "prepare_source": sha_file(PREPARE), "phaseA_question_manifest": sha_file(PHASE_A),
            "prior_adaptive_results": sha_file(ADAPTIVE), "prior_bm25_results": sha_file(EARLIER_BM25),
            "requests": sha_file(REQUESTS),
            "system_prompt": sha_bytes(pipeline_constants()[0].encode()),
            "query_template_bm25": sha_bytes(pipeline_constants()[1].encode()),
            "scorer_rule": sha_bytes(b"SubEM: any(gold.lower() in (predicted or '').lower() for gold in ground_truth_list)"),
        },
        **stats,
    }
    raw = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    FROZEN.write_text(raw, encoding="utf-8")
    manifest["frozen_manifest_sha256"] = sha_file(FROZEN)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def verify_frozen() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(FROZEN.read_text(encoding="utf-8"))
    if manifest["sha256"]["runner"] != sha_file(SCRIPT):
        raise RuntimeError("runner source changed after freeze")
    if manifest["sha256"]["pipeline_source"] != sha_file(PIPELINE):
        raise RuntimeError("pipeline source changed after freeze")
    if manifest["sha256"]["requests"] != sha_file(REQUESTS):
        raise RuntimeError("frozen request set changed")
    rows = [json.loads(line) for line in REQUESTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 200 or len({r["call_id"] for r in rows}) != 200:
        raise RuntimeError("frozen request file must have 200 unique calls")
    return manifest, rows


def run() -> dict[str, Any]:
    manifest, requests = verify_frozen()
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        env_file = ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    os.environ["OPENAI_API_KEY"] = line.partition("=")[2].strip().strip("\"'")
                    break
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY unavailable; no request sent")
    client = OpenAI(max_retries=0, timeout=90)
    ledger = [json.loads(x) for x in LEDGER.read_text(encoding="utf-8").splitlines() if x.strip()] if LEDGER.exists() else []
    by_id = {r["call_id"]: r for r in ledger}
    if len(by_id) != len(ledger):
        raise RuntimeError("duplicate call IDs in append-only ledger")
    spend = sum(float(r.get("actual_cost_usd") or 0) for r in ledger)
    with LEDGER.open("a", encoding="utf-8") as stream:
        for row in requests:
            if row["call_id"] in by_id:
                continue
            started = time.time()
            try:
                response = client.chat.completions.create(
                    model=row["model"], temperature=row["temperature"],
                    max_completion_tokens=row["max_completion_tokens"],
                    messages=row["messages"],
                )
            except Exception as exc:
                rec = {"call_id": row["call_id"], "arm": row["arm"], "status": "request_error", "error_type": type(exc).__name__, "error": str(exc)[:500], "actual_cost_usd": None}
                stream.write(json.dumps(rec, ensure_ascii=False) + "\n"); stream.flush(); os.fsync(stream.fileno())
                by_id[row["call_id"]] = rec
                CHECKPOINT.write_text(json.dumps({"terminal_calls": len(by_id), "spend_usd": spend, "halt_reason": "request_error_no_retry", "last_call_id": row["call_id"]}, indent=2) + "\n")
                raise RuntimeError(f"halted after one un-retried request error at {row['call_id']}") from exc
            usage = response.usage
            if usage is None or usage.prompt_tokens is None or usage.completion_tokens is None:
                rec = {"call_id": row["call_id"], "arm": row["arm"], "status": "usage_missing", "actual_cost_usd": None}
                stream.write(json.dumps(rec, ensure_ascii=False) + "\n"); stream.flush(); os.fsync(stream.fileno())
                by_id[row["call_id"]] = rec
                CHECKPOINT.write_text(json.dumps({"terminal_calls": len(by_id), "spend_usd": spend, "halt_reason": "usage_missing", "last_call_id": row["call_id"]}, indent=2) + "\n")
                raise RuntimeError(f"halted: usage missing for {row['call_id']}; no retry")
            amount = usage.prompt_tokens * INPUT_USD_PER_M / 1_000_000 + usage.completion_tokens * OUTPUT_USD_PER_M / 1_000_000
            spend += amount
            rec = {
                "call_id": row["call_id"], "arm": row["arm"], "source_index": row["source_index"],
                "question": row["question"], "gold": row["gold"], "k": row["k"],
                "saved_adaptive_route": row["saved_adaptive_route"], "model": MODEL,
                "temperature": TEMPERATURE, "max_completion_tokens": MAX_COMPLETION_TOKENS,
                "status": "truncated_failure" if response.choices[0].finish_reason == "length" else "completed",
                "finish_reason": response.choices[0].finish_reason,
                "usage_prompt_tokens": usage.prompt_tokens, "usage_completion_tokens": usage.completion_tokens,
                "actual_cost_usd": round(amount, 10), "cumulative_spend_usd": round(spend, 10),
                "request_id": getattr(response, "_request_id", None),
                "elapsed_seconds": round(time.time() - started, 3),
                "response_text": response.choices[0].message.content or "",
            }
            stream.write(json.dumps(rec, ensure_ascii=False) + "\n"); stream.flush(); os.fsync(stream.fileno())
            by_id[row["call_id"]] = rec
            halt = spend > HARD_STOP_USD
            CHECKPOINT.write_text(json.dumps({"terminal_calls": len(by_id), "expected_calls": 200, "spend_usd": round(spend, 10), "hard_stop_usd": HARD_STOP_USD, "halt_reason": "spend_exceeded" if halt else None, "last_call_id": row["call_id"]}, indent=2) + "\n")
            print(json.dumps({k: rec[k] for k in ("call_id", "arm", "k", "usage_prompt_tokens", "usage_completion_tokens", "finish_reason", "actual_cost_usd", "cumulative_spend_usd")}), flush=True)
            if halt:
                break
    return {"terminal_calls": len(by_id), "expected_calls": 200, "new_spend_usd": round(spend, 8), "remaining_calls": 200-len(by_id), "hard_stop_usd": HARD_STOP_USD, "halt_reason": "complete" if len(by_id) == 200 else "spend_exceeded_or_interrupted"}


def score() -> dict[str, Any]:
    verify_frozen()
    ledger = [json.loads(x) for x in LEDGER.read_text(encoding="utf-8").splitlines() if x.strip()]
    results = {}
    for arm in ("uniform_k10", "routed_k10_25"):
        rows = [r for r in ledger if r.get("arm") == arm]
        correct = 0
        for r in rows:
            if r.get("status") != "completed" or r.get("finish_reason") == "length":
                continue
            answer = (r.get("response_text") or "").lower()
            correct += any(g.lower() in answer for g in r["gold"])
        results[arm] = {"n": len(rows), "correct": correct, "accuracy": correct / len(rows) if rows else None}
    out = {"run_id": RUN_ID, "frozen_manifest_sha256": sha_file(FROZEN), "results": results,
           "spend_usd": round(sum(float(r.get("actual_cost_usd") or 0) for r in ledger), 8),
           "prompt_tokens": sum(int(r.get("usage_prompt_tokens") or 0) for r in ledger),
           "completion_tokens": sum(int(r.get("usage_completion_tokens") or 0) for r in ledger),
           "truncated_failures": sum(r.get("status") == "truncated_failure" for r in ledger),
           "request_errors": sum(r.get("status") in {"request_error", "usage_missing"} for r in ledger)}
    (OUT / "results_summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true", help="prepare requests and freeze hashes offline")
    group.add_argument("--run", action="store_true", help="send the frozen calls once with no retries")
    group.add_argument("--score", action="store_true", help="score saved outputs offline")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.run:
        print(json.dumps(run(), indent=2))
    else:
        score()


if __name__ == "__main__":
    main()
