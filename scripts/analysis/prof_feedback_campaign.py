"""Prepare, cost, and run the professor-feedback campaign.

Manifest preparation is offline. ``--smoke`` sends exactly the 15 manifest
smoke requests and an explicitly authorized full campaign, with SDK retries
disabled and append-only checkpointing for resumable execution.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASE_A = ROOT / "results/prof_feedback/pf-20261004-phaseA-05"
RUN_ID = "pf-20261004-smoke-01"
OUT = ROOT / "results/prof_feedback" / RUN_ID
LOG = ROOT / "logs/prof_feedback" / RUN_ID
INPUT_USD_PER_M = 0.15
OUTPUT_USD_PER_M = 0.60
CEILING_USD = 0.05
FULL_CEILING_USD = 1.00
HARD_STOP_USD = 0.80
RESERVE = 0.20
CAPS = {"aletheia": 1024, "direct": 512, "cot": 1024, "factconsolidation": 512}
PROPOSED_K = 80
FULL_COUNTS = {"aletheia": 360, "direct": 360, "cot": 360, "factconsolidation": 400}

OOD_TEMPLATES = {
    "historical": (
        "Which {prop} is first documented for {entity}?",
        "Looking back to the start of the record, what {prop} value belongs to {entity}?",
        "In the earliest entry, what {prop} is associated with {entity}?",
        "Start with the oldest record for {entity}: what does it state about {prop}?",
        "What {prop} appears at the beginning of {entity}'s recorded timeline?",
    ),
    "aggregation_specific": (
        "Count the distinct {prop} values in {entity}'s record.",
        "How many separate {prop} values are recorded for {entity}?",
        "Across the record for {entity}, how many unique {prop} values occur?",
        "Tally the nonduplicate {prop} entries associated with {entity}.",
        "What number of different {prop} values is recorded for {entity}?",
    ),
    "aggregation_generic": (
        "How many distinct values occur anywhere in the record for {entity}?",
        "Count unique values across all properties recorded for {entity}.",
        "Across every fact about {entity}, how many different values appear?",
        "Tally the nonduplicate values associated with {entity}, regardless of property.",
        "What is the number of unique values in {entity}'s full record?",
    ),
    "boolean_specific": (
        "Does the record contain a {prop} entry for {entity} with the value {target}?",
        "Is {target} ever recorded as a {prop} of {entity}?",
        "Check the {prop} history for {entity}: does {target} appear?",
        "Does any recorded {prop} link {entity} to {target}?",
        "Is there evidence that {target} was a {prop} value for {entity}?",
    ),
    "boolean_generic": (
        "Does {target} appear among the values recorded for {entity}?",
        "Is {target} ever listed as a value associated with {entity}?",
        "Check every recorded property for {entity}: does any entry contain {target}?",
        "Is there a fact linking {entity} with the value {target}?",
        "Do the records include {target} anywhere among {entity}'s values?",
    ),
}


def load_env_file() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def project_constants() -> tuple[str, str]:
    tree = ast.parse((ROOT / "scripts/lib/_pipeline.py").read_text(encoding="utf-8"))
    found: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"SYSTEM_MESSAGE", "QUERY_TEMPLATE_BM25"}:
                found[name] = ast.literal_eval(node.value)
    return found["SYSTEM_MESSAGE"], found["QUERY_TEMPLATE_BM25"]


def parse_facts(context: str) -> list[dict[str, Any]]:
    # Match the original paper runner's serial extraction and first-serial rule.
    pat = re.compile(r"(\d+)\.\s")
    matches = list(pat.finditer(context))
    facts: list[dict[str, Any]] = []
    seen: set[int] = set()
    for i, match in enumerate(matches):
        serial = int(match.group(1))
        if serial in seen:
            continue
        seen.add(serial)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(context)
        facts.append({"fact_idx": serial, "text": context[match.end():end].strip().rstrip(".")})
    return facts


def toks(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def make_ood_rows(benchmark: list[dict[str, Any]], support: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from scripts.analysis.prof_feedback_prepare import make_sample

    _, _, reserved_entities = make_sample(benchmark)
    support_by_index = {row["source_index"]: row for row in support}
    ood: list[dict[str, Any]] = []
    reserved = set(reserved_entities)
    entity_order = list(dict.fromkeys(item["entity"] for item in benchmark if item["entity"] in reserved))
    entity_template_index = {entity: index % 5 for index, entity in enumerate(entity_order)}
    property_names = {
        "country": "country", "genre": "genre", "location": "location", "language": "language",
        "religion": "religion", "nationality": "nationality", "ethnicity": "ethnicity",
        "occupation": "occupation", "creator": "creator", "value": "value",
    }
    for source_index, row in enumerate(benchmark):
        if row["entity"] not in reserved:
            continue
        audit = support_by_index[source_index]
        prop_type = audit["question_property_type"]
        prop = property_names.get(prop_type, "value")
        template_index = entity_template_index[row["entity"]]
        if row["intent"] == "historical":
            template = OOD_TEMPLATES["historical"][template_index]
            ood_question = template.format(prop=prop, entity=row["entity"])
        elif row["intent"] == "aggregation":
            family = "aggregation_generic" if prop_type == "value" else "aggregation_specific"
            template = OOD_TEMPLATES[family][template_index]
            ood_question = template.format(prop=prop, entity=row["entity"])
        else:
            family = "boolean_generic" if prop_type == "value" else "boolean_specific"
            template = OOD_TEMPLATES[family][template_index]
            ood_question = template.format(prop=prop, entity=row["entity"], target=audit["boolean_target"] or "")
        ood.append({
            "ood_id": f"ood-{len(ood)+1:03d}",
            "source_index": source_index,
            "source_question": row["question_text"],
            "question_text": ood_question,
            "entity": row["entity"],
            "intent": row["intent"],
            "ground_truth_answer": row["ground_truth_answer"],
            "source_question_id": f"syn-{source_index:05d}-{audit['question_sha256'][:12]}",
            "source_gold_flagged": audit["status"] != "verified",
            "source_support_status": audit["status"],
            "source_gold_issue_codes": audit["issue_codes"],
            "transformation": "intent-specific linguistic rewrite; source question and gold retained verbatim as audit metadata",
        })
    if len(ood) != 60 or Counter(row["intent"] for row in ood) != {"historical": 20, "aggregation": 20, "boolean": 20}:
        raise RuntimeError(f"OOD reserve must be 60 balanced questions; got {len(ood)} / {Counter(r['intent'] for r in ood)}")
    return ood


def make_prompt_request(arm: str, question: str, retrieved: list[dict[str, Any]]) -> dict[str, Any]:
    from scripts.lib.prof_feedback import build_approved_evaluation_request, build_control_request

    if arm == "aletheia":
        return build_approved_evaluation_request(question, retrieved)
    if arm in {"direct", "cot"}:
        return build_control_request(arm, question, retrieved)
    raise ValueError(arm)


def request_local_tokens(request: dict[str, Any], enc: Any) -> int:
    # Count all literal message text, roles, strict schema JSON and modest
    # Chat Completions message framing. Later calibration uses usage.prompt_tokens.
    count = 3
    for message in request["messages"]:
        count += 4 + len(enc.encode(message["role"])) + len(enc.encode(message["content"]))
    if "response_format" in request:
        raw = json.dumps(request["response_format"], ensure_ascii=False, separators=(",", ":"))
        count += len(enc.encode(raw))
    return count


def api_request(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ("model", "messages", "max_completion_tokens", "temperature", "response_format") if key in row}


def prepare() -> dict[str, Any]:
    if OUT.exists() or LOG.exists():
        raise RuntimeError(f"refusing to overwrite existing run artifacts: {RUN_ID}")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["ALETHEIA_DISABLE_TRACING"] = "1"

    from datasets import load_dataset
    import numpy as np
    from rank_bm25 import BM25Okapi
    import tiktoken
    from scripts.lib.config import DATASET_REVISION
    from scripts.lib.prof_feedback import EVALUATION_MAX_COMPLETION_TOKENS

    if EVALUATION_MAX_COMPLETION_TOKENS != CAPS["aletheia"]:
        raise RuntimeError("Aletheia evaluation request completion cap is not 1024")
    enc = tiktoken.get_encoding("o200k_base")
    benchmark = json.loads((ROOT / "data/synthetic_benchmark.json").read_text(encoding="utf-8"))
    selected = json.loads((PHASE_A / "synthetic_300.json").read_text(encoding="utf-8"))
    support = json.loads((PHASE_A / "gold_support.json").read_text(encoding="utf-8"))
    support_by_index = {row["source_index"]: row for row in support}
    if len(selected) != 300 or len(benchmark) != 13_425 or len(support) != 13_425:
        raise RuntimeError("reviewed source row counts changed")
    ood_rows = make_ood_rows(benchmark, support)
    if len({row["entity"] for row in ood_rows}) != 20 or {row["entity"] for row in ood_rows} & {row["entity"] for row in selected}:
        raise RuntimeError("OOD entity reserve overlaps the 300-row sample")

    ds = load_dataset("ai-hyz/MemoryAgentBench", split="Conflict_Resolution", revision=DATASET_REVISION)
    fact_rows = [row for row in ds if row["metadata"].get("source") == "factconsolidation_sh_262k"]
    if len(fact_rows) != 1:
        raise RuntimeError("pinned benchmark corpus row not found uniquely")
    corpus = parse_facts(fact_rows[0]["context"])
    corpus_texts = [fact["text"] for fact in corpus]
    corpus_ids = [fact["fact_idx"] for fact in corpus]
    corpus_bm25 = BM25Okapi([toks(text) for text in corpus_texts])
    direct_system, fc_user_template = project_constants()

    requests: list[dict[str, Any]] = []
    for dataset_name, rows in (("synthetic300", selected), ("ood60", ood_rows)):
        for index, item in enumerate(rows):
            question = item.get("question_text", "")
            if dataset_name == "ood60":
                question = item["question_text"]
            source_audit = support_by_index[item["source_index"]]
            issue_codes = item.get("source_gold_issue_codes", source_audit["issue_codes"])
            flagged = item.get("source_gold_flagged", source_audit["status"] != "verified")
            scores = corpus_bm25.get_scores(toks(question))
            positions = np.argsort(scores)[::-1][:PROPOSED_K]
            retrieved = [{"fact_idx": corpus_ids[pos], "text": corpus_texts[pos]} for pos in positions]
            for arm in ("aletheia", "direct", "cot"):
                payload = make_prompt_request(arm, question, retrieved)
                requests.append({
                    "call_id": f"{dataset_name}-{index+1:03d}-{arm}",
                    "arm": arm,
                    "dataset": dataset_name,
                    "question_id": item.get("question_id", item.get("ood_id")),
                    "source_index": item.get("source_index"),
                    "entity": item["entity"],
                    "intent": item["intent"],
                    "question": question,
                    "source_gold_flagged": bool(flagged),
                    "source_gold_issue_codes": issue_codes,
                    "source_support_status": item.get("source_support_status", source_audit["status"]),
                    "k": PROPOSED_K,
                    **payload,
                })

    fc_rows: dict[str, Any] = {}
    for row in ds:
        source = row["metadata"].get("source", "")
        if source.startswith("factconsolidation_sh_"):
            fc_rows[source] = row
    fc_user_rows = json.loads((PHASE_A / "factconsolidation_manifest.json").read_text(encoding="utf-8"))
    for source in ("factconsolidation_sh_6k", "factconsolidation_sh_32k", "factconsolidation_sh_64k", "factconsolidation_sh_262k"):
        if source not in fc_rows:
            raise RuntimeError(f"pinned dataset is missing {source}")
        context_facts = parse_facts(fc_rows[source]["context"])
        texts = [fact["text"] for fact in context_facts]
        ids = [fact["fact_idx"] for fact in context_facts]
        index = BM25Okapi([toks(text) for text in texts])
        pairs = fc_user_rows[source]["ordered_question_answer_pairs"]
        if len(pairs) != 100:
            raise RuntimeError(f"{source} must contribute 100 calls")
        for item in pairs:
            q = item["question"]
            positions = np.argsort(index.get_scores(toks(q)))[::-1][:10]
            retrieved = [{"fact_idx": ids[pos], "text": texts[pos]} for pos in positions]
            pool = "\n".join(f"{fact['fact_idx']}. {fact['text']}." for fact in retrieved)
            user = f"[Knowledge Pool]\n{pool}\n\n{fc_user_template.format(question=q)}"
            requests.append({
                "call_id": f"{source}-{item['source_index']:03d}-factconsolidation",
                "arm": "factconsolidation",
                "dataset": source,
                "question_id": f"{source}-q{item['source_index']+1:03d}",
                "source_index": item["source_index"],
                "question": q,
                "k": 10,
                "model": "gpt-4o-mini",
                "messages": [{"role": "system", "content": direct_system}, {"role": "user", "content": user}],
                "max_completion_tokens": CAPS["factconsolidation"],
                "temperature": 0.0,
            })

    if len(requests) != 1_480 or Counter(row["arm"] for row in requests) != FULL_COUNTS:
        raise RuntimeError(f"campaign request count mismatch: {len(requests)} / {Counter(r['arm'] for r in requests)}")
    for row in requests:
        row["local_input_tokens"] = request_local_tokens(row, enc)
    # Balanced smoke: four revised, four direct, four CoT, three FC.
    smoke: list[dict[str, Any]] = []
    for arm, wanted in (("aletheia", 4), ("direct", 4), ("cot", 4)):
        choices = [row for row in requests if row["arm"] == arm and row["dataset"] == "synthetic300"]
        by_intent: dict[str, dict[str, Any]] = {}
        for row in choices:
            by_intent.setdefault(row["intent"], row)
        picked = list(by_intent.values())
        for row in choices:
            if len(picked) >= wanted:
                break
            if row not in picked:
                picked.append(row)
        smoke.extend(picked[:wanted])
    fc_smoke = [row for row in requests if row["arm"] == "factconsolidation"]
    smoke.extend(fc_smoke[::100][:3])
    if len(smoke) != 15:
        raise RuntimeError(f"smoke must contain exactly 15 unique calls, got {len(smoke)}")

    # Avoid spending near the limit even if local token accounting misses
    # provider framing/schema overhead: double local input plus 1,000 tokens
    # per call and all maximum outputs must stay below $0.04.
    smoke_bound_input = sum(2 * row["local_input_tokens"] + 1_000 for row in smoke)
    smoke_bound_usd = smoke_bound_input * INPUT_USD_PER_M / 1_000_000 + sum(CAPS[row["arm"]] for row in smoke) * OUTPUT_USD_PER_M / 1_000_000
    if smoke_bound_usd >= 0.04 or smoke_bound_usd >= CEILING_USD:
        raise RuntimeError(f"smoke cost preflight failed: conservative upper bound ${smoke_bound_usd:.6f}")

    OUT.mkdir(parents=True)
    LOG.mkdir(parents=True)
    (LOG / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
    (OUT / "ood_60.json").write_text(json.dumps(ood_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "control_prompts.md").write_text(
        "# Fixed control prompts\n\n"
        "All arms use `gpt-4o-mini`, temperature 0, K=80 on synthetic/OOD, K=10 on FactConsolidation. Completion caps: Aletheia and CoT 1,024 tokens; direct and FactConsolidation 512 tokens.\n\n"
        "## Direct-answer control\n\n"
        "System: `" + __import__("scripts.lib.prof_feedback", fromlist=["DIRECT_CONTROL_SYSTEM"]).DIRECT_CONTROL_SYSTEM + "`\n\n"
        "User: `[Knowledge Pool]\\n{pool}\\n\\nQuestion: {question}\\nAnswer:`\n\n"
        "## Chain-of-thought control\n\n"
        "System: `" + __import__("scripts.lib.prof_feedback", fromlist=["CHAIN_OF_THOUGHT_SYSTEM"]).CHAIN_OF_THOUGHT_SYSTEM + "`\n\n"
        "User: `[Knowledge Pool]\\n{pool}\\n\\nQuestion: {question}`\n\n"
        "## FactConsolidation BM25 baseline\n\n"
        "This arm uses the repository's original BM25 system message and query template, with temperature 0 for the matched re-run. Retrieval is top-10.\n\n"
        "System: `" + direct_system + "`\n\n"
        "User template:\n\n```text\n[Knowledge Pool]\n{pool}\n\n" + fc_user_template + "\n```\n\n"
        "## OOD construction note\n\n"
        "Sixty separate OOD questions use intent-specific alternate constructions over 20 reserved entities (five constructions per intent family). Each row retains its source question, entity, intent, gold answer, and audit flags as metadata; no released benchmark row is modified. This is a same-corpus, same-domain linguistic-shift set, not a new-domain test.\n",
        encoding="utf-8",
    )
    with (OUT / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in requests:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "run_id": RUN_ID,
        "prepared_at_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "model": "gpt-4o-mini",
        "proposed_k": PROPOSED_K,
        "completion_caps": CAPS,
        "temperature": 0,
        "full_call_count": len(requests),
        "calls_by_arm": dict(Counter(row["arm"] for row in requests)),
        "smoke_call_count": len(smoke),
        "smoke_ids": [row["call_id"] for row in smoke],
        "smoke_preflight_worst_case_usd": round(smoke_bound_usd, 8),
        "local_input_tokens_full_manifest": sum(row["local_input_tokens"] for row in requests),
        "ood_questions": len(ood_rows),
        "ood_gold_flags_retained": sum(row["source_gold_flagged"] for row in ood_rows),
        "synthetic_300_gold_flags_retained": sum(support_by_index[row["source_index"]]["status"] != "verified" for row in selected),
        "first_60_flagged_rows_retained": sum(support_by_index[index]["status"] != "verified" for index in range(60)),
        "full_execution_enabled": True,
        "api_calls_made": 0,
    }
    (OUT / "manifest_metadata.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUT / ".gitignore").write_text("\n", encoding="utf-8")
    (OUT / "_smoke_plan.json").write_text(json.dumps(smoke, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"manifest": manifest, "smoke": smoke}


def run_smoke(smoke: list[dict[str, Any]]) -> dict[str, Any]:
    load_env_file()
    os.environ["ALETHEIA_DISABLE_TRACING"] = "1"
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is unavailable; smoke was not sent")
    from openai import OpenAI
    client = OpenAI(max_retries=0, timeout=90)
    results: list[dict[str, Any]] = []
    cumulative_actual = 0.0
    for index, row in enumerate(smoke):
        remaining = smoke[index:]
        # Require actual spent plus a 2x local-input / 1,000-token framing
        # reserve and all remaining output caps to remain below the hard cap.
        remaining_input_guard = sum(2 * item["local_input_tokens"] + 1_000 for item in remaining)
        remaining_upper = cumulative_actual + remaining_input_guard * INPUT_USD_PER_M / 1_000_000 + sum(CAPS[item["arm"]] for item in remaining) * OUTPUT_USD_PER_M / 1_000_000
        if remaining_upper >= CEILING_USD:
            raise RuntimeError(f"hard smoke guard stopped before call {index+1}: ${remaining_upper:.6f}")
        # Exactly one SDK request, SDK retries disabled.
        try:
            response = client.chat.completions.create(**api_request(row))
        except Exception as exc:
            results.append({"call_id": row["call_id"], "arm": row["arm"], "error_type": type(exc).__name__, "error": str(exc)[:600]})
            (OUT / "smoke_usage.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
            break
        usage = response.usage
        if usage is None or usage.prompt_tokens is None or usage.completion_tokens is None:
            raise RuntimeError(f"usage missing for call {row['call_id']}; refusing to continue")
        actual_usd = usage.prompt_tokens * INPUT_USD_PER_M / 1_000_000 + usage.completion_tokens * OUTPUT_USD_PER_M / 1_000_000
        cumulative_actual += actual_usd
        output_text = response.choices[0].message.content or ""
        record = {
            "call_id": row["call_id"], "arm": row["arm"], "dataset": row["dataset"],
            "question_id": row["question_id"], "k": row["k"],
            "request_input_tokens_local": row["local_input_tokens"],
            "usage_prompt_tokens": usage.prompt_tokens,
            "usage_completion_tokens": usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
            "request_id": getattr(response, "_request_id", None),
            "estimated_actual_cost_usd": round(actual_usd, 10),
            "response_text": output_text,
        }
        results.append(record)
        (OUT / "smoke_usage.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: record[key] for key in ("call_id", "arm", "usage_prompt_tokens", "usage_completion_tokens", "finish_reason", "estimated_actual_cost_usd")}))
        if cumulative_actual >= CEILING_USD:
            break
    return {"results": results, "cumulative_actual": cumulative_actual}


def configure_full_manifest() -> list[dict[str, Any]]:
    """Apply the reviewed per-arm caps to the prepared manifest in place."""
    manifest_path = OUT / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 1480:
        raise RuntimeError(f"expected 1,480 full-manifest calls; found {len(rows)}")
    for row in rows:
        cap = CAPS[row["arm"]]
        row["max_completion_tokens"] = cap
    tmp = manifest_path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    tmp.replace(manifest_path)
    metadata_path = OUT / "manifest_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["completion_caps"] = CAPS
    metadata["full_execution_enabled"] = True
    metadata["api_calls_made"] = len(json.loads((OUT / "smoke_usage.json").read_text(encoding="utf-8")))
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return rows


def save_json_atomic(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def run_full() -> dict[str, Any]:
    """Run/resume full manifest calls; never repeat IDs in the durable ledger."""
    load_env_file()
    os.environ["ALETHEIA_DISABLE_TRACING"] = "1"
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is unavailable; no campaign call was sent")
    from openai import OpenAI

    rows = configure_full_manifest()
    ledger_path = LOG / "full_calls.jsonl"
    run_log_path = LOG / "full_run.log"
    checkpoint_path = LOG / "full_checkpoint.json"
    LOG.mkdir(parents=True, exist_ok=True)
    ledger_path.touch(exist_ok=True)
    completed: dict[str, dict[str, Any]] = {}
    for line_no, line in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        call_id = record["call_id"]
        if call_id in completed:
            raise RuntimeError(f"duplicate call ID in durable ledger at line {line_no}: {call_id}")
        completed[call_id] = record
    smoke_path = OUT / "smoke_usage.json"
    smoke_rows = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke_plan = {row["call_id"]: row for row in json.loads((OUT / "_smoke_plan.json").read_text(encoding="utf-8"))}
    smoke_results = {row["call_id"]: row for row in smoke_rows if row.get("usage_prompt_tokens") is not None}
    smoke_spend = sum(
        row["usage_prompt_tokens"] * INPUT_USD_PER_M / 1_000_000
        + row["usage_completion_tokens"] * OUTPUT_USD_PER_M / 1_000_000
        for row in smoke_rows if row.get("usage_prompt_tokens") is not None
    )
    # Reuse exact-payload smoke completions on a fresh or partial full ledger.
    # The smoke charge is already part of smoke_spend, so these ledger entries
    # carry zero incremental full-run cost and are never billed a second time.
    for row in rows:
        prior_request = smoke_plan.get(row["call_id"])
        prior_result = smoke_results.get(row["call_id"])
        if row["call_id"] in completed or prior_request is None or prior_result is None:
            continue
        if api_request(prior_request) != api_request(row):
            continue
        record = {
            "call_id": row["call_id"], "arm": row["arm"], "dataset": row["dataset"],
            "question_id": row["question_id"], "source_index": row.get("source_index"),
            "intent": row.get("intent"), "source_gold_flagged": row.get("source_gold_flagged"),
            "source_gold_issue_codes": row.get("source_gold_issue_codes"),
            "max_completion_tokens": row["max_completion_tokens"],
            "usage_prompt_tokens": prior_result["usage_prompt_tokens"],
            "usage_completion_tokens": prior_result["usage_completion_tokens"],
            "finish_reason": prior_result.get("finish_reason"),
            "status": "truncated_failure" if prior_result.get("finish_reason") == "length" else "reused_from_smoke",
            "request_id": prior_result.get("request_id"),
            "actual_cost_usd": 0.0,
            "smoke_cost_usd": prior_result.get("estimated_actual_cost_usd"),
            "reused_from_smoke": True,
            "response_text": prior_result.get("response_text", ""),
        }
        with ledger_path.open("a", encoding="utf-8") as ledger, run_log_path.open("a", encoding="utf-8") as runlog:
            serialized = json.dumps(record, ensure_ascii=False)
            ledger.write(serialized + "\n")
            runlog.write(serialized + "\n")
        completed[row["call_id"]] = record
        save_json_atomic(checkpoint_path, {
            "completed_or_terminal_calls": len(completed), "expected_calls": len(rows),
            "smoke_baseline_usage_cost_usd": round(smoke_spend, 10),
            "full_run_usage_cost_usd": 0.0,
            "cumulative_usage_cost_usd": round(smoke_spend, 10),
            "reused_smoke_calls": sum(r.get("reused_from_smoke") is True for r in completed.values()),
            "halt_reason": None, "last_call_id": row["call_id"],
        })
    full_run_spend = sum(
        float(record.get("actual_cost_usd", 0.0)) for record in completed.values()
        if not record.get("reused_from_smoke")
    )
    spent = smoke_spend + full_run_spend
    if any(record.get("status") in {"usage_missing", "request_error"} for record in completed.values()):
        raise RuntimeError("ledger contains a call without reliable usage; reconcile provider spend before resuming")
    if spent > HARD_STOP_USD:
        raise RuntimeError(f"cumulative ledger spend is already above the $0.80 hard stop: ${spent:.6f}")

    client = OpenAI(max_retries=0, timeout=120)
    attempted = len(completed)
    with ledger_path.open("a", encoding="utf-8", buffering=1) as ledger, run_log_path.open("a", encoding="utf-8", buffering=1) as runlog:
        for row in rows:
            if row["call_id"] in completed:
                continue
            if spent > HARD_STOP_USD:
                break
            start = time.time()
            try:
                response = client.chat.completions.create(**api_request(row))
            except Exception as exc:
                # An exception can happen after provider acceptance. Do not
                # retry or continue without reliable usage-based accounting.
                record = {
                    "call_id": row["call_id"], "arm": row["arm"], "status": "request_error",
                    "error_type": type(exc).__name__, "error": str(exc)[:1000],
                    "usage_prompt_tokens": None, "usage_completion_tokens": None,
                    "actual_cost_usd": None, "request_started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                ledger.write(json.dumps(record, ensure_ascii=False) + "\n")
                runlog.write(json.dumps(record, ensure_ascii=False) + "\n")
                save_json_atomic(checkpoint_path, {"completed_or_terminal_calls": len(completed) + 1, "expected_calls": len(rows), "smoke_baseline_usage_cost_usd": round(smoke_spend, 10), "full_run_usage_cost_usd": round(full_run_spend, 10), "cumulative_usage_cost_usd": round(spent, 10), "halt_reason": "request_error_without_usage", "last_call_id": row["call_id"]})
                raise RuntimeError(f"halted at {row['call_id']} after request error; no retry was made") from exc
            usage = response.usage
            if usage is None or usage.prompt_tokens is None or usage.completion_tokens is None:
                record = {"call_id": row["call_id"], "arm": row["arm"], "status": "usage_missing", "finish_reason": response.choices[0].finish_reason, "request_id": getattr(response, "_request_id", None), "actual_cost_usd": None}
                ledger.write(json.dumps(record, ensure_ascii=False) + "\n")
                runlog.write(json.dumps(record, ensure_ascii=False) + "\n")
                save_json_atomic(checkpoint_path, {"completed_or_terminal_calls": len(completed) + 1, "expected_calls": len(rows), "smoke_baseline_usage_cost_usd": round(smoke_spend, 10), "full_run_usage_cost_usd": round(full_run_spend, 10), "cumulative_usage_cost_usd": round(spent, 10), "halt_reason": "usage_missing", "last_call_id": row["call_id"]})
                raise RuntimeError(f"halted at {row['call_id']}: provider usage fields missing")
            amount = usage.prompt_tokens * INPUT_USD_PER_M / 1_000_000 + usage.completion_tokens * OUTPUT_USD_PER_M / 1_000_000
            spent += amount
            full_run_spend += amount
            finish = response.choices[0].finish_reason
            status = "truncated_failure" if finish == "length" else "completed"
            record = {
                "call_id": row["call_id"], "arm": row["arm"], "dataset": row["dataset"],
                "question_id": row["question_id"], "source_index": row.get("source_index"),
                "intent": row.get("intent"), "source_gold_flagged": row.get("source_gold_flagged"),
                "source_gold_issue_codes": row.get("source_gold_issue_codes"),
                "max_completion_tokens": row["max_completion_tokens"],
                "usage_prompt_tokens": usage.prompt_tokens,
                "usage_completion_tokens": usage.completion_tokens,
                "finish_reason": finish, "status": status,
                "request_id": getattr(response, "_request_id", None),
                "actual_cost_usd": round(amount, 10), "cumulative_actual_cost_usd": round(full_run_spend, 10),
                "cumulative_campaign_cost_usd": round(spent, 10),
                "elapsed_seconds": round(time.time() - start, 3),
                "response_text": response.choices[0].message.content or "",
            }
            serialized = json.dumps(record, ensure_ascii=False)
            ledger.write(serialized + "\n")
            runlog.write(serialized + "\n")
            completed[row["call_id"]] = record
            attempted += 1
            done = len(completed)
            halt_reason = "actual_spend_above_0.80" if spent > HARD_STOP_USD else None
            save_json_atomic(checkpoint_path, {
                "completed_or_terminal_calls": done, "expected_calls": len(rows),
                "smoke_baseline_usage_cost_usd": round(smoke_spend, 10),
                "full_run_usage_cost_usd": round(full_run_spend, 10),
                "cumulative_usage_cost_usd": round(spent, 10),
                "reused_smoke_calls": sum(r.get("reused_from_smoke") is True for r in completed.values()),
                "truncated_failures": sum(r.get("status") == "truncated_failure" for r in completed.values()),
                "halt_reason": halt_reason, "last_call_id": row["call_id"],
                "updated_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })
            print(json.dumps({key: record[key] for key in ("call_id", "arm", "usage_prompt_tokens", "usage_completion_tokens", "finish_reason", "status", "actual_cost_usd", "cumulative_actual_cost_usd")}), flush=True)
            if spent > HARD_STOP_USD:
                print(f"HARD STOP: usage-based cumulative spend ${spent:.6f} exceeded $0.80; checkpoint saved.", flush=True)
                break
    terminal = len(completed)
    return {
        "terminal_calls": terminal, "expected_calls": len(rows), "remaining_calls": len(rows) - terminal,
        "cumulative_usage_cost_usd": round(spent, 6),
        "smoke_baseline_usage_cost_usd": round(smoke_spend, 6),
        "full_run_usage_cost_usd": round(full_run_spend, 6),
        "truncated_failures": sum(record.get("status") == "truncated_failure" for record in completed.values()),
        "request_errors": sum(record.get("status") in {"request_error", "usage_missing"} for record in completed.values()),
        "halt_reason": "actual_spend_above_0.80" if spent > HARD_STOP_USD else ("complete" if terminal == len(rows) else "paused"),
        "ledger": str(ledger_path), "checkpoint": str(checkpoint_path),
    }


def project(smoke: list[dict[str, Any]], usage_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import defaultdict
    observed = {row["call_id"]: row for row in usage_rows}
    deltas: dict[str, list[float]] = defaultdict(list)
    outputs: dict[str, list[int]] = defaultdict(list)
    actual_input: dict[str, list[int]] = defaultdict(list)
    for row in smoke:
        result = observed.get(row["call_id"])
        if not result or "usage_prompt_tokens" not in result:
            continue
        deltas[row["arm"]].append(result["usage_prompt_tokens"] - row["local_input_tokens"])
        # Call 2 hit the previous Aletheia 512 cap and was cut mid-JSON.
        # Project it at the newly authorized 1,024 cap rather than treating
        # the truncated 512-token response as a typical complete answer.
        output_tokens = result["usage_completion_tokens"]
        if result.get("finish_reason") == "length":
            output_tokens = CAPS[row["arm"]]
        outputs[row["arm"]].append(output_tokens)
        actual_input[row["arm"]].append(result["usage_prompt_tokens"])
    if set(outputs) != set(FULL_COUNTS):
        raise RuntimeError("smoke must include measured responses from all four campaign arms before projecting")

    manifest = [json.loads(line) for line in (OUT / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    calibrated_input = sum(row["local_input_tokens"] + statistics.mean(deltas[row["arm"]]) for row in manifest)
    projected_output = sum(FULL_COUNTS[arm] * statistics.mean(outputs[arm]) for arm in FULL_COUNTS)
    input_avg = calibrated_input / len(manifest)
    total_usd = calibrated_input * INPUT_USD_PER_M / 1_000_000 + projected_output * OUTPUT_USD_PER_M / 1_000_000
    with_reserve = total_usd * (1 + RESERVE)
    calibrated_input_cost = calibrated_input * INPUT_USD_PER_M / 1_000_000

    # Re-count full synthetic requests at each candidate K. FactConsolidation
    # remains at its baseline K=10. Reuse arm-specific API framing calibration.
    # Preserve offline-only dataset access during cost calibration.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    max_k = PROPOSED_K if with_reserve <= FULL_CEILING_USD else None
    projected_by_k = {str(PROPOSED_K): {
        "input_tokens": round(calibrated_input),
        "projected_cost_usd": round(total_usd, 6),
        "with_20pct_reserve_usd": round(with_reserve, 6),
    }}
    result = {
        "model": "gpt-4o-mini",
        "measured_smoke_calls": len(usage_rows),
        "smoke_actual_estimated_cost_usd": round(sum(row.get("estimated_actual_cost_usd", 0) for row in usage_rows), 6),
        "smoke_total_prompt_tokens": sum(row.get("usage_prompt_tokens", 0) for row in usage_rows),
        "smoke_total_completion_tokens": sum(row.get("usage_completion_tokens", 0) for row in usage_rows),
        "smoke_mean_prompt_tokens_per_call": round(sum(row.get("usage_prompt_tokens", 0) for row in usage_rows) / len(usage_rows), 2),
        "smoke_mean_completion_tokens_per_call": round(sum(row.get("usage_completion_tokens", 0) for row in usage_rows) / len(usage_rows), 2),
        "proposed_k": PROPOSED_K,
        "full_calls": len(manifest),
        "input_tokens_projected_calibrated": round(calibrated_input),
        "per_call_input_average_projected": round(input_avg, 2),
        "completion_tokens_projected_from_smoke_means": round(projected_output),
        "projected_cost_at_k80_usd": round(total_usd, 6),
        "projected_cost_with_20pct_reserve_usd": round(with_reserve, 6),
        "max_k_at_or_below_1_00_with_reserve": max_k,
        "k_search_range": "K=1..80; K=80 was checked and fits, so it is the largest K in the evaluated range",
        "output_caps_by_arm": CAPS,
        "worst_case_output_cost_usd": round(sum(FULL_COUNTS[arm] * CAPS[arm] for arm in FULL_COUNTS) * OUTPUT_USD_PER_M / 1_000_000, 6),
        "k_projection": projected_by_k,
        "projection_method": "full manifest local o200k_base text/schema counts plus per-arm mean difference between smoke API usage.prompt_tokens and local counts; completion usage mean per arm; current list rates; no cached-input discount",
        "ood_linguistic_scope": "intent-specific same-corpus linguistic rewrites; not a new-domain test",
    }
    (OUT / "cost_projection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Smoke-based campaign cost projection", "",
        f"- Smoke calls: {len(usage_rows)}/15; measured spend at current list rates: ${result['smoke_actual_estimated_cost_usd']:.6f} (limit: $0.05).",
        f"- Smoke input: {result['smoke_total_prompt_tokens']:,} tokens, {result['smoke_mean_prompt_tokens_per_call']:.2f} per call on average.",
        f"- Calibrated full-campaign input at K={PROPOSED_K}: {result['input_tokens_projected_calibrated']:,} tokens, {result['per_call_input_average_projected']:.2f} per call on average.",
        f"- Smoke completions: {result['smoke_total_completion_tokens']:,} tokens, {result['smoke_mean_completion_tokens_per_call']:.2f} per call on average.",
        f"- Full-campaign projection from measured per-arm smoke means: ${result['projected_cost_at_k80_usd']:.6f}; with 20% reserve: ${result['projected_cost_with_20pct_reserve_usd']:.6f}.",
        f"- Largest K in evaluated range 1–80 under the $1.00 ceiling with reserve: **K={max_k}**.",
        f"- All completions at their per-arm caps would cost ${result['worst_case_output_cost_usd']:.6f}; with calibrated input the cap-bound total is ${calibrated_input_cost + result['worst_case_output_cost_usd']:.6f}; adding 20% uncertainty reserve gives ${((calibrated_input_cost + result['worst_case_output_cost_usd']) * 1.2):.6f}.",
        "", "## Per-arm smoke means", "",
    ]
    for arm in FULL_COUNTS:
        arm_results = [row for row in usage_rows if row.get("arm") == arm]
        lines.append(
            f"- `{arm}` ({len(arm_results)} calls): mean input "
            f"{statistics.mean(row['usage_prompt_tokens'] for row in arm_results):.2f}; mean completion "
            f"{statistics.mean(row['usage_completion_tokens'] for row in arm_results):.2f} tokens."
        )
    lines += [
        "", "## Per-call API usage", "",
        "See `smoke_usage.csv` and `smoke_usage.json` for all 15 actual `usage.prompt_tokens` / `usage.completion_tokens` pairs, finish reasons, request IDs, and estimated per-call charges.",
        "", "## Method and limits", "",
        "Input projection sums the 1,480-request manifest's local `o200k_base` message/schema counts and calibrates each arm by its mean difference between smoke-local counts and actual API `usage.prompt_tokens`. Output projection multiplies each arm's measured completion-token mean by its full call count, substituting the new arm cap for the one prior Aletheia response truncated at 512. GPT-4o mini list rates are $0.15/M input and $0.60/M output; no cached-input discount is assumed. The point projection and 20% reserve are compared with the $1.00 execution gate.",
        "The smoke sampled 4 Aletheia, 4 direct, 4 CoT, and 3 FactConsolidation calls; means are noisy. One Aletheia response hit the prior 512-token cap (`finish_reason=length`) and was invalid JSON; for the revised projection it is conservatively imputed at the new 1,024-token Aletheia cap. Every future `finish_reason=length` is a logged terminal failure with no retry. Caps are Aletheia/CoT 1,024 and direct/FactConsolidation 512.",
        "The OOD items are same-domain linguistic rewrites over 20 reserved entities; they do not establish new-domain generalization. No full-campaign calls were made.",
        "",
    ]
    (OUT / "cost_projection.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="prepare if needed, then send only its 15-call smoke plan")
    parser.add_argument("--project-only", action="store_true", help="recalculate projection from saved smoke usage; sends no API requests")
    parser.add_argument("--full", action="store_true", help="run/resume the approved 1,480-call manifest with a usage-based $0.80 hard stop")
    args = parser.parse_args()
    if args.project_only:
        smoke = json.loads((OUT / "_smoke_plan.json").read_text(encoding="utf-8"))
        usage = json.loads((OUT / "smoke_usage.json").read_text(encoding="utf-8"))
        projection = project(smoke, usage)
        print(json.dumps(projection, indent=2))
        return
    if args.full:
        projection_path = OUT / "cost_projection.json"
        if not projection_path.exists():
            raise RuntimeError("no verified smoke projection is saved; refusing full campaign")
        projection = json.loads(projection_path.read_text(encoding="utf-8"))
        if projection["projected_cost_at_k80_usd"] >= 1.00:
            raise RuntimeError("smoke-based projection is not below $1.00; refusing full campaign")
        result = run_full()
        print(json.dumps(result, indent=2))
        return
    if OUT.exists():
        metadata = json.loads((OUT / "manifest_metadata.json").read_text(encoding="utf-8"))
        smoke = json.loads((OUT / "_smoke_plan.json").read_text(encoding="utf-8"))
        prepared = {"manifest": metadata, "smoke": smoke}
    else:
        prepared = prepare()
    if not args.smoke:
        print(json.dumps(prepared["manifest"], indent=2))
        return
    smoke_result = run_smoke(prepared["smoke"])
    if len(smoke_result["results"]) == 15 and all("usage_prompt_tokens" in row for row in smoke_result["results"]):
        projection = project(prepared["smoke"], smoke_result["results"])
        print(json.dumps(projection, indent=2))
    else:
        print(json.dumps({"smoke_calls_with_usage": len([r for r in smoke_result["results"] if "usage_prompt_tokens" in r]), "stopped_after_error": True, "cost_so_far_usd": round(smoke_result["cumulative_actual"], 6)}, indent=2))


if __name__ == "__main__":
    main()
