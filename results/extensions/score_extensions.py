"""Offline scoring for saved extension experiment outputs; makes no API calls."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXT = ROOT / "results/extensions"
BASE = ROOT / "results/prof_feedback/pf-20261006-hybrid300-final-01"
RRFDIR = EXT / "experiment_1_rrf_all300"
REACTDIR = EXT / "experiment_2_react"
INTENT_NAMES = {"historical": "temporal", "aggregation": "aggregation", "boolean": "boolean"}


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(bool(r["correct"]) for r in rows)
    return {"n": n, "correct": correct, "accuracy": correct / n if n else None}


def groups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {INTENT_NAMES[intent]: summary([r for r in rows if r["intent"] == intent])
           for intent in ("historical", "aggregation", "boolean")}
    out["overall"] = summary(rows)
    return out


def check_complete(path: Path, expected: int, key: str) -> list[dict[str, Any]]:
    rows = jsonl(path)
    found = [r for r in rows if r.get(key) is not None]
    if len(found) != expected or len({r[key] for r in found}) != expected:
        raise RuntimeError(f"No complete {expected}-question score: {path} contains {len(found)} unique outcomes")
    return rows


def score_rrf() -> None:
    sys.path.insert(0, str(ROOT))
    from scripts.lib.prof_feedback import QueryPlan
    from scripts.lib.prof_feedback_final import execute_final_plan
    from scripts.lib.evaluation_scorer import score_answer

    calls = jsonl(RRFDIR / "calls.jsonl")
    requests = jsonl(RRFDIR / "requests.jsonl")
    if len(calls) != 300 or len(requests) != 300:
        raise RuntimeError(f"RRF was not a complete 300-call run ({len(calls)} calls); full metrics withheld")
    call_by_id = {r["call_id"]: r for r in calls}
    if len(call_by_id) != 300:
        raise RuntimeError("Duplicate RRF call IDs; full metrics withheld")
    baseline = json.loads((BASE / "scored_calls.json").read_text(encoding="utf-8"))
    baseline_by = {(r["source_index"], r["arm"]): r for r in baseline}
    frozen_counts = json.loads((EXT / "entity_history_counts.json").read_text(encoding="utf-8"))
    entity_counts = frozen_counts["reliable_entity_counts"]
    property_counts = {tuple(key.split("\t", 1)): value for key, value in frozen_counts["reliable_entity_property_counts"].items()}
    scored = []
    for row in requests:
        raw = call_by_id[row["call_id"]]
        prediction = None
        answered = False
        complete = False
        parse_ok = False
        if raw.get("status") == "completed":
            try:
                obj = json.loads(raw.get("response_text", ""))
                plan = QueryPlan.from_object(obj["query_plan"])
                _, result, complete = execute_final_plan(row["question"], plan, row["retrieved_items"], entity_counts, property_counts)
                prediction = result.get("answer")
                answered = result.get("status") == "answered"
                if plan.intent != "boolean":
                    answered = bool(plan.supported and complete and answered)
                else:
                    answered = bool(plan.entity and plan.boolean_target and answered)
                parse_ok = True
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                prediction = None
        correct = bool(score_answer(row["intent"], prediction, row["ground_truth_answer"]))
        original = baseline_by[(row["source_index"], "aletheia")]
        scored.append({
            "call_id": row["call_id"], "source_index": row["source_index"], "question_id": row["question_id"],
            "intent": row["intent"], "question": row["question"], "gold": row["ground_truth_answer"],
            "flagged": row["source_gold_flagged"], "answered": answered, "complete_history": complete,
            "parse_ok": parse_ok, "status": raw.get("status"), "prediction": prediction, "correct": correct,
            "baseline_prediction": original["prediction"], "baseline_correct": bool(original["correct"]),
        })
    if len(scored) != 300:
        raise RuntimeError("RRF scoring did not produce exactly 300 rows")
    flips = {
        "baseline_right_rrf_wrong": [r for r in scored if r["baseline_correct"] and not r["correct"]],
        "baseline_wrong_rrf_right": [r for r in scored if not r["baseline_correct"] and r["correct"]],
    }
    baseline_arms = {
        arm: groups([r for r in baseline if r["arm"] == arm])
        for arm in ("aletheia", "direct", "cot")
    }
    rrf_groups = groups(scored)
    hybrids = []
    for row in scored:
        for fallback in ("direct", "cot"):
            use_rrf = bool(row["answered"])
            fallback_row = baseline_by[(row["source_index"], fallback)]
            chosen_correct = row["correct"] if use_rrf else bool(fallback_row["correct"])
            chosen_prediction = row["prediction"] if use_rrf else fallback_row["prediction"]
            hybrids.append({"source_index": row["source_index"], "intent": row["intent"],
                            "flagged": row["flagged"], "fallback": fallback,
                            "source": "aletheia_rrf" if use_rrf else fallback,
                            "deterministic": use_rrf, "prediction": chosen_prediction,
                            "gold": row["gold"], "correct": bool(chosen_correct)})
    hybrid_groups = {fallback: groups([r for r in hybrids if r["fallback"] == fallback])
                     for fallback in ("direct", "cot")}
    call_spend = sum(float(r.get("actual_cost_usd") or 0) for r in calls)
    report = {
        "status": "complete", "n": 300, "audit_flags_retained": sum(r["flagged"] for r in scored),
        "baseline_arms": baseline_arms, "aletheia_rrf": rrf_groups,
        "hybrid_rrf_direct_fallback": hybrid_groups["direct"],
        "hybrid_rrf_cot_fallback": hybrid_groups["cot"],
        "hybrid_deterministic_share": {name: sum(r["deterministic"] for r in hybrids if r["fallback"] == name) / 300
                                       for name in ("direct", "cot")},
        "flips": {name: [{"source_index": r["source_index"], "intent": r["intent"],
                           "question": r["question"], "gold": r["gold"],
                           "baseline_prediction": r["baseline_prediction"], "rrf_prediction": r["prediction"]}
                          for r in values] for name, values in flips.items()},
        "calls": 300, "prompt_tokens": sum(int(r.get("usage_prompt_tokens") or 0) for r in calls),
        "completion_tokens": sum(int(r.get("usage_completion_tokens") or 0) for r in calls),
        "spend_usd_from_usage": round(call_spend, 10),
        "truncated_failures": sum(r.get("status") == "truncated_failure" for r in calls),
        "request_errors": sum(r.get("status") != "completed" and r.get("status") != "truncated_failure" for r in calls),
    }
    (RRFDIR / "scored_calls.json").write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RRFDIR / "hybrid_scored_calls.json").write_text(json.dumps(hybrids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RRFDIR / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def score_react() -> None:
    from scripts.lib.evaluation_scorer import score_answer
    runs_path = REACTDIR / "question_runs.jsonl"
    runs = check_complete(runs_path, 300, "source_index")
    if len(runs) != 300:
        raise RuntimeError("ReAct question outcomes include extra records")
    baseline = json.loads((BASE / "scored_calls.json").read_text(encoding="utf-8"))
    base_by = {(r["source_index"], r["arm"]): r for r in baseline}
    scored = []
    for row in runs:
        answer = row.get("answer") if row.get("status") == "completed" else None
        correct = bool(score_answer(row["intent"], answer, row["gold"]))
        scored.append({"source_index": row["source_index"], "question_id": row["question_id"],
                       "intent": row["intent"], "question": row["question"], "gold": row["gold"],
                       "flagged": row["flagged"], "answer": answer, "correct": correct,
                       "answered": answer is not None, "status": row["status"],
                       "search_steps": row["search_steps"], "model_api_calls": row["model_api_calls"],
                       "prompt_tokens": row["prompt_tokens"], "completion_tokens": row["completion_tokens"],
                       "actual_cost_usd": row["actual_cost_usd"]})
    calls = jsonl(REACTDIR / "calls.jsonl")
    report = {
        "status": "complete", "n": 300, "audit_flags_retained": sum(r["flagged"] for r in scored),
        "react": groups(scored),
        "comparators": {arm: groups([r for r in baseline if r["arm"] == arm])
                        for arm in ("aletheia", "direct", "cot")},
        "prompt_tokens": sum(int(r.get("usage_prompt_tokens") or 0) for r in calls),
        "completion_tokens": sum(int(r.get("usage_completion_tokens") or 0) for r in calls),
        "model_api_calls": len(calls),
        "search_steps": sum(int(r.get("search_steps") or 0) for r in scored),
        "spend_usd_from_usage": round(sum(float(r.get("actual_cost_usd") or 0) for r in calls), 10),
        "truncated_question_failures": [r["source_index"] for r in scored if r["status"] == "truncated_failure"],
        "question_errors": [{"source_index": r["source_index"], "question": r["question"], "status": r["status"]}
                            for r in scored if r["status"] != "completed"],
        "per_question_costs_usd": [{"source_index": r["source_index"], "question_number": i + 1,
                                    "cost_usd": r["actual_cost_usd"], "api_calls": r["model_api_calls"]}
                                   for i, r in enumerate(scored)],
    }
    (REACTDIR / "scored_calls.json").write_text(json.dumps(scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (REACTDIR / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("rrf", "react"))
    args = parser.parse_args()
    if args.experiment == "rrf":
        score_rrf()
    else:
        score_react()


if __name__ == "__main__":
    main()
