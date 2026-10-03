"""Run the synthetic benchmark against the adaptive router pipeline.

By default, evaluates the first 60 questions (20 historical, 20 aggregation,
and 20 boolean), matching the paper. Use --n 13425 or --all for the full
released dataset. Output defaults to a separate rerun file so the committed
paper results are preserved.

Usage:
    python scripts/experiments/run_synthetic_benchmark.py
    python scripts/experiments/run_synthetic_benchmark.py --n 13425
    python scripts/experiments/run_synthetic_benchmark.py --all
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

def parse_facts(ctx: str) -> list[tuple[int, str]]:
    pat = re.compile(r"(\d+)\.\s")
    matches = list(pat.finditer(ctx))
    facts: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        idx = int(m.group(1))
        s = m.end()
        e = matches[i + 1].start() if i + 1 < len(matches) else len(ctx)
        text = ctx[s:e].strip().rstrip(".")
        facts.append((idx, text))
    return facts


def normalize(s: str) -> str:
    """Lowercase, strip, collapse whitespace for soft matching."""
    return re.sub(r"\s+", " ", s.strip().lower())


def main():
    parser = argparse.ArgumentParser(description="Evaluate the adaptive router on the synthetic benchmark.")
    parser.add_argument("--n", type=int, default=60,
                        help="Number of leading questions to evaluate (default: 60; 0 means all).")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate the full released benchmark (same as --n 0).")
    parser.add_argument("--output", default=None,
                        help="Results path (default: results/poc_results/synthetic_benchmark_results_rerun.json).")
    args = parser.parse_args()
    if args.n < 0:
        parser.error("--n must be zero or a positive integer")

    output_path = Path(args.output) if args.output else ROOT_DIR / "results" / "poc_results" / "synthetic_benchmark_results_rerun.json"
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    requested_n = "all" if args.all or args.n == 0 else str(args.n)
    print(f"WARNING: n={requested_n}; output={output_path}; this run makes OpenAI API calls.")

    # These imports initialize local models and tracing, so keep them after --help parsing.
    from datasets import load_dataset
    from rank_bm25 import BM25Okapi
    from scripts.lib._lf import OpenAI
    from scripts.lib._pipeline import tokenize, run_adaptive_router_pipeline, native_route

    # ── Load facts context ────────────────────────────────────────────────
    print("Loading MemoryAgentBench dataset for context...")
    ds = load_dataset("ai-hyz/MemoryAgentBench", split="Conflict_Resolution", revision="main")
    row = next(s for s in ds if s["metadata"]["source"] == "factconsolidation_sh_262k")
    ctx = row["context"]

    facts = parse_facts(ctx)
    fact_indices = [f[0] for f in facts]
    fact_texts = [f[1] for f in facts]
    bm25 = BM25Okapi([tokenize(t) for t in fact_texts])
    print(f"  → {len(facts)} facts indexed.\n")

    # ── Load and limit the synthetic benchmark ────────────────────────────
    benchmark_path = ROOT_DIR / "data" / "synthetic_benchmark.json"
    with benchmark_path.open("r", encoding="utf-8") as f:
        benchmark = json.load(f)
    if not (args.all or args.n == 0):
        benchmark = benchmark[:args.n]
    print(f"Loaded {len(benchmark)} synthetic questions from {benchmark_path}\n")

    # ── Run pipeline ──────────────────────────────────────────────────────
    client = OpenAI()
    
    route_correct = 0
    qa_correct = 0
    total = len(benchmark)
    misrouted = []
    qa_wrong = []
    results = []

    print("=" * 90)
    print(f"{'#':>3}  {'Intent':>12}  {'Routed':>12}  {'Route✓':>6}  {'QA✓':>4}  Entity")
    print("-" * 90)

    for i, item in enumerate(benchmark):
        entity = item["entity"]
        expected_intent = item["intent"]
        question = item["question_text"]
        ground_truth = item["ground_truth_answer"]

        # Get the route classification (no LLM cost)
        actual_route = native_route(question)
        route_ok = (actual_route == expected_intent)
        if route_ok:
            route_correct += 1

        # Run the full pipeline (costs LLM tokens for candidate extraction)
        try:
            sh = run_adaptive_router_pipeline(
                question=question, question_index=i,
                ground_truth=[ground_truth],
                bm25=bm25, fact_indices=fact_indices, fact_texts=fact_texts,
                client=client, dataset_name="synthetic_benchmark",
                competency="Conflict_Resolution"
            )
            answer = sh.get("answer", "(no answer)")
        except Exception as e:
            answer = f"<error: {str(e)[:80]}>"
            sh = {}

        # Soft match for QA accuracy
        qa_ok = (normalize(str(answer)) == normalize(str(ground_truth))
                 or normalize(str(ground_truth)) in normalize(str(answer)))
        if qa_ok:
            qa_correct += 1

        # Track misroutes
        if not route_ok:
            misrouted.append({
                "idx": i + 1,
                "entity": entity,
                "expected": expected_intent,
                "actual": actual_route,
                "question": question,
            })

        # Track QA failures
        if not qa_ok:
            qa_wrong.append({
                "idx": i + 1,
                "entity": entity,
                "intent": expected_intent,
                "expected_answer": ground_truth,
                "actual_answer": answer,
                "route": actual_route,
            })

        route_sym = "✓" if route_ok else "✗"
        qa_sym = "✓" if qa_ok else "✗"
        print(f"{i+1:3d}  {expected_intent:>12}  {actual_route:>12}  {route_sym:>6}  {qa_sym:>4}  {entity}")

        results.append({
            "entity": entity,
            "intent": expected_intent,
            "route": actual_route,
            "route_correct": route_ok,
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "qa_correct": qa_ok,
        })

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    print(f"  Routing Accuracy:  {route_correct}/{total} ({100*route_correct/total:.1f}%)")
    print(f"  QA Accuracy:       {qa_correct}/{total} ({100*qa_correct/total:.1f}%)")

    # Per-intent breakdown
    for intent in ["historical", "aggregation", "boolean"]:
        intent_items = [r for r in results if r["intent"] == intent]
        r_ok = sum(1 for r in intent_items if r["route_correct"])
        q_ok = sum(1 for r in intent_items if r["qa_correct"])
        n = len(intent_items)
        print(f"\n  [{intent.upper()}] (n={n})")
        print(f"    Route accuracy: {r_ok}/{n} ({100*r_ok/n:.1f}%)")
        print(f"    QA accuracy:    {q_ok}/{n} ({100*q_ok/n:.1f}%)")

    # Misrouted queries table
    if misrouted:
        print(f"\n{'─' * 90}")
        print(f"MISROUTED QUERIES ({len(misrouted)} total)")
        print(f"{'─' * 90}")
        print(f"{'#':>3}  {'Expected':>12}  {'Actual':>12}  Question")
        print(f"{'─' * 90}")
        for m in misrouted:
            q_short = m["question"][:55] + "..." if len(m["question"]) > 55 else m["question"]
            print(f"{m['idx']:3d}  {m['expected']:>12}  {m['actual']:>12}  {q_short}")
    else:
        print("\n  ✅ No misrouted queries! Perfect routing accuracy.")

    # Save full results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Full results saved to: {output_path}")


if __name__ == "__main__":
    main()
