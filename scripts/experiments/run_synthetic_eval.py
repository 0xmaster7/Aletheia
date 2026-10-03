"""Run the separate 100-query synthetic evaluation (the paper reports 60 questions).

Usage:
  python scripts/experiments/run_synthetic_eval.py --pipeline author
  python scripts/experiments/run_synthetic_eval.py --pipeline custom
"""
import argparse
import json
import os
import sys
import time
from datasets import load_dataset
from rank_bm25 import BM25Okapi

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)
from scripts.lib._lf import OpenAI
from scripts.lib._pipeline import tokenize, run_car_v2, run_adaptive_router_pipeline

def parse_facts(ctx: str) -> list[tuple[int, str]]:
    import re
    pat = re.compile(r"(\d+)\.\s")
    matches = list(pat.finditer(ctx))
    facts = []
    for i, m in enumerate(matches):
        idx = int(m.group(1))
        s = m.end()
        e = matches[i + 1].start() if i + 1 < len(matches) else len(ctx)
        facts.append((idx, ctx[s:e].strip().rstrip(".")))
    return facts

def normalize(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", str(s).strip().lower())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", required=True, choices=["author", "custom"])
    args = p.parse_args()

    # 1. Load context from original dataset
    print("Loading MemoryAgentBench context...")
    ds = load_dataset("ai-hyz/MemoryAgentBench", split="Conflict_Resolution", revision="main")
    row = next(s for s in ds if s["metadata"]["source"] == "factconsolidation_sh_262k")
    facts = parse_facts(row["context"])
    fact_indices = [f[0] for f in facts]
    fact_texts = [f[1] for f in facts]
    bm25 = BM25Okapi([tokenize(t) for t in fact_texts])

    # 2. Load synthetic benchmark (n=100)
    bench_path = os.path.join(ROOT_DIR, "data", "synthetic_benchmark.json")
    with open(bench_path, "r") as f:
        benchmark = json.load(f)[:100]

    print(f"Starting {args.pipeline} evaluation on {len(benchmark)} synthetic queries...")
    client = OpenAI()
    correct = 0
    results = []
    t0 = time.time()

    for i, item in enumerate(benchmark):
        q = item["question_text"]
        gt = item["ground_truth_answer"]

        try:
            if args.pipeline == "author":
                res = run_car_v2(
                    question=q, question_index=i, ground_truth=[gt],
                    bm25=bm25, fact_indices=fact_indices, fact_texts=fact_texts,
                    client=client, dataset_name="synthetic_100", competency="Conflict_Resolution"
                )
                answer = res.get("answer", "(no answer)")
            else:
                res = run_adaptive_router_pipeline(
                    question=q, question_index=i, ground_truth=[gt],
                    bm25=bm25, fact_indices=fact_indices, fact_texts=fact_texts,
                    client=client, dataset_name="synthetic_100", competency="Conflict_Resolution"
                )
                answer = res.get("answer", "(no answer)")
        except Exception as e:
            answer = f"<error: {e}>"

        is_correct = (normalize(answer) == normalize(gt) or normalize(gt) in normalize(answer))
        if is_correct:
            correct += 1
            
        results.append({
            "question": q, "ground_truth": gt, "answer": answer, "is_correct": is_correct,
            "intent": item["intent"]
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(benchmark)}] Accuracy: {100 * correct / (i + 1):.1f}%")

    elapsed = time.time() - t0
    acc = correct / len(benchmark)
    
    out = {
        "pipeline": args.pipeline, "n": len(benchmark), "accuracy": acc,
        "elapsed_s": elapsed, "results": results
    }
    os.makedirs(os.path.join(ROOT_DIR, "results", "poc_results"), exist_ok=True)
    out_path = os.path.join(ROOT_DIR, "results", "poc_results", f"synthetic_{args.pipeline}_n100.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n✅ DONE. Accuracy: {100 * acc:.1f}%")
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
