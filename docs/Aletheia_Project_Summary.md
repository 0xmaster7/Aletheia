# Aletheia: Comprehensive Project Summary

## 1. Project Aim & Background
**Aletheia** explores memory conflict resolution with model-based planning and extraction followed by deterministic operators. Deterministic execution avoids delegating sorting, counting, and Boolean evaluation to free-form generation, while planning, retrieval, and extraction can still fail.

Some retrieval-augmented generation (RAG) pipelines rely on free-form generation for sorting, counting, and Boolean operations, which can produce calculation errors. Freshness-oriented retrieval may also miss the intent of historical, aggregation, or Boolean questions unless the system explicitly handles those operations.

**The Goal:** To separate intent planning, evidence extraction, and mathematical execution, and to evaluate the resulting pipeline against Direct and CoT controls.

---

## 2. The Aletheia Architecture
Aletheia's repository contains the original embedding-router pipeline and a later LLM-planner evaluation path. Final-v3 and the confirmatory synthetic runs use the LLM planner, not the embedding router; Aletheia, Direct, and CoT receive the same top-80 BM25 context in those runs.

### A. Original Embedding Router and Final-v3 Planner
The original pipeline uses an offline `semantic-router` with local `all-MiniLM-L6-v2` embeddings and predefined anchor utterances to classify queries into four routes:
1. **Freshness:** Requires the most recent state.
2. **Historical:** Requires a past state.
3. **Aggregation:** Requires a count or list of states.
4. **Boolean:** Requires a True/False validation.

Final-v3 and the confirmatory synthetic evaluation instead ask an LLM planner to return a structured query plan from the question and retrieved context. The embedding router is not used in those runs.

### B. Entity Extraction Layer (BM25 + LLM)
The final-v3 comparison gives Aletheia, Direct, and CoT the same top-80 BM25 facts. Aletheia's planner returns a structured plan without candidate facts; Python selects and validates candidate facts before applying the intent-specific operator.

### C. Adaptive Operator Layer
In final-v3, Python validates the planner's query plan, selects and checks facts, and applies the intent-specific operator. Historical, aggregation, and negative Boolean answers require verified complete history; positive Boolean evidence can answer when a matching fact is found. The original embedding-router pipeline remains a separate legacy path.

---

## 3. Testing & Benchmarking
The architecture was tested using two distinct paradigms:

### A. Final-v3 Synthetic Benchmark
On the 300-question final-v3 sample, with audit flags retained, Aletheia scored **68.0% (204/300)**, Direct **77.3% (232/300)**, and CoT **78.7% (236/300)**. With flags retained, Aletheia alone is below both baselines overall.

### B. Confirmatory Hybrid Run
On a separate 300-question sample, with audit flags retained, Aletheia alone scored **70.3% (211/300)**, Direct **73.3% (220/300)**, and CoT **74.3% (223/300)**. With flags retained, Aletheia alone is below both baselines overall in this confirmatory run. The Direct-fallback hybrid scored **88.3% (265/300)** and the CoT-fallback hybrid **87.3% (262/300)**. These are policy results that use the named baseline when Aletheia is ineligible to answer.

### Confirmatory-set extensions

The RRF arm scored **214/300 (71.3%)**, improving 16 questions and regressing 13 against saved Aletheia; ReAct scored **165/300 (55.0%)**. RRF with either Direct or CoT fallback scored **261/300 (87.0%)**, compared with the saved Direct- and CoT-fallback hybrids at **265/300 (88.3%)** and **262/300 (87.3%)**.

On aggregation items (n=100), original → reworded correct counts were Aletheia 94 → 89, Direct 68 → 65, and CoT 68 → 72. On Boolean items (n=100), counts were 40 → 60, 87 → 89, and 87 → 93; the Aletheia Boolean increase is unexplained. The historical test recast questions as descending-serial-rank questions, an operator shift rather than a paraphrase. On 98 matched IDs, Aletheia changed 75 → 19, Direct 63 → 48, and CoT 66 → 93.

The 60 Boolean failures comprised 39 abstentions without target evidence, 4 abstentions despite evidence, 9 planner-field errors, 8 operator mismatches, and 0 confirmed gold issues. A polarity fix scored 40/100 on the saved baseline and 48/100 on its fresh run. Same-fresh-plan replay was 42/100 → 48/100 (6 wrong-to-right, 0 right-to-wrong); planner drift occurred on 41 items. The fix was designed on these same questions, so this is not independent validation. See [`results/extensions/`](../results/extensions/README.md) and [`results/ood_boolean/`](../results/ood_boolean/README.md) for details.

---

## 4. Ablation Studies
The ablation suite (`scripts/experiments/14_ablations.py`) evaluates several configurations. The ablation driver supports:
1. **Chunk-Size Ablation:** Comparing fact-level chunking against 4096-character sliding windows to test context concentration.
2. **Pipeline Strategy Ablation:** Comparing Single-Hop (`sh`), Multi-Hop CAR pipelines (`mh`), and the legacy adaptive embedding-router pipeline (`adaptive`).
3. **Model Backbone Comparison:** Running single-hop extraction with `gpt-4o-mini` and `gpt-4o`; this comparison does not isolate routing effects.

The earlier ablations in `results/poc_results/` are separate historical experiments. At 262k, the matched BM25 runs score 59/100 with fixed K=10 and 58/100 with Aletheia-routed K=10/25, compared with the saved adaptive Aletheia reference of 81/100. On final-v3, Aletheia alone is below both baselines overall, with flags retained.
