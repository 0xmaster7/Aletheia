# Aletheia project summary

Aletheia studies whether separating language understanding from deterministic execution improves resolution of conflicting memory facts. The final paper evaluates a local embedding router, BM25 retrieval, GPT-4o-mini candidate extraction, and Python operators. It is a research prototype.

## Code path described by the paper

1. `scripts/lib/_pipeline.py:native_route` embeds the question with MiniLM and selects the highest maximum cosine similarity among the precomputed example banks. The code route `current_value` is the paper's freshness intent.
2. `run_adaptive_router_pipeline` retrieves K=10 for current-value/Boolean, K=25 for historical, and K=40 for aggregation. Historical/aggregation questions undergo rule-based query clean-up in the own-K route. The K=80 evaluation wrappers instead reuse the shared raw-question retrieval pool.
3. `_extract_candidates` calls GPT-4o-mini in JSON mode with `CANDIDATE_PROMPT`; it returns serial, fact text, and answer entity.
4. Python chooses the latest/earlier state, normalizes/deduplicates/counts values, or checks Boolean containment and negation. Empty candidates produce no answer for freshness/historical, False for Boolean, and zero or an empty list for aggregation.

Historical selection uses the oldest state by default, or a cue-dependent offset over consecutive distinct states sorted by serial. An out-of-range offset falls back to the earliest available state. Boolean "ever" questions check all candidates; other Boolean questions check the latest state. A bare ordinal without a recognized temporal cue (for example, "second city") returns the oldest state. Deduplication is string-level: NYC and New York City remain distinct. Missing-candidate False/zero outputs mean no evidence was extracted; they are not verified negative facts or counts. Only the fixed negation strings are recognized. These are fixed hand-written rules, not a learned general parser.

The native router uses the first `ROUTE_UTTERANCES` bank through precomputed `ROUTE_EMBEDDINGS`. The same module also contains a separate routing helper with another utterance dictionary. The final paper's saved native runs must be interpreted through `native_route`, not that other helper. The pipeline and scorer match the hashes recorded in the native-run manifests.

## Main results

| Evaluation | Native Aletheia | Direct | CoT |
| --- | ---: | ---: | ---: |
| Final 300, own-K | 152/300 (50.7%) | 232/300 (77.3%) | 236/300 (78.7%) |
| Final 300, shared K=80 | 151/300 (50.3%) | 232/300 (77.3%) | 236/300 (78.7%) |
| Confirmatory 300, K=80 | 152/300 (50.7%) | 220/300 (73.3%) | 223/300 (74.3%) |

The confirmatory Direct-fallback hybrid scores 193/300 (64.3%) and the CoT-fallback hybrid 192/300 (64.0%). Both remain below standalone controls.

At 262k tokens on FactConsolidation, native adaptive Aletheia scores 81/100 versus matched BM25 at 59/100 and 58/100. GPT-4o extraction scores 94/100, while fixed 4096-character windows tie Aletheia at 81/100. No component ablation isolates the reason for the gap to the generic baseline.

Routing accuracy is 80.0% on the final sample and 82.3% on the confirmatory sample; the second sample is 84.3%. The earlier best 93.3% routing figure is 56/60 on in-distribution templates, not end-to-end answer accuracy. Routing-only latency averages 61.73 ms in the saved CPU benchmark; no end-to-end latency was measured.

## Reproduction and limitations

See [the evaluation report](evaluation_report.md) for all six numerical tables and [the reproducibility guide](REPRODUCIBILITY.md) for saved inputs and read-only checks. The benchmark contains 13,425 templated questions, with 684 flagged rows; it was not evaluated in full. Deterministic operators do not remove routing, retrieval, extraction, or interpretation errors.

The older LLM-planner code and its extension experiments remain for provenance. Their documentation is in [the archive](../archive/legacy_planner_docs/). They are separate from the paper's native-router system. The authoritative Word manuscript is included locally in the faculty ZIP at `paper/Aletheia_IEEE_Access_Updated.docx`.
