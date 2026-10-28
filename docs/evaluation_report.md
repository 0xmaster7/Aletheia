# Final native-router paper evaluation

This report follows the supplied final IEEE Access Word manuscript. The scientific path is `native_route()` plus candidate extraction and deterministic operators in `scripts/lib/_pipeline.py`. The older planner evaluation is archived in [legacy planner documentation](../archive/legacy_planner_docs/evaluation_report.md).

## Results in the final paper

All results are single runs. Synthetic scores retain audit-flagged questions unless marked unflagged; abstentions and truncated answers count as incorrect. Routing accuracy measures intent selection, not answer correctness.

### FactConsolidation at 262k tokens — Table I

Each row evaluates the same 100 single-hop questions with GPT-4o-mini. The direct BM25 controls use the same generic baseline prompt and SubEM scorer; Aletheia uses its candidate-extraction prompt and deterministic operators.

| System | Temperature | Retrieval K | Correct / accuracy |
| --- | ---: | --- | ---: |
| BM25, original baseline | 0.7 | 10 | 56/100 (56.0%) |
| BM25, matched baseline | 0.0 | 10 | 59/100 (59.0%) |
| BM25, matched baseline | 0.0 | 10 or 25 | 58/100 (58.0%) |
| Aletheia, adaptive router | 0.0 | 10 or 25 | 81/100 (81.0%) |

Routed K is 25 for 15 questions and 10 for 85. Matching temperature and retrieval size does not isolate routing, extraction, or execution individually. The GPT-4o and fixed-window reruns below reach the same accuracy or higher.

### Synthetic samples — Table II

The released benchmark has 13,425 templated questions. The full set was not evaluated. The final sample excludes 660 previously used question IDs; the confirmatory sample excludes 960.

| Sample | n | Routing accuracy | Aletheia answers | Direct answers | CoT answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| Second, seed 20261004 | 300 | 84.3% (253/300) | Not run | 78.0% | 80.0% |
| Final, own-K, seed 20261005 | 300 | 80.0% (240/300) | 50.7% (152/300) | 77.3% (232/300) | 78.7% (236/300) |
| Final, K=80, same IDs | 300 | 80.0% (240/300) | 50.3% (151/300) | 77.3% (232/300) | 78.7% (236/300) |
| Confirmatory, seed 20261006 | 300 | 82.3% (247/300) | 50.7% (152/300) | 73.3% (220/300) | 74.3% (223/300) |

Own-K uses historical K=25, aggregation K=40, and Boolean/freshness K=10, with query clean-up for historical and aggregation retrieval. The K=80 runs use the same raw-question BM25 context as Direct and CoT. The final K=80 run reuses saved native routes. Direct and CoT are the same saved controls in both final-sample rows. These comparisons are not prompt-matched.

The best earlier first-60 routing run was 56/60 (93.3%), an in-distribution result on shared template vocabulary. The earlier first-60 answer run was 22/60 (36.7%) with an earlier pipeline and scorer; it is not a baseline comparison.

### Final sample by intent — Table III

| Intent | n | Aletheia own-K | Aletheia K=80 | Direct | CoT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical | 102 | 33.3% | 33.3% | 75.5% | 78.4% |
| Aggregation | 101 | 51.5% | 50.5% | 70.3% | 70.3% |
| Boolean | 97 | 68.0% | 68.0% | 86.6% | 87.6% |
| Overall | 300 | 50.7% | 50.3% | 77.3% | 78.7% |

Aletheia trails both controls on every intent. Own-K answers 251 questions and abstains on 49, with 60.6% accuracy among answered questions (152/251). K=80 answers 253 and abstains on 47, with 59.7% accuracy among answered questions (151/253). One answer in each run was truncated at the 1,024-token cap and counted as incorrect. Answered-only accuracy does not replace overall accuracy.

### FactConsolidation ablations — Table IV

Each reported cell has 100 questions. `n/r` means not run. Asterisks identify imported results from the source paper/repository, not our reruns.

| Configuration | 6k | 32k | 64k | 262k |
| --- | ---: | ---: | ---: | ---: |
| Aletheia adaptive router, GPT-4o-mini | n/r | 77% | 81% | 81% |
| Earlier single-hop fact extraction, GPT-4o-mini, mixed provenance | 43% | 78%* | 81%* | 43% |
| Single-hop fact chunks, GPT-4o rerun | 98% | 93% | 95% | 94% |
| Single-hop 4096-character windows, GPT-4o-mini rerun | 90% | 82% | 81% | 81% |
| BM25 baseline, GPT-4o-mini | 58% | 70%* | 75%* | 56% |
| Multi-hop CAR, GPT-4o-mini rerun | 47% | 39% | 45% | 34% |

CAR uses the separate multi-hop split and is not directly comparable with the single-hop rows. The stronger extractor exceeds adaptive routing, and fixed windows tie it at 64k and 262k. No component-level ablation establishes the cause of the earlier-pipeline versus adaptive-router difference.

### Confirmatory standalone results — Table V

All arms read the same K=80 pool. Aletheia uses a 512-token cap; three truncated answers count as incorrect.

| Subset | n | Aletheia | Direct | CoT |
| --- | ---: | ---: | ---: | ---: |
| Historical | 98 | 39.8% | 64.3% | 67.3% |
| Aggregation | 102 | 50.0% | 68.6% | 68.6% |
| Boolean | 100 | 62.0% | 87.0% | 87.0% |
| Overall | 300 | 50.7% | 73.3% | 74.3% |
| Aggregation, unflagged | 101 | 50.5% | 69.3% | 69.3% |
| Boolean, unflagged | 79 | 73.4% | 84.8% | 83.5% |
| Overall, unflagged | 278 | 53.2% | 71.9% | 72.7% |

Overall correct counts are 152, 220, and 223, respectively; after excluding flagged questions, they are 148, 200, and 202. Aletheia answers 242 questions and abstains on 58, with 62.8% accuracy among answered questions (152/242).

### Confirmatory fallback hybrids — Table VI

Use Aletheia's answer when it produces one; otherwise use the named baseline answer, uniformly across intents.

| Subset | n | Direct fallback | CoT fallback |
| --- | ---: | ---: | ---: |
| Historical | 98 | 64.3% | 63.3% |
| Aggregation | 102 | 50.0% | 50.0% |
| Boolean | 100 | 79.0% | 79.0% |
| Overall | 300 | 64.3% (193/300) | 64.0% (192/300) |
| Overall, unflagged | 278 | 63.7% (177/278) | 63.3% (176/278) |

Both hybrids remain below standalone Direct and CoT. They add 41 and 40 correct answers over standalone Aletheia. Aletheia supplies 242/300 answers (80.7%), or 232/278 (83.5%) on the unflagged subset. These are policy results, not standalone Aletheia accuracy.

### Measured API cost and routing latency

| Run, 300 questions | Recorded API spend, USD |
| --- | ---: |
| Final Aletheia own-K | $0.040 |
| Final Aletheia K=80 | $0.078 |
| Final Direct | $0.059 |
| Final CoT | $0.071 |
| Confirmatory Aletheia K=80 | $0.078 |

These are historical usage charges, not current price quotes or full deployment costs. Aletheia is cheaper than the controls with own-K but more expensive with K=80; sequential fallback deployment cost was not measured.

The saved local embedding-router benchmark reports a 61.73 ms mean (about 62 ms), 42–259 ms range, and a 395.5 ms first warm-up inference excluded from the mean (model loading is outside that timer). It used an Intel Core i7-8850H at 2.60 GHz, one warm-up, and four distinct questions repeated 25 times (100 timed queries). This is routing latency only, not end-to-end pipeline latency. See [the benchmark](../benchmarks/latency/test_latency.py) and [saved log](../logs/latency/routing_latency_test.log).


## Evidence and scoring

The result-to-file mapping and runnable offline checks are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The current paper is included in the local faculty package at `paper/Aletheia_IEEE_Access_Updated.docx`.

Use `scores_fixed.json` for the final own-K arm: its initial `scores.json` counted 16 Boolean abstentions as correct. The native-router own-K evidence-gated hybrid diagnostic is separate from the uniform confirmatory fallback policy reported in Table VI.

The RRF, ReAct, rewording, and Boolean-fix packages remain preserved under `results/extensions/` and `results/ood_boolean/`. They belong to older planner experiments and are not additional tables in this supplied manuscript.

## Interpretation

The 81% versus 58–59% FactConsolidation comparison is a whole-pipeline result. GPT-4o extraction reaches 94% at 262k, and 4096-character windows reach 81%, so no isolated routing advantage is established. On synthetic questions, the native-router system trails Direct and CoT for every intent; the confirmatory fallback hybrids also trail both controls. The benchmark is templated and contains flagged labels; single runs and sampled subsets do not establish generalization or production readiness.
