# Aletheia: Memory Conflict Resolution Evaluation

## Final-v3 synthetic benchmark (300 questions; audit flags retained)

The frozen final-v3 evaluation used the fixed scorer and retained all audit-flagged rows.

| Arm | Correct | Accuracy |
| :--- | ---: | ---: |
| Aletheia | 204/300 | 68.0% |
| Direct | 232/300 | 77.3% |
| CoT | 236/300 | 78.7% |

With flags retained, Aletheia alone is below both baselines overall.

## Confirmatory hybrid run (300 questions; audit flags retained)

This run used a separate frozen sample and the same fixed scorer.

| Arm or policy | Correct | Accuracy |
| :--- | ---: | ---: |
| Aletheia alone | 211/300 | 70.3% |
| Direct | 220/300 | 73.3% |
| CoT | 223/300 | 74.3% |
| Aletheia with Direct fallback | 265/300 | 88.3% |
| Aletheia with CoT fallback | 262/300 | 87.3% |

The hybrid policies use Aletheia's answer when eligible and otherwise use the named fallback. Their scores are not Aletheia-alone scores.
With flags retained, Aletheia alone is also below both baselines overall in this confirmatory run.

## Matched FactConsolidation baseline check (262k; 100 questions)

| System | Correct | Accuracy |
| :--- | ---: | ---: |
| Adaptive Aletheia saved reference | 81/100 | 81.0% |
| BM25, fixed K=10 | 59/100 | 59.0% |
| BM25, Aletheia-routed K=10/25 | 58/100 | 58.0% |

The BM25 arms use the same saved question set and context size, GPT-4o-mini at temperature 0, and the unchanged `QUERY_TEMPLATE_BM25`. The 81/100 Aletheia value is the earlier adaptive result. This comparison does not isolate the effect of any single component. The older 81%-versus-56% figures are from a separate, unmatched comparison and should not be presented as this matched result.

## Routing and scope

The original pipeline includes an embedding-based semantic router. Final-v3 and the confirmatory synthetic run instead use an LLM planner and do not use the embedding router. Any reported 93.3% routing figure applies only to the first-60, in-distribution routing examples, not to final-v3.
