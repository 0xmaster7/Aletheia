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

## Confirmatory-set extensions (GPT-4o-mini, temperature 0)

| Arm or policy | Correct | Accuracy |
| :--- | ---: | ---: |
| Aletheia | 211/300 | 70.3% |
| Direct | 220/300 | 73.3% |
| CoT | 223/300 | 74.3% |
| ReAct | 165/300 | 55.0% |
| Aletheia + RRF | 214/300 | 71.3% |
| RRF + Direct fallback | 261/300 | 87.0% |
| RRF + CoT fallback | 261/300 | 87.0% |
| Saved Direct-fallback hybrid | 265/300 | 88.3% |
| Saved CoT-fallback hybrid | 262/300 | 87.3% |

RRF improved 16 questions and regressed 13. On the 100 aggregation items, original → reworded counts were Aletheia 94 → 89, Direct 68 → 65, and CoT 68 → 72. On the 100 Boolean items, they were 40 → 60, 87 → 89, and 87 → 93; the Aletheia increase is unexplained. The historical set was recast as descending-serial-rank questions, an operator shift rather than paraphrase. On 98 matched IDs, counts were Aletheia 75 → 19, Direct 63 → 48, and CoT 66 → 93.

The 60 Boolean failures were 39 abstentions without target evidence, 4 abstentions despite evidence, 9 planner-field errors, 8 operator mismatches, and 0 confirmed gold issues. The polarity-fix score was 40/100 on the saved baseline and 48/100 in the fresh run. Same-fresh-plan replay was 42/100 → 48/100, with 6 wrong-to-right and 0 right-to-wrong flips. The planner differed from the saved plans on 41 questions; the fix was designed on these same questions, so this result is not an independent test.

Detailed data and reproduction notes are in [`results/extensions/`](../results/extensions/README.md) and [`results/ood_boolean/`](../results/ood_boolean/README.md).

## Routing and scope

The original pipeline includes an embedding-based semantic router. Final-v3 and the confirmatory synthetic run instead use an LLM planner and do not use the embedding router. Any reported 93.3% routing figure applies only to the first-60, in-distribution routing examples, not to final-v3.
