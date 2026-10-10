# Aletheia: Memory Conflict Resolution Evaluation

Unless a subsection says otherwise, scores use the fixed scorer and retain audit-flagged rows. The original embedding-router design and the later LLM-planner experiments are separate system paths: final-v3 and the confirmatory runs use an LLM planner, not the embedding router.

## Final-v3 synthetic benchmark (300 questions; audit flags retained)

| Arm | Correct | Accuracy |
| :--- | ---: | ---: |
| Aletheia | 204/300 | 68.0% |
| Direct | 232/300 | 77.3% |
| CoT | 236/300 | 78.7% |

With flags retained, Aletheia alone is below both baselines overall. It answered 228 questions (76%); accuracy when answered was 89.5% (204/228). It abstained on 72 questions. On those same answered items, Direct scored 78.1% and CoT scored 78.9%; on the 72 abstentions, Direct scored 75.0% and CoT 77.8%.

## Confirmatory hybrid run (300 questions; audit flags retained)

| Arm or policy | Correct | Accuracy |
| :--- | ---: | ---: |
| Aletheia alone | 211/300 | 70.3% |
| Direct | 220/300 | 73.3% |
| CoT | 223/300 | 74.3% |
| Aletheia with Direct fallback | 265/300 | 88.3% |
| Aletheia with CoT fallback | 262/300 | 87.3% |

With flags retained, Aletheia alone is below both baselines overall. Hybrid results use the named control when Aletheia is ineligible to answer; they are policy scores, not Aletheia-alone scores.

## FactConsolidation at 262k tokens (100 questions)

| System | Correct | Accuracy |
| :--- | ---: | ---: |
| Adaptive Aletheia saved reference | 81/100 | 81.0% |
| BM25, fixed K=10 | 59/100 | 59.0% |
| BM25, Aletheia-routed K=10/25 | 58/100 | 58.0% |

The BM25 runs use GPT-4o-mini at temperature 0, the same saved questions and the unchanged `QUERY_TEMPLATE_BM25`. Routed K=10/25 matches Aletheia's per-question retrieval size; fixed K=10 is a separate control. The 81/100 Aletheia score is the earlier adaptive result; the comparison is a whole-pipeline comparison and does not isolate one component. The older 81-versus-56 comparison used different retrieval/generation settings and is not the matched result.

### FactConsolidation ablations (Table VI; 100 questions per cell)

| System | 6k | 32k | 64k | 262k |
| :--- | ---: | ---: | ---: | ---: |
| Aletheia, adaptive router, GPT-4o-mini | n/r | 77% | 81% | 81% |
| Earlier single-hop GPT-4o-mini pipeline (mixed provenance) | 43% | 78%* | 81%* | 43% |
| Single-hop fact extraction, GPT-4o rerun | 98% | 93% | 95% | 94% |
| Single-hop, 4096-character windows, GPT-4o-mini rerun | 90% | 82% | 81% | 81% |
| BM25 baseline, GPT-4o-mini | 58% | 70%* | 75%* | 56% |
| Multi-hop CAR, GPT-4o-mini, multi-hop split | 47% | 39% | 45% | 34% |

`*` The 32k and 64k earlier-pipeline/BM25 values are from [8], Table 3 and its repository; they were not rerun here. The earlier-pipeline row has mixed provenance. The 262k BM25 score of 56% is from the original temperature-0.7, K=10 run; the matched temperature-0 BM25 reruns are 59/100 and 58/100 above. CAR uses a different multi-hop split. The October 3 GPT-4o, 4096-character-window and CAR rows are reruns with this repository's implementations and prompts; they are not exact reproductions of [8]. The 4096-character windows match or exceed adaptive Aletheia at 32k, 64k and 262k, and the GPT-4o rerun is higher than adaptive Aletheia at those same lengths; there is no adaptive result at 6k. These results do not isolate a causal router effect.

## Section VI-F extensions (confirmatory set; GPT-4o-mini, temperature 0)

The saved 300-question confirmatory sample has 98 historical, 102 aggregation and 100 Boolean questions. All 22 audit-flagged rows were retained. The baseline values below come from the saved confirmatory run; RRF and ReAct are additional arms on those same questions.

| Intent | n | Aletheia | RRF | ReAct | Direct | CoT |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical | 98 | 75 (76.5%) | 77 (78.6%) | 34 (34.7%) | 63 (64.3%) | 66 (67.3%) |
| Aggregation | 102 | 96 (94.1%) | 95 (93.1%) | 49 (48.0%) | 70 (68.6%) | 70 (68.6%) |
| Boolean | 100 | 40 (40.0%) | 42 (42.0%) | 82 (82.0%) | 87 (87.0%) | 87 (87.0%) |
| Overall | 300 | 211 (70.3%) | 214 (71.3%) | 165 (55.0%) | 220 (73.3%) | 223 (74.3%) |

RRF used BM25 and `all-MiniLM-L6-v2` dense retrieval with equal-weight reciprocal rank fusion, rank constant 60 and K=80. It improved 16 items and regressed 13, a net gain of three questions (about one percentage point). Its Direct- and CoT-fallback hybrids each scored 261/300 (87.0%), below the saved BM25 hybrids at 265/300 (88.3%) and 262/300 (87.3%); RRF did not improve hybrid accuracy. The RRF run cost $0.10219620 from usage fields.

ReAct searched the same corpus with BM25 top-10 results, up to six sequential search steps per question, a 512-token completion limit, and no removal of repeated results. It made 1,709 calls, including 1,409 search steps, with no errors or skipped questions. The run cost $0.33895905. These settings were chosen for this simple baseline and do not show what a tuned ReAct agent could achieve.

### Rewording and historical operator-shift test

The aggregation and Boolean questions preserve the requested operation. Historical questions were recast as descending-serial-rank questions, changing the operation; they are an operator-shift test, not paraphrases. Paired comparisons use only the 298 matched source IDs: 98 historical, 100 aggregation and 100 Boolean. No pooled 300-question paraphrase score is reported.

| Intent / interpretation | n | Aletheia original / changed | Direct original / changed | CoT original / changed |
| :--- | ---: | ---: | ---: | ---: |
| Aggregation rewording | 100 | 94 / 89 | 68 / 65 | 68 / 72 |
| Boolean rewording | 100 | 40 / 60 | 87 / 89 | 87 / 93 |
| Historical descending-serial-rank operator shift | 98 | 75 / 19 | 63 / 48 | 66 / 93 |

Audit flags were retained. On aggregation, excluding flagged rows, Aletheia scored 94/99 on originals and 89/99 on reworded questions. On Boolean, it scored 32/79 and 47/79. The Aletheia Boolean rise from 40/100 to 60/100 is unexplained and should not be read as an improvement. The paraphrases were model-written and may differ in difficulty.

### Boolean failure diagnosis and same-sample polarity fix

Among the 60 Boolean failures in the confirmatory run, 39 were abstentions with no matching target in the saved K=80 context and incomplete history; 4 were abstentions despite supporting context; 9 were planner-field errors (8 affirmative questions marked `negated=true`, 1 target extraction error); 8 were operator value/span mismatches; and 0 were confirmed gold-answer issues. The 39 cases are direct observations of the saved context and completeness check, not proof that the source corpus lacks the target. Classifying the four evidence-present abstentions and planner conflicts involves interpreting the question wording. Thirteen failures carried the `boolean_target_not_found` audit flag, which does not establish a wrong gold label. The saved context does not always show whether a target absent from top-K is absent from the full source history.

The evaluation-only polarity rule set `negated=false` if a Boolean plan had `negated=true` and the question contained none of the whole-word cues `not`, `never` or `no`. The fresh run scored 48/100, but 41 plans differed from the saved plans in other fields. On same-fresh-plan replay, the score changed from 42/100 to 48/100, with six wrong-to-right flips and no right-to-wrong flips. The fix was developed and evaluated on the same 100 Boolean questions, so this is a same-sample diagnostic, not held-out validation. The prompt and planner were unchanged, and the hybrid was not rerun with the fix.

## Scope of evidence

The original embedding router's first-60 in-distribution routing accuracy is separate from final-v3: final-v3 and confirmatory experiments use the LLM planner. The 93.3% routing figure describes only the first-60 in-distribution examples. It is not a final-v3 routing metric. The synthetic benchmark is templated, single-sample results do not establish generalization, and the extension findings include a negative RRF result, an unexplained Boolean rewording increase, and a same-sample Boolean fix. The saved artifact paths, commands, and provenance are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md), [`results/extensions/README.md`](../results/extensions/README.md), and [`results/ood_boolean/README.md`](../results/ood_boolean/README.md).
