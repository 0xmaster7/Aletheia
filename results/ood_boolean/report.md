# OOD and Boolean follow-up report

Branch: `scratch/ood-boolean`, based on `scratch/extensions`. No commit, push, or merge was made. The starting code revision was `ecb373a7e41d92cdcdeebc43c27b6267d662e610`.

## Task 0 — saved extension report check (offline)

All requested values in `results/extensions/report.md` match the saved summary files: Aletheia 211/300; RRF 214/300 (16 baseline-right items improved, 13 regressed); ReAct 165/300; saved Direct-fallback hybrid 265/300; saved CoT-fallback hybrid 262/300; RRF recomputed hybrids 261/300 for each fallback. No mismatch found.

## Task 1 — rewording evaluation

The original confirmatory sample contains 98 historical, 102 aggregation, and 100 Boolean questions. The reworded set contains 100 items per type. It is not a one-to-one 300-item sample of the originals. **The 100 historical items are not paraphrases:** they recast the operation as selecting a descending-serial rank, an operator shift. Aggregation and Boolean items reword the same answer-bearing operation and target.

**Paired-comparison rule:** score all 300 reworded items as the evaluation set. For comparisons with the original confirmatory results, compare only exact matched source question IDs, retaining one variant `a` per source ID. This yields 298 paired IDs: 98 historical, 100 aggregation, and 100 Boolean. Do not compare unmatched sets.

The complete reworded-item → original question ID map, including original question text, is in [`task1/paraphrase_source_ids.csv`](task1/paraphrase_source_ids.csv) and [`task1/matched_comparison_plan.json`](task1/matched_comparison_plan.json). Repeated sources: `syn-03753-f5a27a666f1e` (historical-030/a and historical-031/b) and `syn-05490-df2416c53a6d` (historical-047/a and historical-048/b). The `b` variants are retained in all-300 scoring but excluded from paired comparison. Aggregation sources omitted from the reworded set: `syn-08635-073e9e1bf51e` (index 8635; “How many heterogeneous values has Pakistan Hockey Federation accumulated?”) and `syn-10285-2b3387533188` (index 10285; “Counting only unique values, how many values are on file for Jayne Meadows?”). No other source IDs were excluded from the per-type construction; those two are unmatched and are not used in paired comparisons.

All-300 reworded-set results, fixed scorer, audit flags retained:

| Arm | Overall | Historical | Aggregation | Boolean |
|---|---:|---:|---:|---:|
| Aletheia | 169/300 (56.3%) | 20/100 (20.0%) | 89/100 (89.0%) | 60/100 (60.0%) |
| Direct | 202/300 (67.3%) | 48/100 (48.0%) | 65/100 (65.0%) | 89/100 (89.0%) |
| CoT | 260/300 (86.7%) | 95/100 (95.0%) | 72/100 (72.0%) | 93/100 (93.0%) |

All-300 scores excluding flagged rows (22 flagged overall; historical 0, aggregation 1, Boolean 21):

| Arm | Overall | Historical | Aggregation | Boolean |
|---|---:|---:|---:|---:|
| Aletheia | 156/278 (56.1%) | 20/100 (20.0%) | 89/99 (89.9%) | 47/79 (59.5%) |
| Direct | 181/278 (65.1%) | 48/100 (48.0%) | 65/99 (65.7%) | 68/79 (86.1%) |
| CoT | 239/278 (86.0%) | 95/100 (95.0%) | 72/99 (72.7%) | 72/79 (91.1%) |

Paired comparison against original answers on the same 298 source IDs, flags retained. Historical values show the operator shift; aggregation and Boolean values show rewording of the same operation:

| Arm | Paraphrased matched IDs | Original same IDs | Historical new/original | Aggregation new/original | Boolean new/original |
|---|---:|---:|---:|---:|---:|
| Aletheia | 168/298 (56.4%) | 209/298 (70.1%) | 19/98 (19.4%) / 75/98 (76.5%) | 89/100 (89.0%) / 94/100 (94.0%) | 60/100 (60.0%) / 40/100 (40.0%) |
| Direct | 202/298 (67.8%) | 218/298 (73.2%) | 48/98 (49.0%) / 63/98 (64.3%) | 65/100 (65.0%) / 68/100 (68.0%) | 89/100 (89.0%) / 87/100 (87.0%) |
| CoT | 258/298 (86.6%) | 221/298 (74.2%) | 93/98 (94.9%) / 66/98 (67.3%) | 72/100 (72.0%) / 68/100 (68.0%) | 93/100 (93.0%) / 87/100 (87.0%) |

On the 276 unflagged paired IDs, reworded scores are Aletheia 155/276 (56.2%), Direct 181/276 (65.6%), and CoT 237/276 (85.9%). The original scores on those exact IDs are Aletheia 201/276 (72.8%), Direct 198/276 (71.7%), and CoT 200/276 (72.5%). Per-intent flag-excluded values and all scoring rows are preserved in `task1/results_summary.json`. The Aletheia Boolean score rose from 40/100 to 60/100; the reason for that increase is unexplained.

Task 1 used 900 calls, with 0 logged errors and 0 truncations. Usage: 1,400,448 prompt tokens and 61,514 completion tokens. Evaluation API spend was $0.24697560. Two preliminary paraphrase-generation attempts cost $0.00481005; these were not used in the frozen paraphrase set. **Task 1 total: $0.25178565.** Per evaluation arm: Aletheia $0.10485060; Direct $0.06045000; CoT $0.08167500.

The frozen reworded set is 300 unique strings, built once using deterministic structural rewrites with unchanged gold labels and seeded source selection (seed 20261007). Its 100 historical items explicitly change the operation to descending-serial rank; the aggregation and Boolean items rephrase the original operation. Two model-generation drafts were rejected before finalization because historical prompts leaked cue-list words; neither draft was used in the frozen set. Some fixed rewrites are awkwardly phrased; no questions were edited after the set was frozen or after scoring began.

## Task 2a — saved Boolean failure analysis (offline)

Source: confirmatory run `pf-20261006-hybrid300-final-01`; no API calls. All 60 saved Boolean failures replayed exactly from stored plans, contexts, and operator state (43 abstentions, 17 wrong answered). Categories are mutually exclusive:

| Cause | Count | Example question IDs |
|---|---:|---|
| Abstention: no target evidence in K=80 and incomplete history prevented a negative answer | 39 | `syn-09899-3f14162ba310`, `syn-11063-bf793cd201cf`, `syn-09659-189213c324d3` |
| Abstention despite relevant supporting evidence in context (plan/entity/target or span mismatch) | 4 | `syn-00572-6f84a5a47369`, `syn-10046-e215928c5651`, `syn-09530-bd882ed8cd94` |
| Planner field error (8 wrong `negated` values on affirmative questions, 1 wrong target); no intent misroutes | 9 | `syn-11912-2204b0185efd`, `syn-03875-f9c8d8347a7d`, `syn-09020-1298312fd46f` |
| Operator exact-value/span mismatch despite context evidence | 8 | `syn-04184-a4359bec961f`, `syn-01766-da75c32981e2`, `syn-08222-63267832a055` |
| Confirmed gold-answer issue | 0 | — |

Thirteen failed rows carry the audit code `boolean_target_not_found`; that code is not confirmation that the gold is wrong. No confirmed Boolean label disagreement with source support was found. From saved files alone, an absent target in top-K cannot always be distinguished from absence in source history unless source audit facts resolve it. Full item-level diagnostics and the limits of inference are in `task2a/failures.json` and `task2a/analysis.md`.

## Task 2b — single Boolean polarity fix

The one frozen code fix set `negated=false` only when the fresh plan said `negated=true` and the question contained no whole-word `not`, `never`, or `no`. No prompt, retrieval, scorer, or other plan field was changed. It ran once on all 100 confirmatory Boolean items: 100 calls, no retries, 0 errors, 0 truncations; usage 199,787 prompt and 7,627 completion tokens; actual spend **$0.03454425** (under the $0.30 cap). Final run score: 48/100; 43 abstentions.

For causal isolation, replaying the fix on the same fresh planner outputs changed 42/100 correct to 48/100: 6 wrong→right, 0 right→wrong. The six IDs are `syn-03875-f9c8d8347a7d`, `syn-00155-8b9dcae3aa49`, `syn-04055-b1ba570be537`, `syn-03035-ac55f5922583`, `syn-03785-af53b2488d80`, and `syn-10265-f3b1b813a43d`. The fix changed the `negated` field on 21 plans, but only those six changed correctness. Saved baseline was 40/100, while the fresh planner before the fix was 42/100; 41 fresh plans differed from saved plans in at least one field, so comparing 40→48 mixes planner output drift with the code fix. The isolated fix effect is 42→48, not 40→48. Full flips and raw plans are in `task2b/results_summary.json` and `task2b/plans_and_scores.jsonl`.

## Spend and deviations

| Work | Calls | Actual API spend |
|---|---:|---:|
| Task 0 saved-file audit | 0 | $0 |
| Task 1 paraphrase generation drafts | 2 | $0.00481005 |
| Task 1 three-arm OOD run | 900 | $0.24697560 |
| Task 2a saved-file analysis | 0 | $0 |
| Task 2b Boolean fix run | 100 | $0.03454425 |
| **Total** | **1,002** | **$0.28632990** |

The paraphrase construction deviated from an LLM paraphrasing workflow: deterministic structural rewrites were used after both generation drafts failed cue-leak validation. This preserved gold labels and enabled a fixed, auditable set but may yield less natural language variation. The historical set is more than a phrasing rewrite: it changes the task wording to an explicit descending-serial rank operation. A Task 1 runner checkpoint initially remained `running` after all calls; terminal call rows were verified and the checkpoint was corrected to `complete`. The Task 1 scorer wrapper's matched-ID bookkeeping was corrected after scoring to enforce the user's paired-only rule; no requests, model outputs, scorer logic, or reworded questions changed. Task 2b's fresh planner outputs differed from the saved planner outputs on 41 items; results are reported with that caveat. The polarity fix was designed on these same questions, so the observed change is not independent validation. No experiment was retried or tuned after result review.

Combined Task 1 + Task 2b spend was $0.28632990, below the aggregate $1.00 ceiling. No experiment remains to run.
