# Confirmatory-set extension experiments

Branch: `scratch/extensions` (local only; no commit, push, or merge). Code commit recorded before the runs: `ecb373a7e41d92cdcdeebc43c27b6267d662e610`.
Sample: the same ordered 300 confirmatory questions; 98 temporal/historical, 102 aggregation, 100 Boolean. All 22 audit-flagged rows are retained.
Scorer: frozen `scripts/lib/evaluation_scorer.py` snapshot; exact hash in `frozen_manifest.json`. No paper, prompt, planner, scorer, or cue-list edits were made.

## Accuracy

| Type | N | Saved Aletheia | Aletheia + RRF | Direct (saved) | CoT (saved) | ReAct |
|---|---:|---:|---:|---:|---:|---:|
| Temporal / historical | 98 | 75/98 (76.5%) | 77/98 (78.6%) | 63/98 (64.3%) | 66/98 (67.3%) | 34/98 (34.7%) |
| Aggregation | 102 | 96/102 (94.1%) | 95/102 (93.1%) | 70/102 (68.6%) | 70/102 (68.6%) | 49/102 (48.0%) |
| Boolean | 100 | 40/100 (40.0%) | 42/100 (42.0%) | 87/100 (87.0%) | 87/100 (87.0%) | 82/100 (82.0%) |
| Overall | 300 | 211/300 (70.3%) | 214/300 (71.3%) | 220/300 (73.3%) | 223/300 (74.3%) | 165/300 (55.0%) |

RRF improves Aletheia by 3 answers overall (214/300 versus 211/300, +1.0 percentage point): 16 questions improved and 13 regressed. It remains below Direct and CoT. ReAct scores 165/300 (55.0%), below all three saved arms.

## RRF hybrid results

| Type | N | Direct fallback | CoT fallback |
|---|---:|---:|---:|
| Temporal / historical | 98 | 83/98 (84.7%) | 85/98 (86.7%) |
| Aggregation | 102 | 97/102 (95.1%) | 96/102 (94.1%) |
| Boolean | 100 | 81/100 (81.0%) | 80/100 (80.0%) |
| Overall | 300 | 261/300 (87.0%) | 261/300 (87.0%) |

The deterministic RRF Aletheia path answered 239/300 (79.7%). Both RRF hybrids score 261/300 (87.0%). The earlier saved hybrids scored 265/300 (88.3%) with Direct fallback and 262/300 (87.3%) with CoT fallback, so neither RRF hybrid improved on its saved counterpart.

## Question-level RRF flips

All rows below include the exact question, source index, gold, and both predictions.

### Saved baseline right; RRF wrong (13)

| Source index | Type | Question | Gold | Saved answer | RRF answer |
|---:|---|---|---|---|---|
| 13401 | temporal | What was the primordial value designation for David Hicks? | Australia | Australia | abstain |
| 6349 | aggregation | What is the sum total of varied values recorded against Andrew Caddick? | 2 | 2 | abstain |
| 2015 | boolean | Was the religion Pyranthism ever a documented fact about Cadwallon ap Cadfan? | False | False | True |
| 7167 | temporal | Before any modifications, what value corresponded to Al-Waleed bin Talal? | Saudi Arabia | Saudi Arabia | abstain |
| 11375 | boolean | Was the value Narwhal Inc. ever a documented fact about Harry Kendall Thaw? | False | False | True |
| 9492 | temporal | In the original records, which value was attributed to Brandon Tartikoff? | NBC | NBC | abstain |
| 9420 | temporal | What was the initial value recorded for FIFPro? | the sport of association football | the sport of association football | abstain |
| 12129 | temporal | Regarding Tom Davis, what was the foundational value first documented? | United States of America | United States of America | abstain |
| 4797 | temporal | What value did the seminal record of Julie Brown contain? | United States of America | United States of America | abstain |
| 11169 | temporal | Regarding Doug Morris, what was the foundational value first documented? | United States of America | United States of America | abstain |
| 11087 | boolean | Was Mirethism at any point logged as the religion for University of Dallas? | False | False | True |
| 7509 | temporal | Regarding Texas Legends, what was the foundational value first documented? | the sport of basketball | the sport of basketball | abstain |
| 8003 | boolean | Has Bractharism been attributed as the religion for Giordano Bruno in any record? | False | False | abstain |

### Saved baseline wrong; RRF right (16)

| Source index | Type | Question | Gold | Saved answer | RRF answer |
|---:|---|---|---|---|---|
| 11912 | boolean | Is it accurate that Windows Communication Foundation was at some point recorded under the value of VMware Inc? | True | False | True |
| 3732 | temporal | In the original records, which value was attributed to Gustav I of Sweden? | Sweden | abstain | Sweden |
| 7314 | temporal | What was the embryonic value information for XML Schema? | World Wide Web Consortium | abstain | World Wide Web Consortium |
| 6251 | boolean | Can you confirm that Spectra Holdings appeared as a value entry for Will Self? | False | abstain | False |
| 2535 | temporal | What value was Dell Inc. first catalogued under? | Michael Dell | abstain | Michael Dell |
| 7866 | temporal | Looking at the earliest entry, which location did The headquarters of Sapienza University of Rome have? | the city of Rome | abstain | the city of Rome |
| 8769 | temporal | Regarding 2015 CAF Confederation Cup, what was the foundational value first documented? | the sport of association football | abstain | the sport of association football |
| 7787 | boolean | Was Dunhaven at any point logged as the country for Merseybeat? | False | abstain | False |
| 5490 | temporal | What archival value was initially assigned to 1983 Cricket World Cup? | the sport of cricket | abstain | the sport of cricket |
| 9792 | temporal | In the original records, which value was attributed to Jamie Redknapp? | the sport of association football | abstain | the sport of association football |
| 6500 | boolean | Does the evidence support that Islam served as the religion for John Witherspoon? | True | False | True |
| 11505 | temporal | What value marked the debut record of MathML? | World Wide Web Consortium | abstain | World Wide Web Consortium |
| 5013 | temporal | What value did the founding record of Jusuf Kalla specify? | Indonesia | abstain | Indonesia |
| 4655 | boolean | Was the creator Quasar Labs ever a documented fact about Bernard Quatermass? | False | abstain | False |
| 8964 | temporal | Which value was the genesis entry for Hans Küng? | University of Tübingen | abstain | University of Tübingen |
| 7781 | boolean | Can you confirm that Meridian Corp appeared as a creator entry for Eeyore? | False | abstain | False |

The flip totals are 13 regressions and 16 improvements. No question was rerun.

## Spend and execution

| Experiment | Questions | API calls | Prompt tokens | Completion tokens | Usage-based spend | Hard stop |
|---|---:|---:|---:|---:|---:|---:|
| RRF Aletheia | 300 | 300 | 600,172 | 20,284 | $0.10219620 | $0.20 |
| ReAct | 300 | 1,709 | 2,134,691 | 31,259 | $0.33895905 | $0.55 |
| **Combined** | **600 arm-question outcomes** | **2,009** | **2,734,863** | **51,543** | **$0.44115525** | **$0.75** |

Spend is computed from the real `usage.prompt_tokens` and `usage.completion_tokens` fields at the frozen rates ($0.15/M input, $0.60/M output). ReAct used 1,409 successful search steps; the other 300 model calls were final-answer turns. Truncated questions: 0. Question errors: 0. Skipped questions: 0. Automatic retries: 0.

### ReAct 10-question cost pilot

Pilot spend was $0.00786570; mean $0.00078657/question; 300-question projection $0.23597100, below the $0.55 gate. The actual full-set spend was $0.33895905.

| Pilot question | Source index | API calls | Search steps | Actual spend |
|---:|---:|---:|---:|---:|
| 1 | 8773 | 7 | 6 | $0.00148005 |
| 2 | 1489 | 7 | 6 | $0.00147120 |
| 3 | 11912 | 3 | 2 | $0.00028185 |
| 4 | 5091 | 2 | 1 | $0.00014865 |
| 5 | 5128 | 7 | 6 | $0.00148110 |
| 6 | 9036 | 7 | 6 | $0.00141645 |
| 7 | 3506 | 2 | 1 | $0.00012960 |
| 8 | 9899 | 6 | 5 | $0.00120000 |
| 9 | 4148 | 2 | 1 | $0.00012690 |
| 10 | 9276 | 2 | 1 | $0.00012990 |

## Frozen setup and deviations

- RRF used the saved question order and exact same final-v3 Aletheia prompt, planner/operator, scorer, model (`gpt-4o-mini`), temperature (0), completion cap (1,024), and K=80. The only intended treatment change was dense retrieval fused with BM25 using equal-weight RRF, constant 60.
- RRF ranking ties were resolved by stable descending score with original corpus order as the tie-break, matching the earlier 89-question RRF test. The original pure-BM25 confirmatory runner uses reverse NumPy argsort for ties; tied positions can therefore differ from the original BM25-only ordering before fusion.
- ReAct is a new arm and necessarily uses the exact saved system prompt in `config/react_system_prompt.txt` plus the frozen `search_memory` tool schema. Each search returns BM25 top 10 from the same 18,332-fact pinned store. The model can issue up to six search calls, followed by a forced final turn if it used all six. A question can use up to seven model calls. ReAct uses a 512-token completion cap per model call; the user fixed model, temperature, scorer, and memory store but did not specify this cap or per-search K, so these choices are disclosed.
- ReAct queries are model-generated from its tool arguments; repeat facts across searches are not deduplicated. Parallel tool calls are disabled so each search step is sequential.
- Direct and CoT were not rerun. Their saved answers on the same 300 questions were reused as comparators and as the RRF hybrid fallback answers.
- The 10-question ReAct pilot was the first ten in the frozen order and is included once in the 300-question total; it was not repeated. Its linear cost projection understated final spend, but the full run remained below budget.
- The 22 audit-flagged questions were retained. No question was excluded, altered, or relabelled.

## Artifact paths

- RRF requests, complete K=80 retrievals, exact prompt/schema, and local input token count: `experiment_1_rrf_all300/requests.jsonl`.
- RRF raw responses, per-call usage, finish reason, full request messages, and spend: `experiment_1_rrf_all300/calls.jsonl`.
- RRF scored rows, flips, hybrids, and summary: `experiment_1_rrf_all300/scored_calls.json`, `hybrid_scored_calls.json`, `summary.json`.
- ReAct exact questions: `experiment_2_react/questions.jsonl`; per-call request messages, outputs, retrieved fact contexts and usage: `experiment_2_react/calls.jsonl`; per-question complete transcripts and actual costs: `experiment_2_react/question_runs.jsonl`.
- Frozen prompt/operator/scorer snapshots and source manifest: `config/frozen_setup/`; run configuration and hashes: `frozen_manifest.json`.
- The pinned Arrow file and model weights were read from the local cache and were not copied into the package.
