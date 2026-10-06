# Task 2a — confirmatory Boolean failures (offline replay)

Source: `pf-20261006-hybrid300-final-01`; the 100 saved Boolean items contain 60 failures. The final-v3 operator replayed on the saved plans and K=80 contexts reproduces all saved predictions and completeness flags. No API call was made for this audit.

| Requested cause category | Count | Three example question IDs | What the saved files establish |
|---|---:|---|---|
| Abstained because retrieved evidence was insufficient to verify a negative across the entity's full history | 39 | `syn-09899-3f14162ba310`, `syn-11063-bf793cd201cf`, `syn-09659-189213c324d3` | These are gold-False questions with no matching fact in K=80 and an incomplete entity history. The operator correctly abstained under its completeness gate; the files do not prove those gold labels wrong. |
| Abstained despite relevant evidence in context | 4 | `syn-00572-6f84a5a47369`, `syn-10046-e215928c5651`, `syn-09530-bd882ed8cd94` | One target span differed (`Liverpool` vs `the city of Liverpool`); three plans reversed subject and target. The full fourth ID is `syn-03674-49cb83a91bbc`. |
| Planner field error (polarity or target) | 9 | `syn-11912-2204b0185efd`, `syn-03875-f9c8d8347a7d`, `syn-09020-1298312fd46f` | Eight affirmative questions had `negated=true`; one plan used generic target `value` instead of the named target. There were no intent misroutes among the 60 failures (or the 100 Boolean items). |
| Wrong operator/value matching | 8 | `syn-04184-a4359bec961f`, `syn-01766-da75c32981e2`, `syn-08222-63267832a055` | A matching fact was retrieved but exact target/value equality failed because the fact value included a relation phrase, such as `the sport of baseball`. |
| Confirmed gold-answer issue | 0 | None | Thirteen failures carry the audit code `boolean_target_not_found`; that is a warning flag, not evidence by itself that the expected answer is wrong. Three flagged examples: `syn-00155-8b9dcae3aa49`, `syn-04055-b1ba570be537`, `syn-03035-ac55f5922583`. No failure had `boolean_label_disagrees_with_source_support`. |

Counts sum to 60: 43 abstentions and 17 wrong answered predictions. Saved files cannot establish full-history absence beyond the local source audit, nor whether an audit-flagged expected label is actually incorrect. No second cause was inferred from missing evidence.

## Single fix preregistered for Task 2b

The one specific planner-field issue selected before Task 2b is the eight `negated=true` plans on affirmative-existence questions. The evaluation-only fix sets `negated=false` only when the question contains none of the explicit whole-word markers `not`, `never`, or `no`. It leaves prompts, planner, scorer, cue lists, retrieval, and every other operator rule unchanged. It is applied to all 100 Boolean items, including already-correct cases, so regressions remain visible. See `../task2b/config/fix_policy.txt` and `../task2b/config/boolean_polarity_fix.py`.

Full saved questions, plans, retrieved facts, replays, and scores are in `failures.json`.
