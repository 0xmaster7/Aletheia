# Local evidence audit for the revised paper

Checked locally on 2026-10-28. The authoritative manuscript for this revision is `paper/Aletheia_IEEE_Access_Updated.docx`, copied unchanged from the local Downloads version. Its title is:

**Aletheia: A Semantic Routing Architecture with a Deterministic Operator Layer for Memory Resolution in Long-Context Language Models**

Manuscript SHA-256: `b70661996ae3541a5a3c55d5af5f6c351dcf76c3c05702cdcfe6e8f5184fc477`.

The earlier supplied `Final (1)` manuscript has an older title. The revised version retains the same six numerical tables and adds explanations of bare ordinal fallback, string-only deduplication, missing-evidence outputs, benchmark auditing, string-length costs, and fixed negation cues. The README and current guides now follow this revised wording. The older Word file is preserved locally under `archive/manuscript_drafts/`.

## Historical API costs

Each row below has 300 completed calls. Saved `actual_cost_usd` values were summed and independently checked against the recorded input/output token counts using the frozen run's $0.15/$0.60 per million token rates. These are historical accounting figures, not current API price claims.

| Arm | Usage-derived USD | Paper / README rounded USD | Local call ledger |
| --- | ---: | ---: | --- |
| Final own-K router | 0.04025535 | 0.040 | `results/prof_feedback/router-final300-native-20261016/calls.jsonl` |
| Final K=80 router | 0.07790085 | 0.078 | `results/prof_feedback/router-final300-native-k80-20261016/calls.jsonl` |
| Confirmatory router | 0.07801065 | 0.078 | `results/router_runs/pf-20261016-router-confirmatory-01/r2_calls.jsonl` |
| Final Direct | 0.05853960 | 0.059 | `logs/prof_feedback/pf-20261005-final300-v3-01/calls.jsonl`, Direct arm |
| Final CoT | 0.07143225 | 0.071 | Same ledger, CoT arm |

The token-derived and recorded totals agree within floating-point storage precision. Router model-usage cost excludes local CPU work, model loading, hardware, and deployment cost. No new paid calls were made for this audit.

## Latency evidence

`logs/latency/routing_latency_test.log` records the following on an Intel Core i7-8850H at 2.60 GHz:

| Quantity | Saved value |
| --- | ---: |
| Timed queries | 100 |
| Mean | 61.73 ms |
| Minimum | 42.22 ms |
| Maximum | 259.32 ms, iteration 4 |
| Median | 52.44 ms |
| First warm-up inference, excluded | 395.50 ms |

`benchmarks/latency/test_latency.py` contains four distinct questions repeated 25 times, with the warm-up outside the timed loop. The model is initialized during module import before that warm-up timer, so the recorded warm-up is not a measurement of total cold model startup. This is routing-only latency. Per-query timing samples were not retained, so the aggregate statistics are verified against the saved log, not independently recomputed or reproduced on a new benchmark run.

## Hashes, cached inputs, and paths

- The current pipeline, fixed scorer, and candidate prompt match their frozen native-run hashes, as checked by `check_final_paper_results.py`.
- All 17 included input-file hashes in the confirmatory manifest match the local files, including source, scorer, tracing shim, sampled inputs, saved requests, controls, router replay, runtime requirements and tokenizer data.
- The local MiniLM cache ref is checkpoint `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; all 12 snapshot-file hashes in the confirmatory manifest match the cached bytes, including model weights and tokenizer files.
- The local MemoryAgentBench Arrow cache is under revision `7ea066982b140a19337e17e60d45d4076e042faf`, and its SHA-256 matches `de3c70189935b532d21707ca92ae4ee835cc335cd0c002b803355fdb16e5b474` in the own-K manifest.
- Required result/control paths exist locally; current documentation links resolve locally. New native-run result packages and the revised Word manuscript remain local-only and are included in the faculty ZIP. A GitHub source download alone is incomplete for saved-evidence review.

The model/dataset caches were verified on this machine but are not included in the faculty ZIP. The included manifest hashes document them; a reviewer must obtain those external resources separately for a full model rerun.

## Runnable checks and scope

```sh
python scripts/validation/check_final_paper_results.py
python scripts/validation/check_paper_operators.py
python scripts/validation/check_local_evidence.py
```

On a machine with the original caches, add `--require-local-caches` to the final command to require their byte-level verification. Without caches it still verifies the included ledgers, log, manuscript title and frozen input files, and explicitly reports external caches as absent.

These checks establish consistency with the supplied/revised manuscript and saved local evidence. They do not independently validate every dataset gold label, cited literature claim, or fresh model output; no new full evaluation or latency experiment was run. Historical planner and extension artifacts remain historical rather than being relabeled as native-router results.
