# Reproducing the final native-router paper

This guide maps the supplied final IEEE Access manuscript to the local code and saved evidence. The manuscript is included in the faculty package at `paper/Aletheia_IEEE_Access_Updated.docx`. Older planner/extension reproduction notes are preserved in [the archive](../archive/legacy_planner_docs/REPRODUCIBILITY.md).

## System and frozen source

The paper uses `scripts/lib/_pipeline.py:native_route` and `run_adaptive_router_pipeline`, not the later GPT-4o-mini query planner in `scripts/lib/prof_feedback_final.py`.

| Item | Pinned reference |
| --- | --- |
| Pipeline SHA-256 | `f5d17c86ad080b52bf0763287f8b2403cc01f5b97a581714b42c5b494cae1f40` |
| Fixed scorer SHA-256 | `ece8986a80efc861b099af31bd4251d4bc835bc5ceec81d0461a64006307448c` |
| Candidate prompt SHA-256 | `aa949eadd4f02e8c8291b18ffbd1ac0980ca629f6a29c446baebfdc0fa0a649f` |
| Experiment source commit | `a3a9c839c6ae9d6758a82f4c459073ec796fc8df` |
| MemoryAgentBench revision | `7ea066982b140a19337e17e60d45d4076e042faf` |
| MiniLM checkpoint | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |

Current pipeline/scorer hashes match these records. The existing local `_lf.py` tracing-disable shim is preserved. The native code uses `current_value` for the paper's freshness route. Its precomputed embedding bank is the original bank before the module's later utterance-dictionary rebind.

Model calls use GPT-4o-mini at temperature 0 in JSON mode. Own-K retrieval uses 10/25/40/10 for freshness/historical/aggregation/Boolean, with historical/aggregation query clean-up. K=80 wrappers use the saved raw-question contexts and saved routes, without rerunning the router.

## Paper table and figure map

All paths below are relative to the repository root.

| Paper item | Saved artifact | Interpretation |
| --- | --- | --- |
| Table I, matched FactConsolidation | `results/prof_feedback/fc262k-bm25-matched-20261004-01/results_summary.json`; adaptive reference `results/poc_results/ablation_adaptive_fact_gpt4omini_factconsolidation_sh_262k.json` | Aletheia 81/100; matched BM25 59/100 and 58/100; original baseline 56/100. |
| Table II, second/confirmatory routing | `results/router_runs/pf-20261016-router-confirmatory-01/r1_summary.json`, `routes_seed_20261004.json`, `routes_seed_20261006.json` | 253/300 and 247/300 routing; no second-sample Aletheia answers were generated. |
| Tables II–III, final own-K | `results/prof_feedback/router-final300-native-20261016/scores_fixed.json`, `paired_summary.json` | 152/300 correct; 251 answered; 49 abstained; one truncated response. |
| Tables II–III, final K=80 | `results/prof_feedback/router-final300-native-k80-20261016/scores.json`, `results_summary.json` | 151/300 correct; 253 answered; 47 abstained; one truncated response. |
| Table IV, ablations | `results/poc_results/ablation_sh_fact_gpt4o_factconsolidation_sh_{6k,32k,64k,262k}.json`, `ablation_sh_chunk4096_gpt4omini_factconsolidation_sh_{6k,32k,64k,262k}.json`, `ablation_mh_fact_gpt4omini_factconsolidation_mh_{6k,32k,64k,262k}.json` | Fact extraction, fixed-window and CAR reruns; CAR uses the separate multi-hop split. |
| Table IV, adaptive and earlier/imported rows | `results/poc_results/ablation_adaptive_fact_gpt4omini_factconsolidation_sh_{32k,64k,262k}.json`; `paper_sh_conflict_factconsolidation_sh_{6k,32k,64k,262k}.json`; imported controls under `results/poc_results_backup_imported/` | No adaptive 6k run. Earlier single-hop 32k/64k and baseline 32k/64k are imported, not author reruns. |
| Table V, confirmatory answers | `results/router_runs/pf-20261016-router-confirmatory-01/r2_answers.jsonl`, `r2_calls.jsonl`, `r2_run_summary.json`, `r3_summary.json` | 152/300 correct; 242 answered; 58 abstained; three 512-token truncations. |
| Table VI, confirmatory hybrids | `results/router_runs/pf-20261016-router-confirmatory-01/r3_hybrid_scored_calls.jsonl`, `r3_summary.json` | Uniform fallback on router abstention: Direct 193/300; CoT 192/300. |
| Figures 1–3 | `paper/diagrams/diagram1_architecture.mmd`, `diagram2_noise_resilience.mmd`, `diagram3_complex_intents.mmd` | Native architecture; mixed-provenance context comparison; final own-K intent comparison. |
| Routing latency | `benchmarks/latency/test_latency.py`, `logs/latency/routing_latency_test.log` | 61.73 ms mean over 100 timed queries; routing only. |

The native result folders contain raw requests/calls, per-question scores and manifests. Supporting Direct/CoT outputs and frozen question sets remain under `results/prof_feedback/pf-20261004-fresh300-v2-01/`, `pf-20261005-final300-v3-01/`, and `pf-20261006-hybrid300-final-01/`. A path such as `{6k,32k,64k,262k}` above denotes four separate filenames.

## Important scoring distinctions

- Own-K's original `scores.json` has an abstention bug: 16 Boolean abstentions were scored as False answers, giving 168/300. Use `scores_fixed.json` for the paper's 152/300.
- Every Boolean no-answer must be passed as `None` to the fixed scorer. The literal text `(no answer)` contains "no" and can otherwise be accepted as False.
- Truncated responses count as incorrect. Final own-K/K=80 cap is 1,024 tokens; confirmatory cap is 512.
- The own-K package's evidence-gated hybrid diagnostic is not Table VI. The latter selects every produced router answer and falls back only on abstention, with no per-intent exceptions.
- SubEM scores FactConsolidation by gold-as-substring. The synthetic fixed scorer accepts Boolean synonyms and exact final aggregation integers; historical matching does not verify temporal provenance.
- Audit flags are retained in headline scores. The benchmark audit flags 684/13,425 rows: 98 potentially invalid gold labels and 586 ambiguous rows.

## Read-only verification without API calls

From the project root with Python 3.10 or later:

```sh
python scripts/validation/validate_dataset.py
python scripts/validation/check_synthetic_benchmark_results.py
python scripts/validation/check_final_paper_results.py
python scripts/validation/check_paper_operators.py
python scripts/validation/check_local_evidence.py
```

The final-paper verifier uses only the standard library plus the included scorer, checks frozen pipeline/prompt hashes, and rescoring reads saved files without overwriting outputs. The operator check extracts the actual pure operator functions from the pipeline and exercises paper-described behavior with a stubbed retrieval/extraction boundary. It does not download MiniLM or call an LLM. Neither check proves robustness on new questions.

The dataset validator is structural; its success message does not validate all gold labels. The first-60 checker verifies legacy correspondence, not current 300-question accuracy.

The saved package scoring scripts are also available, but can rewrite their summary files: `score_router_arm_offline.py`, `run_router_arm_k80.py summary`, and `router_eval.py score`. Use a working copy if rerunning those commands so frozen outputs and hashes are preserved.

## Environment and paid reruns

The baseline environment is Python 3.10.18 with dependencies in `requirements.txt` and `requirements-lock.txt`. Native-run package versions are in their saved `package_versions` files; K=80 also has `requirements-run.txt`, and confirmatory has `router_runtime_requirements.txt`.

```sh
conda create --name Aletheia python=3.10 -y
conda activate Aletheia
pip install -r requirements.txt
cp .env.example .env
```

A full model rerun requires the pinned corpus, MiniLM weights and tokenizer data. They may be downloaded separately; external Hugging Face caches and the Conda environment are not included in a source ZIP. The original fixed-cache Arrow path is documented in the archived guide. Historical temporary environments under `/private/tmp/` no longer exist and must be recreated.

Only intentional paid reruns require `OPENAI_API_KEY`. Inspect each saved runner's command help and README first; preserve the frozen result directories and use a separate reproduction workspace/output. `router_eval.py run` and K=80 `smoke/full` issue API calls. The legacy `run_synthetic_benchmark.py` defaults to the earlier first 60 questions, not the final or confirmatory sampled runs. No new paid runs were made for this documentation alignment.

## Local cost, latency, and cache evidence

[LOCAL_EVIDENCE_AUDIT.md](LOCAL_EVIDENCE_AUDIT.md) records the local ledger, latency, input-hash and cached-file checks. `check_local_evidence.py` verifies included artifacts and checks external caches when available. Add `--require-local-caches` to require byte-level checks of the pinned local model and dataset. External caches are not shipped in the faculty ZIP. The latency source is an aggregate saved log; per-query samples were not retained, so the historical mean is matched to that log rather than independently recomputed.

## Local faculty package versus GitHub

The current Word manuscript and the newer native-run request/call packages are local-only, untracked material. They are included in the faculty ZIP but are not uploaded by the documentation-only GitHub update. The raw planner/extension packages retain their own historical manifests and documentation. [FINAL_PAPER_AUDIT.md](FINAL_PAPER_AUDIT.md) records the submission checks and remaining reproducibility limits.
