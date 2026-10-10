# Reproducibility

## Experiment generations and provenance

The repository contains two distinct Aletheia paths. The original design uses a local MiniLM embedding router and adaptive route K. Final-v3, the confirmatory sample, and the Section VI-F extensions use GPT-4o-mini as an LLM planner over the same BM25 top-80 context given to Direct and CoT; the embedding router is not used to select their routes.

The saved extension calls and configuration were produced from code commit `ecb373a7e41d92cdcdeebc43c27b6267d662e610`, recorded by the extension manifests. The per-question extension artifacts and scripts were committed at `4b71da21aa76e3776a00a62f3f52804db04a9e37`; the interpretation/docs update was committed at `6a11c1d9f830ed01efa434d936daafc5fa097059`. SHA-256 hashes for prompts, operators, scorer, request/config files, and model inputs are in each `frozen_manifest.json`. The manifests identify file contents and the source code commit; they do not claim the calls can be recreated bit-for-bit.

The results are one-run evaluations. Section VI-F retains all 22 audit-flagged questions in the 300-question confirmatory sample. The paraphrase task retains its source IDs and flags. Historical serial-rank items change the requested operation and are reported separately as an operator-shift test. The Boolean polarity fix was designed and evaluated on the same 100 Boolean questions; it is a same-sample diagnostic, not held-out validation.

## Dataset, models and environment

The direct dependency pins in `requirements.txt` were read from the repository's Aletheia Conda environment on 2026-10-03 (Python 3.10.18); `requirements-lock.txt` contains its full `pip freeze` output. Install with:

```sh
conda create --name Aletheia python=3.10 -y
conda activate Aletheia
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` only when intentionally making paid calls. Langfuse tracing is optional and reads `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST`.

All project `load_dataset` calls use `DATASET_REVISION` in `scripts/lib/config.py`, pinned to `7ea066982b140a19337e17e60d45d4076e042faf` of [ai-hyz/MemoryAgentBench](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/tree/7ea066982b140a19337e17e60d45d4076e042faf), checked 2026-10-03. Existing split choices are unchanged. The original router model names are `all-MiniLM-L6-v2` and `sentence-transformers/all-MiniLM-L6-v2`; final-v3 and extensions use the `gpt-4o-mini` planner. GPT-4o appears in the FactConsolidation single-hop reruns only.

## Paper result to saved artifact map

| Paper result | Saved per-question result / summary | Script or source | Reconciliation note |
| --- | --- | --- | --- |
| Table I, FactConsolidation context-length comparison | `results/poc_results/paper_sh_conflict_factconsolidation_sh_6k.json`, `paper_sh_conflict_factconsolidation_sh_32k.json`, `paper_sh_conflict_factconsolidation_sh_64k.json`, `paper_sh_conflict_factconsolidation_sh_262k.json`; adaptive results are `ablation_adaptive_fact_gpt4omini_factconsolidation_sh_32k.json`, `_64k.json`, `_262k.json` in the same folder | `scripts/experiments/13_paper_experiment.py`; `scripts/experiments/14_ablations.py --task adaptive` | Mixed historical provenance, including baseline source results from [8]. It is not the matched comparison. |
| Table II, matched FactConsolidation baselines | `results/prof_feedback/fc262k-bm25-matched-20261004-01/results_summary.json`; saved adaptive reference in `results/poc_results/` | `scripts/analysis/factconsolidation_matched_baseline.py` | 81/100 adaptive Aletheia vs 59/100 fixed-K BM25 and 58/100 routed-K BM25. Uses GPT-4o-mini, temperature 0, same question set and unchanged `QUERY_TEMPLATE_BM25`; the routed-K control matches Aletheia's per-question K=10/25, while fixed K=10 is a separate control. |
| Table III, staged synthetic evaluation | `results/poc_results/synthetic_benchmark_results.json`; first-campaign result/plan artifacts under `results/prof_feedback/`; later stage summaries there | `scripts/experiments/run_synthetic_benchmark.py`; `scripts/analysis/fresh300_final_v3.py` | First-60 is 22/60 (36.7%) under the earlier pipeline and original scorer, with no baseline. Stage 1 has plan/candidate checks, not Aletheia end-to-end answer accuracy. The first campaign's Direct/CoT numbers use the original soft matcher. |
| Tables IV–V, final-v3 accuracy and same-item answered/abstained comparison | `results/prof_feedback/pf-20261005-final300-v3-01/results_summary.json` (tracked) | `scripts/analysis/fresh300_final_v3.py` | 204/300 Aletheia, 232/300 Direct, 236/300 CoT; Aletheia answered 228/300 at 89.5%. Per-question `requests.jsonl`, `scored_calls.json`, and `logs/prof_feedback/pf-20261005-final300-v3-01/calls.jsonl` are present only in this local checkout, not the public branch. |
| Table VI, ablation runs | GPT-4o fact-level: `ablation_sh_fact_gpt4o_factconsolidation_sh_6k.json`, `_32k.json`, `_64k.json`, `_262k.json`; 4096-character windows: `ablation_sh_chunk4096_gpt4omini_factconsolidation_sh_6k.json`, `_32k.json`, `_64k.json`, `_262k.json`; CAR: `ablation_mh_fact_gpt4omini_factconsolidation_mh_6k.json`, `_32k.json`, `_64k.json`, `_262k.json`; adaptive: `ablation_adaptive_fact_gpt4omini_factconsolidation_sh_32k.json`, `_64k.json`, `_262k.json`; all paths are under `results/poc_results/`. The dated log is `logs/rerun_2026-10-03.log`. | `scripts/experiments/13_paper_experiment.py`; `scripts/experiments/14_ablations.py` | The old single-hop row is mixed provenance and its local 6k/262k results are in the four `paper_sh_conflict_factconsolidation_sh_{6k,32k,64k,262k}.json` files listed for Table I; the paper marks 32k/64k sources as [8]. |
| Tables VII–IX, confirmatory sample, hybrids, matched answered/abstained results | `results/prof_feedback/pf-20261006-hybrid300-final-01/results_summary.json` and `frozen_manifest.json` (tracked) | `scripts/analysis/final_hybrid_confirmatory.py` | Separate 300-question sample; all flags retained. Standalone Aletheia 211/300; Direct fallback hybrid 265/300; CoT fallback hybrid 262/300. `requests.jsonl`, `scored_calls.json`, and `logs/prof_feedback/pf-20261006-hybrid300-final-01/calls.jsonl` are local-only. |
| Table X, ReAct | `results/extensions/experiment_2_react/summary.json`, `scored_calls.json`, `question_runs.jsonl`, `calls.jsonl` | `results/extensions/run_extensions.py`; `results/extensions/score_extensions.py` | 165/300 (55.0%): historical 34/98, aggregation 49/102, Boolean 82/100. |
| Table XI, RRF and fallback policies | `results/extensions/experiment_1_rrf_all300/summary.json`, `scored_calls.json`, `hybrid_scored_calls.json`, `requests.jsonl`, `calls.jsonl` | `results/extensions/run_extensions.py`; `results/extensions/score_extensions.py` | RRF 214/300 versus saved Aletheia 211/300, with 16 gains and 13 regressions. RRF fallback hybrids both score 261/300, below the saved BM25 hybrids (265/300 Direct fallback; 262/300 CoT fallback). |
| Table XII, aggregation/Boolean rewordings and historical operation shift | `results/ood_boolean/task1/results_summary.json`, `scored_calls.json`, `paraphrases.jsonl`, `paraphrase_source_ids.csv`, `matched_comparison_plan.json` | `results/ood_boolean/task1/build_paraphrases.py`; `results/ood_boolean/task1/evaluate_ood.py --score` | Aggregation and Boolean are paired rewordings; historical serial-rank questions change the requested operation. Only 298 matched source IDs are compared; the historical row is not a paraphrase score and is not pooled into a 300-item paraphrase result. |
| Section VI-F Boolean failure diagnosis and polarity fix | `results/ood_boolean/task2a/failures.json`; `results/ood_boolean/task2b/results_summary.json`, `plans_and_scores.jsonl`, `calls.jsonl` | `results/ood_boolean/task2b/run_fix.py --score` | Counts: 39 insufficient-evidence/incomplete-history abstentions, 4 abstentions despite evidence, 9 planner-field errors, 8 operator value/span mismatches, 0 confirmed gold issues. Same-plan replay 42/100→48/100 (six gains, zero regressions) is same-sample evidence. |
| Figure 1, architecture | `paper/diagrams/diagram1_architecture.mmd` | Mermaid source; no renderer script found | Updated to label the embedding-router diagram as the original design and avoid a zero-hallucination claim. |
| Figure 2, context-length comparison | `paper/diagrams/diagram2_noise_resilience.mmd`; Table I source rows above | Mermaid source; no renderer script found | Historical/mixed-provenance comparison, explicitly distinguished from Table II's matched BM25 runs. |
| Figure 3, final-v3 intent accuracy | `paper/diagrams/diagram3_complex_intents.mmd`; Table IV source above | Mermaid source; no renderer script found | Updated to the final-v3 per-intent scores; not the first-60 85/15/10 result. |

The pre-v20 Markdown and LaTeX are labeled as archived superseded drafts. The current Word/PDF manuscript is not copied into this repository. The older drafts' author identifiers and author blocks have been removed from the archived copies. No current paper author list is published here.

## Section VI-F setup and exact commands

Saved settings:

- ReAct: GPT-4o-mini, temperature 0, BM25 top 10 per search, maximum six sequential search steps per question, 512 completion tokens; repeated results were retained.
- RRF: GPT-4o-mini, temperature 0, BM25 plus local `all-MiniLM-L6-v2` dense retrieval, equal rank weights, rank constant 60 and K=80. The final-v3 prompt, planner and scorer were kept unchanged.
- Rewording arms: GPT-4o-mini, temperature 0 and final-v3 scorer. The question set was built once and frozen. Historical questions were deliberately recast as descending-serial-rank questions.
- Boolean polarity fix: evaluation-only helper, all 100 confirmatory Boolean questions, no prompt/planner/scorer changes; 0 retries. Designed on the evaluation set.

To recompute the saved RRF and ReAct score files (offline; these commands rewrite their score/summary JSON outputs):

```sh
python results/extensions/score_extensions.py rrf
python results/extensions/score_extensions.py react
```

To recompute saved rewording and Boolean-fix scores (offline; these commands rewrite score/summary JSON outputs):

```sh
python results/ood_boolean/task1/evaluate_ood.py --score
python results/ood_boolean/task2b/run_fix.py --score
```

The exact paid-run commands are recorded in [`results/extensions/README.md`](../results/extensions/README.md) and [`results/ood_boolean/README.md`](../results/ood_boolean/README.md). RRF/ReAct commands are `python results/extensions/run_extensions.py prepare`, `run-rrf`, `pilot-react`, and `run-react-full`; the OOD sequence is `python results/ood_boolean/task1/build_paraphrases.py`, `python results/ood_boolean/task1/evaluate_ood.py --prepare`, then `--run`; the Boolean fix uses `python results/ood_boolean/task2b/prepare_requests.py` and `python results/ood_boolean/task2b/run_fix.py --run`. The run commands make paid API calls and must not be invoked just to inspect this repository.

## Inputs not included in the public branch

Saved extension outcomes and request/call ledgers are committed. The final-v3 and confirmatory summary/manifests are committed, but exact rescoring and reruns also need original feedback inputs that remain local and untracked, plus the pinned corpus/model caches. They were not copied into the public file set because they contain raw request/call material or source data. The local-only feedback paths are:

- `results/prof_feedback/pf-20261005-final300-v3-01/requests.jsonl`
- `results/prof_feedback/pf-20261005-final300-v3-01/scored_calls.json`
- `logs/prof_feedback/pf-20261005-final300-v3-01/calls.jsonl`
- `results/prof_feedback/pf-20261006-hybrid300-final-01/requests.jsonl`
- `results/prof_feedback/pf-20261006-hybrid300-final-01/scored_calls.json`
- `logs/prof_feedback/pf-20261006-hybrid300-final-01/calls.jsonl`
- `results/prof_feedback/pf-20261004-phaseA-05/gold_support.json` (required to rebuild the frozen reworded requests)

The MemoryAgentBench `Conflict_Resolution` Arrow cache must be available at `~/.cache/huggingface/datasets/ai-hyz___memory_agent_bench/default/0.0.0/7ea066982b140a19337e17e60d45d4076e042faf/memory_agent_bench-Conflict_Resolution.arrow`; the manifests record expected Arrow/context hashes. RRF preparation also needs cached `all-MiniLM-L6-v2` weights and `torch`; request preparation uses `pyarrow`, `tiktoken` and the local `o200k_base` tokenizer cache. No raw dataset file, private author detail, API key or `.env` is required for reviewing the committed summaries.

## Legacy benchmark and no-API checks

`data/synthetic_benchmark.json` has 13,425 questions. The legacy saved result is its first 60 entries (20 per intent); the result file uses the original soft-match scorer and is separate from final-v3. The legacy runner defaults to 60 and writes a different output file. `--n 0`, `--all`, or `--n 13425` selects the full dataset and calls the API.

Offline validation commands:

```sh
python scripts/validation/validate_dataset.py
python scripts/validation/check_synthetic_benchmark_results.py
```

The committed dataset was not regenerated. An offline replay of the current incremental generators differed from the committed dataset in 2,803 rows; the committed dataset remains the evaluation input. The repository does not claim bit-for-bit dataset regeneration.

## Legacy reruns and latency

The dated output is `logs/rerun_2026-10-03.log`; ablation logs are under `logs/ablations/`. The routing benchmark is `benchmarks/latency/test_latency.py`, with saved output `logs/latency/routing_latency_test.log` (61.73 ms mean on an Intel i7-8850H). This belongs to the original local embedding-router path, not final-v3's LLM-planner evaluation.
