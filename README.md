# Aletheia

**A Deterministic Semantic Routing Architecture for Memory Resolution in Long-Context Language Models**

Aletheia separates local semantic intent routing, LLM candidate extraction, and deterministic Python execution. A MiniLM embedding router selects freshness, historical, aggregation, or Boolean intent; BM25 retrieves evidence; GPT-4o-mini extracts candidates; Python applies the corresponding operator.

This README reports the results in the supplied **Aletheia IEEE Access Final** manuscript. The repository also preserves older LLM-planner experiments. Their 68.0%/70.3% standalone and 88.3%/87.3% hybrid results belong to a different pipeline and are not the native-router results reported below. Current evaluation and reproducibility documents follow the native-router paper; superseded planner documents are preserved under `archive/legacy_planner_docs/`, and extension packages retain their own historical records.

The project builds on [Reddy and Challaram's post-retrieval memory assembly work](https://arxiv.org/abs/2606.01435) and the [original memory-conflict-resolution implementation](https://github.com/cvikasreddy/memory-conflict-resolution).

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

The saved local embedding-router benchmark reports a 61.73 ms mean (about 62 ms), 42–259 ms range, and a 395.5 ms cold first call excluded from the mean. It used an Intel Core i7-8850H at 2.60 GHz, one warm-up, and four distinct questions repeated 25 times (100 timed queries). This is routing latency only, not end-to-end pipeline latency. See [the benchmark](benchmarks/latency/test_latency.py) and [saved log](logs/latency/routing_latency_test.log).

## Artifact provenance and local submission

The Desktop checkout contains the following newer native-router packages. They are local/untracked artifacts at the time of this README update; updating this README on GitHub does not publish them. A GitHub source ZIP therefore does not contain all evidence used by the final manuscript. Preserve these folders when preparing the local faculty submission.

| Paper evidence | Local path |
| --- | --- |
| Final own-K scores, 300 raw calls, requests, runner and manifest | `results/prof_feedback/router-final300-native-20261016/` |
| Final shared-K=80 scores, 300 raw calls, requests, runner and manifest | `results/prof_feedback/router-final300-native-k80-20261016/` |
| Second/confirmatory routing replay, confirmatory 300 answers and hybrid scoring | `results/router_runs/pf-20261016-router-confirmatory-01/` |
| Matched FactConsolidation controls | `results/prof_feedback/fc262k-bm25-matched-20261004-01/` |
| FactConsolidation and ablation outputs | `results/poc_results/` and `results/poc_results_backup_imported/` |
| Legacy control outputs and frozen sampled question IDs | `results/prof_feedback/pf-20261004-fresh300-v2-01/`, `pf-20261005-final300-v3-01/`, `pf-20261006-hybrid300-final-01/` |

For final own-K, use `scores_fixed.json` and `paired_summary.json`: the original `scores.json` incorrectly counted 16 Boolean abstentions as correct, yielding 168/300 instead of 152/300. The corrected scores normalize abstentions to `None`. The native-router own-K folder also includes a different evidence-gated hybrid diagnostic; that is not the confirmatory fallback policy in Table VI. The latter is recorded in `results/router_runs/pf-20261016-router-confirmatory-01/r3_summary.json`.

The current [evaluation report](docs/evaluation_report.md), [reproducibility guide](docs/REPRODUCIBILITY.md), and [project summary](docs/Aletheia_Project_Summary.md) follow this native-router paper. Superseded planner documents are in [the archive](archive/legacy_planner_docs/); full pre-v20 manuscript drafts are in `archive/manuscript_drafts/`. The [RRF/ReAct package](results/extensions/README.md) and [rewording/Boolean diagnostics](results/ood_boolean/README.md) preserve separate older experiments. The files `paper/aletheia_paper.md` and `.tex` now point to the supplied authoritative Word manuscript, included locally as `paper/Aletheia_IEEE_Access_Final.docx` in the faculty package.

Full reruns additionally need the pinned MemoryAgentBench revision `7ea066982b140a19337e17e60d45d4076e042faf`, the MiniLM checkpoint `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, and appropriate dataset/model/tokenizer caches. Saved outputs can be reviewed without API credentials. External caches are not part of a normal source-folder ZIP.

## Quickstart and architecture

```sh
git clone https://github.com/0xmaster7/Aletheia.git
cd Aletheia
conda create --name Aletheia python=3.10 -y
conda activate Aletheia
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` only for intentional paid reruns. Langfuse tracing is optional. Never include `.env` or credentials in a faculty ZIP.

The paper's native route is implemented by `native_route()` and `run_adaptive_router_pipeline()` in `scripts/lib/_pipeline.py`: local 384-dimensional MiniLM embeddings select the highest-similarity intent; BM25 uses k1=1.5 and b=0.75; the LLM returns JSON candidates; Python selects the latest/earlier state, deduplicates/counts values, or applies Boolean checks. Historical and aggregation operators sort candidates and cost O(E log E). There is no router rejection threshold. Determinism applies to execution over a fixed candidate representation; it does not eliminate upstream errors.

Separate planner implementations in `scripts/lib/prof_feedback_final.py` and their experiment runners are retained for the older saved evaluations.

## Offline checks and reproduction

These checks make no API calls:

```sh
python scripts/validation/validate_dataset.py
python scripts/validation/check_synthetic_benchmark_results.py
python scripts/validation/check_final_paper_results.py
python scripts/validation/check_paper_operators.py
```

The first two checks validate dataset structure and legacy first-60 correspondence. The final-paper verifier checks frozen source hashes and rescores the saved native runs and hybrids; the operator check exercises actual paper-described functions with a stubbed retrieval/extraction boundary. They make no API calls and do not validate every gold label or replace evaluation on new questions. The benchmark audit flags 684/13,425 rows: 98 potentially invalid gold answers and 586 ambiguous rows.

For saved native-router reproduction, read the local package READMEs, manifests, and `router_eval.py` command help. Offline rescoring requires the saved controls and input files listed above; some older commands point to temporary environments that must be recreated. The confirmatory runner's `score` command makes no API calls; `run` does. Do not run paid experiments merely to inspect saved results.

The legacy runner `scripts/experiments/run_synthetic_benchmark.py` defaults to the earlier first 60 questions and a separate rerun output. Its `--all`/`--n 13425` options call the API and do not recreate the paper's frozen sampled evaluations.

## Limitations

Aletheia does not outperform Direct or CoT on this synthetic benchmark, and fallback hybrids remain below the standalone controls. The benchmark is templated, shares vocabulary with router examples, contains flagged gold labels, and was evaluated only in selected samples with one run per reported cell. Cue lists are manually written. Retrieval, extraction, routing, and normalization can fail, and historical scoring does not verify temporal provenance. The study does not establish production readiness, a general long-context accuracy advantage, or end-to-end latency/cost benefits.
