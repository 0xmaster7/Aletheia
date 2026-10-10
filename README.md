# Aletheia

Aletheia extends a memory conflict resolution baseline with intent-specific operators. The original pipeline includes an embedding-based semantic router; the final-v3 and confirmatory synthetic runs use an LLM planner and Python operators. In those runs, Aletheia, Direct, and CoT receive the same top-80 BM25 context.

Aletheia builds on [“Don't Ask the LLM to Track Freshness: A Deterministic Recipe for Memory Conflict Resolution” (arXiv:2606.01435)](https://arxiv.org/abs/2606.01435) and extends the original authors' implementation in [cvikasreddy/memory-conflict-resolution](https://github.com/cvikasreddy/memory-conflict-resolution). See the [evaluation report](docs/evaluation_report.md) and [reproducibility notes](docs/REPRODUCIBILITY.md). The Markdown and LaTeX files under `paper/` are archived pre-v20 drafts and are not the current manuscript; the supplied Word/PDF and external document are not distributed from this repository.

## Reported results

The results below are from the frozen final-v3, confirmatory hybrid, and matched FactConsolidation summaries. The final-v3 row is the older headline result; the separate confirmatory and extension evaluations are listed below. See [reproducibility notes](docs/REPRODUCIBILITY.md) for result files and scripts.

### Final-v3 synthetic benchmark (300 questions; audit flags retained)

| Arm | Accuracy |
| :--- | ---: |
| Aletheia | 68.0% (204/300) |
| Direct | 77.3% (232/300) |
| CoT | 78.7% (236/300) |

With flags retained, Aletheia alone is below both baselines overall.

### Confirmatory hybrid run (300 questions; audit flags retained)

| Arm or policy | Accuracy |
| :--- | ---: |
| Aletheia alone | 70.3% (211/300) |
| Direct | 73.3% (220/300) |
| CoT | 74.3% (223/300) |
| Aletheia with Direct fallback | 88.3% (265/300) |
| Aletheia with CoT fallback | 87.3% (262/300) |

The hybrid policies use Aletheia's answer when eligible and otherwise use the named fallback. Their scores are not Aletheia-alone scores.
With flags retained, Aletheia alone is also below both baselines overall in this confirmatory run.
On the 228 questions Aletheia answered, its accuracy was 89.5% (204/228); it abstained on 72/300. This answered-only figure does not replace the overall score.

### Matched FactConsolidation baseline check (262k; 100 questions)

| System | Accuracy |
| :--- | ---: |
| Adaptive Aletheia saved reference | 81/100 |
| BM25, fixed K=10 | 59/100 |
| BM25, Aletheia-routed K=10/25 | 58/100 |

The BM25 runs use the saved question set, GPT-4o-mini at temperature 0, and `QUERY_TEMPLATE_BM25`. The 81/100 Aletheia figure is the earlier adaptive result; this comparison does not isolate the effect of any single component. The older 81%-versus-56% figures are from a separate, unmatched comparison.

### Section VI-F extensions (confirmatory set; GPT-4o-mini, temperature 0)

| Arm or policy | Correct | Accuracy |
| :--- | ---: | ---: |
| Aletheia | 211/300 | 70.3% |
| Direct | 220/300 | 73.3% |
| CoT | 223/300 | 74.3% |
| ReAct | 165/300 | 55.0% |
| RRF Aletheia | 214/300 | 71.3% |
| RRF + Direct fallback | 261/300 | 87.0% |
| RRF + CoT fallback | 261/300 | 87.0% |
| Saved Direct-fallback hybrid | 265/300 | 88.3% |
| Saved CoT-fallback hybrid | 262/300 | 87.3% |

The saved per-intent results are:

| Intent | n | Aletheia | RRF | ReAct | Direct | CoT |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical | 98 | 75 | 77 | 34 | 63 | 66 |
| Aggregation | 102 | 96 | 95 | 49 | 70 | 70 |
| Boolean | 100 | 40 | 42 | 82 | 87 | 87 |
| Overall | 300 | 211 | 214 | 165 | 220 | 223 |

RRF improved 16 questions and regressed 13 against saved Aletheia, for a net gain of three questions (about one percentage point). It reduced both saved hybrid scores: RRF + Direct fallback and RRF + CoT fallback each scored 261/300, versus 265/300 and 262/300 with BM25. RRF therefore did not improve the hybrid results.

For aggregation, original → reworded correct counts were Aletheia 94 → 89, Direct 68 → 65, and CoT 68 → 72. For Boolean they were 40 → 60, 87 → 89, and 87 → 93; the Aletheia increase is unexplained. The 100 historical items were recast as descending-serial-rank questions, changing the operation rather than paraphrasing it. On 98 matched IDs, Aletheia changed 75 → 19, Direct 63 → 48, and CoT 66 → 93. We report no pooled 300-question paraphrase score. The paraphrases were generated once and paired comparisons use only matched question IDs.

The 60 Boolean failures comprised 39 abstentions with no matching target in retrieved context and incomplete history, 4 abstentions despite supporting context, 9 planner-field errors, 8 operator value/span mismatches, and 0 confirmed gold issues. These are classifications from saved contexts and plans: missing target evidence in top-K does not prove absence from source history, and interpreting the evidence-present and planner-conflict cases involves judging question wording. Thirteen failed rows had the `boolean_target_not_found` audit flag; that flag does not establish an incorrect gold label. The polarity fix scored 48/100 on its fresh run, while the isolated same-plan replay changed 42/100 to 48/100 (6 wrong-to-right, 0 right-to-wrong). The fresh planner differed from the saved plan on 41 questions, and the fix was designed on this same 100-question set, so the result is not held-out validation. No hybrid was rerun with the fix.

RRF used BM25 plus `all-MiniLM-L6-v2` dense retrieval, equal-weight reciprocal rank fusion, rank constant 60 and K=80, with the final-v3 prompt/planner/scorer. ReAct used BM25 top-10 search, at most six sequential search steps per question, and a 512-token completion limit; repeated search results were retained. Usage-based spend was $0.10219620 for RRF and $0.33895905 for ReAct ($0.44115525 combined). The rewording evaluation cost $0.25178565, and the Boolean fix run cost $0.03454425 ($0.28632990 combined). All extension calls used GPT-4o-mini at temperature 0; see [the saved extension package](results/extensions/README.md) and [the rewording and Boolean package](results/ood_boolean/README.md) for exact commands, configs, manifests, inputs and limitations.

The extension calls used code commit `ecb373a7e41d92cdcdeebc43c27b6267d662e610`; the saved artifacts and scripts are in commit `4b71da21aa76e3776a00a62f3f52804db04a9e37`. The manifests hold exact prompt, operator, scorer and request hashes. Recompute saved outcomes offline with:

```sh
python results/extensions/score_extensions.py rrf
python results/extensions/score_extensions.py react
python results/ood_boolean/task1/evaluate_ood.py --score
python results/ood_boolean/task2b/run_fix.py --score
```

These offline scoring commands rewrite their summary outputs and require the original local feedback inputs listed in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). Full rerun commands are documented in the result package READMEs and make paid API calls.

For a clean reproduction workspace with the listed local inputs restored, the paid command sequence is:

```sh
python results/extensions/run_extensions.py prepare
python results/extensions/run_extensions.py run-rrf
python results/extensions/score_extensions.py rrf
python results/extensions/run_extensions.py pilot-react
python results/extensions/run_extensions.py run-react-full
python results/extensions/score_extensions.py react
python results/ood_boolean/task1/build_paraphrases.py
python results/ood_boolean/task1/evaluate_ood.py --prepare
python results/ood_boolean/task1/evaluate_ood.py --run
python results/ood_boolean/task1/evaluate_ood.py --score
python results/ood_boolean/task2b/prepare_requests.py
python results/ood_boolean/task2b/run_fix.py --run
python results/ood_boolean/task2b/run_fix.py --score
```

The paid commands are for intentional reproduction only. Do not run them to verify the saved report; the per-question output and usage ledgers already exist.

The full RRF and ReAct artifacts are in [`results/extensions/`](results/extensions/README.md); the reworded evaluation and Boolean diagnosis are in [`results/ood_boolean/`](results/ood_boolean/README.md). These reports preserve the question-level outputs, usage, and setup caveats.

## Repository structure

```text
.
├── archive/                     # Preserved one-off utility and notes
├── benchmarks/latency/          # Routing latency benchmark
├── data/                        # Released synthetic benchmark questions
├── docs/                        # Evaluation, project notes, reproducibility, TODO
├── logs/                        # Dated rerun and benchmark logs
├── paper/                       # Paper sources and Mermaid diagrams
├── results/
│   ├── poc_results/             # Aletheia and paper result JSON
│   └── poc_results_backup_imported/  # Imported baseline result JSON
├── scripts/
│   ├── analysis/
│   ├── data_generation/
│   ├── experiments/
│   ├── lib/                     # Shared pipeline, tracing, and dataset config
│   └── validation/
├── .env.example
├── requirements.txt
└── requirements-lock.txt
```

See [docs/REPO_MAP.md](docs/REPO_MAP.md) for the top-level directory map, [archive/README.md](archive/README.md) for archived utilities, and [`paper/`](paper/) for the archived pre-v20 sources and current diagrams.

## Quickstart

Clone the repository and create the Python 3.10 environment:

```sh
git clone https://github.com/0xmaster7/Aletheia.git
cd Aletheia
conda create --name Aletheia python=3.10 -y
conda activate Aletheia
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` before running experiments. Langfuse tracing is optional; configure its keys in `.env` if you want traces.

## Architecture

The original route in `scripts/lib/_pipeline.py` uses local MiniLM embeddings. Final-v3 and the confirmatory hybrid instead use an LLM planner to choose the intent and plan fields; the embedding router is not used for those runs.

In final-v3, Python validates the plan, selects and checks candidate facts, and applies the historical, Boolean, or aggregation operator. Historical, aggregation, and negative Boolean answers require verified complete history; positive Boolean evidence can answer when a matching fact is found. Planning and retrieval can still fail. See the [evaluation report](docs/evaluation_report.md) and [reproducibility notes](docs/REPRODUCIBILITY.md) for the reported analysis.

## Reproduce results

### No API cost: validate the dataset and saved results

These commands inspect local files and make no model API calls:

```sh
python scripts/validation/validate_dataset.py
python scripts/validation/check_synthetic_benchmark_results.py
```

The second command confirms that the saved synthetic results match the first 60 released questions and that they contain 20 questions per intent.

### Costs API money: rerun experiments

These commands call the OpenAI API. They are documented for reproduction and were not run during this repository cleanup:

```sh
python scripts/experiments/13_paper_experiment.py --source factconsolidation_sh_262k
python scripts/experiments/14_ablations.py --source factconsolidation_sh_262k --task adaptive
python scripts/experiments/run_synthetic_benchmark.py
```

The synthetic runner defaults to the legacy first 60 questions and writes to a separate rerun file. Read [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for pinned dataset/model references and the result-to-script map.

## Synthetic benchmark

The released dataset at `data/synthetic_benchmark.json` contains 13,425 questions. The legacy saved result uses the first 60 questions: 20 historical, 20 aggregation, and 20 boolean. The final paper reports the 300-question final-v3 sample and a separate 300-question confirmatory hybrid sample. The full dataset is released for larger evaluations. Run the legacy first-60 evaluation with `python scripts/experiments/run_synthetic_benchmark.py`; use `--n 13425` or `--all` to evaluate the full set. Each run makes OpenAI API calls. The default output is `results/poc_results/synthetic_benchmark_results_rerun.json`, keeping the committed 60-row result file intact.

Dated experiment reruns are recorded in [`logs/rerun_2026-10-03.log`](logs/rerun_2026-10-03.log), with ablation logs in `logs/ablations/`.

## Latency benchmark

[`benchmarks/latency/test_latency.py`](benchmarks/latency/test_latency.py) measures local routing latency. The committed [`logs/latency/routing_latency_test.log`](logs/latency/routing_latency_test.log) reports a 61.73 ms mean on an Intel i7-8850H. Running the benchmark writes a separate rerun log.

## Limitations

The benchmark tests historical, aggregation, and boolean questions derived from synthetic examples over MemoryAgentBench conflict data. Historical and aggregation answers depend on candidate extraction being complete; misses at that stage can make the deterministic operator return an incomplete answer. The historical rewording evaluation changed the operator form to descending-serial rank and is an operator-shift test, not a pure paraphrase test. The Aletheia Boolean score increase on reworded items is unexplained. The extension and paper-source limitations are documented in `docs/REPRODUCIBILITY.md` and the saved result packages; the archived manuscript drafts are not current scientific claims.

## Citation

```bibtex
@article{reddy2026freshness,
  title         = {Don't Ask the LLM to Track Freshness: A Deterministic Recipe for Memory Conflict Resolution},
  author        = {Reddy, Vikas and Challaram, Sumanth},
  year          = {2026},
  eprint        = {2606.01435},
  archivePrefix = {arXiv}
}
```

Aletheia extends the implementation in [cvikasreddy/memory-conflict-resolution](https://github.com/cvikasreddy/memory-conflict-resolution); see that repository for the original baseline and imported baseline result files.
