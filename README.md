# Aletheia

Aletheia extends a memory conflict resolution baseline with intent-specific operators. The original pipeline includes an embedding-based semantic router; the final-v3 and confirmatory synthetic runs use an LLM planner and Python operators. In those runs, Aletheia, Direct, and CoT receive the same top-80 BM25 context.

Aletheia builds on [“Don't Ask the LLM to Track Freshness: A Deterministic Recipe for Memory Conflict Resolution” (arXiv:2606.01435)](https://arxiv.org/abs/2606.01435) and extends the original authors' implementation in [cvikasreddy/memory-conflict-resolution](https://github.com/cvikasreddy/memory-conflict-resolution). See the [evaluation report](docs/evaluation_report.md), the [paper draft in this repository](paper/), and the [paper draft Google Doc](https://docs.google.com/document/d/1ZpxQevBxHqhUzbEm7xpUOoYoArBUVXZboZwb9HqpBaA/edit?tab=t.0).

## Reported results

The results below are from the frozen final-v3, confirmatory hybrid, and matched FactConsolidation summaries. See [reproducibility notes](docs/REPRODUCIBILITY.md) for the result files and scripts.

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

### Matched FactConsolidation baseline check (262k; 100 questions)

| System | Accuracy |
| :--- | ---: |
| Adaptive Aletheia saved reference | 81/100 |
| BM25, fixed K=10 | 59/100 |
| BM25, Aletheia-routed K=10/25 | 58/100 |

The BM25 runs use the saved question set, GPT-4o-mini at temperature 0, and `QUERY_TEMPLATE_BM25`. The 81/100 Aletheia figure is the earlier adaptive result; this comparison does not isolate the effect of any single component. The older 81%-versus-56% figures are from a separate, unmatched comparison.

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

See [docs/REPO_MAP.md](docs/REPO_MAP.md) for the top-level directory map and [archive/README.md](archive/README.md) for the archived utility.

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

In final-v3, Python validates the plan, selects and checks candidate facts, and applies the historical, Boolean, or aggregation operator. Historical, aggregation, and negative Boolean answers require verified complete history; positive Boolean evidence can answer when a matching fact is found. Planning and retrieval can still fail. See the [evaluation report](docs/evaluation_report.md) and [paper source](paper/) for the reported analysis.

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

The benchmark tests historical, aggregation, and boolean questions derived from synthetic examples over MemoryAgentBench conflict data. Historical and aggregation answers depend on candidate extraction being complete; misses at that stage can make the deterministic operator return an incomplete answer. The saved result files and the paper's scope limitations are documented in `docs/REPRODUCIBILITY.md` and `paper/`.

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
