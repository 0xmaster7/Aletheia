# Aletheia

Aletheia extends a deterministic memory conflict resolution baseline with semantic routing and intent specific operators, so queries about history, yes or no facts, and aggregation can follow different execution paths. The implementation pairs semantic-router with local MiniLM embeddings and Python operators, while using an LLM to extract candidate facts where needed.

Aletheia builds on [“Don't Ask the LLM to Track Freshness: A Deterministic Recipe for Memory Conflict Resolution” (arXiv:2606.01435)](https://arxiv.org/abs/2606.01435) and extends the original authors' implementation in [cvikasreddy/memory-conflict-resolution](https://github.com/cvikasreddy/memory-conflict-resolution). See the [evaluation report](docs/evaluation_report.md), the [paper draft in this repository](paper/), and the [paper draft Google Doc](https://docs.google.com/document/d/1ZpxQevBxHqhUzbEm7xpUOoYoArBUVXZboZwb9HqpBaA/edit?tab=t.0).

## Reported results

The tables below are copied from the [evaluation report](docs/evaluation_report.md) and paper draft. See [reproducibility notes](docs/REPRODUCIBILITY.md) for the result files and scripts associated with each paper table and figure.

### Evaluation report

| Context Length | Baseline (BM25) Accuracy | Aletheia (Semantic Router) Accuracy |
| :--- | :--- | :--- |
| **32k** | 70.00% | **77.00%** |
| **64k** | 75.00% | **81.00%** |
| **262k** | 56.00% | **81.00%** |

| Question Intent | Original Author's Repo (Projected) | Aletheia (Semantic Router) | Notes |
| :--- | :--- | :--- | :--- |
| **Historical** (e.g. "What was the initial value?") | **0%** | **15.0%** (3/20) | Original repo fails completely as it blindly returns the newest fact instead of the oldest. |
| **Aggregation** (e.g. "How many unique values?") | **0%** | **10.0%** (2/20) | Original repo fails completely as it only returns one single text string instead of performing a count. |
| **Boolean** (e.g. "Is it true they lived here?") | **0%** | **85.0%** (17/20) | Original repo returns raw fact text rather than evaluating logical True/False. Aletheia's logic gate excels here. |
| **Overall Synthetic Score** | **0%** | **36.6%** (22/60) | |

### Paper tables

**Table 1: System Accuracy Under Escalating Noise Thresholds**

| Context Window | Baseline (BM25) | Aletheia | Delta |
| :--- | :--- | :--- | :--- |
| 32,000 Tokens (Medium) | 70.0% | 77.0% | +7.0% |
| 64,000 Tokens (High) | 75.0% | 81.0% | +6.0% |
| 262,000 Tokens (Extreme) | 56.0% | **81.0%** | **+25.0%** |

**Table 2: Synthetic Benchmark Performance on Multi-Intent Queries**

| Intent Category | Original Baseline (Projected) | Aletheia | Improvement |
| :--- | :--- | :--- | :--- |
| Boolean (Logic) | 0% | 85.0% | +85.0% |
| Historical Tracking | 0% | 15.0% | +15.0% |
| Statistical Aggregation | 0% | 10.0% | +10.0% |

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

The semantic routing layer in `scripts/lib/_pipeline.py` converts each query into a vector with a local embedding model, compares it with semantic anchor phrases, and directs it to a specialized execution pathway. The router uses `semantic-router` with local `sentence-transformers/all-MiniLM-L6-v2` embeddings.

- **Historical and temporal queries:** The original `max(serial)` rule returns the newest fact. Aletheia sorts candidate facts by serial and applies an offset to retrieve the requested earlier record.
- **Boolean queries:** The pipeline finds the latest truth state, then checks whether the query matches that state, with negation handling.
- **Aggregation and counting queries:** The pipeline extracts candidate values across the history, deduplicates them, and applies a count or returns a list.

The LLM extracts candidate facts; Python applies the known operator. The extraction stage remains a source of errors when it misses relevant facts in long contexts. See the [evaluation report](docs/evaluation_report.md) and [paper source](paper/) for the reported analysis.

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

The synthetic runner defaults to the paper's first 60 questions and writes to a separate rerun file. Read [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for pinned dataset/model references and the result-to-script map.

## Synthetic benchmark

The released dataset at `data/synthetic_benchmark.json` contains 13,425 questions. The reported paper evaluation uses the first 60 questions: 20 historical, 20 aggregation, and 20 boolean. The full dataset is released for larger evaluations. Run the default first 60 with `python scripts/experiments/run_synthetic_benchmark.py`; use `--n 13425` or `--all` to evaluate the full set. Each run makes OpenAI API calls. The default output is `results/poc_results/synthetic_benchmark_results_rerun.json`, keeping the committed 60-row result file intact.

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
