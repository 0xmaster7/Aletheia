# Reproducibility

## Environment and dependencies

The pinned direct dependencies in `requirements.txt` were read from the repository's Aletheia Conda environment on 2026-10-03. That environment runs Python 3.10.18. `requirements-lock.txt` is its full `pip freeze` output, including transitive packages.

Create the environment and install the direct pins with:

```sh
conda create --name Aletheia python=3.10 -y
conda activate Aletheia
pip install -r requirements.txt
cp .env.example .env
```

Fill in `OPENAI_API_KEY`. Langfuse tracing reads `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST`; those keys are optional for runs that do not need tracing.

## Dataset and model revisions

All project `load_dataset` calls use the shared `DATASET_REVISION` constant in `scripts/lib/config.py`, currently pinned to commit `7ea066982b140a19337e17e60d45d4076e042faf` of [ai-hyz/MemoryAgentBench](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/tree/7ea066982b140a19337e17e60d45d4076e042faf). This revision was checked on 2026-10-03. The scripts continue to load their existing splits; the revision pin only freezes the dataset snapshot.

The model names used by the code are:

- Router encoder: `all-MiniLM-L6-v2` in `scripts/lib/_pipeline.py`.
- Semantic router model setting: `sentence-transformers/all-MiniLM-L6-v2` (`ROUTER_EMBED_MODEL`).
- Default pipeline LLM: `gpt-4o-mini` (`PIPELINE_MODEL`).
- The gpt-4o ablation sets `PIPELINE_MODEL=gpt-4o`.

## Paper tables and figures

| Paper item | Exact result source | Producing script | Provenance |
| --- | --- | --- | --- |
| Table 1, baseline values | `results/poc_results/paper_sh_conflict_factconsolidation_sh_32k.json`; `results/poc_results/paper_sh_conflict_factconsolidation_sh_64k.json`; `results/poc_results/paper_sh_conflict_factconsolidation_sh_262k.json` | `scripts/experiments/13_paper_experiment.py` | The JSONs contain the corresponding per-question BM25 records and accuracies. |
| Table 1, Aletheia values | `results/poc_results/ablation_adaptive_fact_gpt4omini_factconsolidation_sh_32k.json`; `results/poc_results/ablation_adaptive_fact_gpt4omini_factconsolidation_sh_64k.json`; `results/poc_results/ablation_adaptive_fact_gpt4omini_factconsolidation_sh_262k.json` | `scripts/experiments/14_ablations.py --task adaptive` | The JSONs contain the corresponding per-question Aletheia records and accuracies. |
| Table 2, synthetic multi-intent results | `results/poc_results/synthetic_benchmark_results.json` | `scripts/experiments/run_synthetic_benchmark.py`; input is `data/synthetic_benchmark.json` | The saved result rows match the first 60 released questions. The projected original-baseline row has no per-question result file: **source not found**. |
| Figure 1, architecture | `paper/diagrams/diagram1_architecture.mmd` and implementation in `scripts/lib/_pipeline.py` | No figure-generation script found | Conceptual diagram; no result JSON applies. |
| Figure 2, context-length performance | `paper/diagrams/diagram2_noise_resilience.mmd`; data sources are the six Table 1 files above | No figure-generation script found | Chart values trace to Table 1 sources. |
| Figure 3, intent performance | `paper/diagrams/diagram3_complex_intents.mmd`; data source is `results/poc_results/synthetic_benchmark_results.json` | No figure-generation script found | Chart values trace to the Table 2 source. |

The repository includes no script that renders the Mermaid diagrams; the `.mmd` files are the source diagrams. No numeric source is inferred for the figures beyond their matching table/result files.

## Synthetic benchmark protocol

`data/synthetic_benchmark.json` contains 13,425 questions. The paper's saved evaluation uses its first 60 entries, with 20 questions for each intent. The released full set is available for larger evaluations. Run the no-API consistency check with:

```sh
python scripts/validation/check_synthetic_benchmark_results.py
```

The benchmark runner defaults to those first 60 and saves a new run to `results/poc_results/synthetic_benchmark_results_rerun.json`. `--n 0` or `--all` selects the full set. `--n 13425` also selects all currently released questions. Every execution of the runner makes OpenAI API calls; its `--help` option does not load the model or call an API.

`scripts/experiments/run_synthetic_eval.py` remains a separate 100-query auxiliary comparison. It is not the 60-question synthetic result reported in the paper.

## Reruns and latency

The dated rerun outputs and console log are preserved in `logs/rerun_2026-10-03.log`; the ablation logs are in `logs/ablations/`. The routing latency benchmark is `benchmarks/latency/test_latency.py`, and its recorded output is `logs/latency/routing_latency_test.log`. The existing log reports an average routing latency of 61.73 ms on an Intel i7-8850H. Running the latency benchmark writes a separate `routing_latency_test_rerun.log`.

## Commands by cost

No-API checks:

```sh
python scripts/validation/validate_dataset.py
python scripts/validation/check_synthetic_benchmark_results.py
```

API-costing runs (these commands are documented only; they were not run during cleanup):

```sh
python scripts/experiments/13_paper_experiment.py --source factconsolidation_sh_262k
python scripts/experiments/14_ablations.py --source factconsolidation_sh_262k --task adaptive
python scripts/experiments/run_synthetic_benchmark.py
```

## Generator compatibility note

The committed `data/synthetic_benchmark.json` was not regenerated. Sorting the initial generator's set-derived `all_values` changed four boolean questions in its 60-question output, so that one conversion intentionally retains its prior set iteration to preserve the committed questions. The batch generators and entity extraction sort their set-derived values; isolated sorted and unsorted batch output matched for the tested 40-entity batch under a fixed hash seed.

A full offline replay through the current incremental scripts produced 13,425 rows but differed from the committed file in 2,803 rows. The original batch execution and random-seed sequence are not recoverable from the committed history, so this replay does not establish that those differences are caused by sorting. Keep the released data file as the source of the reported run; do not replace it with generated output.
