# Confirmatory set extension experiments

This local, uncommitted package was created on branch `scratch/extensions` from code commit `ecb373a7e41d92cdcdeebc43c27b6267d662e610`. It does not modify the paper, the saved confirmatory run, or `main`.

`frozen_manifest.json` records the frozen code/configuration hashes, source artifact hashes, question ordering, and pinned FactConsolidation corpus hashes. `config/frozen_setup/` contains byte-for-byte copies of the Aletheia prompt, final-v3 operator, scorer, base operator, and Direct/CoT prompt strings. `experiment_1_rrf_all300/requests.jsonl` stores each question's complete K=80 fused retrieval, exact prompt/schema, and local token count. Its `calls.jsonl` stores the raw model output, request messages, usage fields, finish reason, and per-call spend. Scored rows and the RRF hybrid calculations are saved separately.

The ReAct package stores the frozen system prompt and tool schema in `config/`, the 300 exact questions in `experiment_2_react/questions.jsonl`, and the full tool-call conversation—including each BM25 query and returned facts—in `experiment_2_react/question_runs.jsonl` and `calls.jsonl`. The first ten questions are the cost pilot. The full run command refuses to proceed unless the measured pilot projection is at most $0.55; it reuses those ten and runs only the remaining questions.

## Exact commands

Run from the repository root with the Aletheia Python environment and its configured `OPENAI_API_KEY`:

```sh
python results/extensions/run_extensions.py prepare
python results/extensions/run_extensions.py run-rrf
python results/extensions/score_extensions.py rrf
python results/extensions/run_extensions.py pilot-react
python results/extensions/run_extensions.py run-react-full
python results/extensions/score_extensions.py react
```

`prepare` is offline and refuses to overwrite an existing extension output. Each API action disables SDK retries and checkpoints usage. The experiment hard stops are $0.20 for RRF, $0.55 for ReAct, and $0.75 combined. Do not invoke a run subcommand again after its call ledger exists. Truncations count as failures; errors are not retried. `score_extensions.py` refuses to publish full-sample metrics unless all 300 questions have saved outcomes.

The commands above describe the original execution sequence; they are not a request to rerun it. The saved run ledgers are authoritative. `run-rrf` and the ReAct commands make paid API calls. To reproduce the package from a fresh clone, first restore the saved confirmatory artifacts at their original locations under `results/prof_feedback/pf-20261006-hybrid300-final-01/` and `logs/prof_feedback/`. They provide the frozen sample, Direct/CoT comparisons, and manifests; the runner verifies their hashes. The Task 1 builder also requires `results/prof_feedback/pf-20261004-phaseA-05/gold_support.json`. These feedback-run artifacts and raw logs are not included in this commit.

External data and environment prerequisites:

- The pinned MemoryAgentBench `Conflict_Resolution` Arrow cache must be present at `~/.cache/huggingface/datasets/ai-hyz___memory_agent_bench/default/0.0.0/7ea066982b140a19337e17e60d45d4076e042faf/memory_agent_bench-Conflict_Resolution.arrow`. The frozen manifest records the expected Arrow and extracted-context SHA-256 values. This package does not include the dataset file.
- SentenceTransformers must have the `all-MiniLM-L6-v2` weights cached locally. The RRF preparation uses `local_files_only=True`, so it will not download the model during preparation.
- Use the project environment described by the repository's `requirements.txt`, plus `pyarrow` and `tiktoken` for Arrow loading and exact token counts. Torch is required by SentenceTransformers. Use a local `tiktoken` `o200k_base` encoding cache or allow its one-time tokenizer-file download before running offline.
- An `OPENAI_API_KEY` is needed only to reproduce the paid calls; the saved scoring and report inspection do not need it. Configure the key in the environment, not in these artifacts.

## Scoring and comparison

All arms use `scripts/lib/evaluation_scorer.py`; its exact source snapshot and SHA-256 are in `config/frozen_setup/` and `frozen_manifest.json`. Audit-flagged questions remain in the sample. RRF hybrids use the frozen confirmatory Direct or CoT answer for the same question only when the new RRF Aletheia deterministic path abstains. No hybrid calls are made.

The ReAct search tool returns the top ten BM25 facts per query and can be used at most six times per question. The model then receives a forced final-answer turn if it used all six searches. Per-call usage, prompts, tool results, and completions are retained so the report can be recalculated from the saved artifacts.

The dataset Arrow file and model weights are local prerequisites, not copied into this folder. The loader checks the pinned Arrow and context SHA-256 recorded in the manifest. No raw dataset file is staged here.

## Saved outcomes

| Arm | Historical (98) | Aggregation (102) | Boolean (100) | Overall (300) | Usage spend |
|---|---:|---:|---:|---:|---:|
| Saved Aletheia | 75 (76.5%) | 96 (94.1%) | 40 (40.0%) | 211 (70.3%) | — |
| Aletheia + RRF | 77 (78.6%) | 95 (93.1%) | 42 (42.0%) | 214 (71.3%) | $0.10219620 |
| Saved Direct | 63 (64.3%) | 70 (68.6%) | 87 (87.0%) | 220 (73.3%) | — |
| Saved CoT | 66 (67.3%) | 70 (68.6%) | 87 (87.0%) | 223 (74.3%) | — |
| ReAct | 34 (34.7%) | 49 (48.0%) | 82 (82.0%) | 165 (55.0%) | $0.33895905 |

The RRF hybrids scored 261/300 (87.0%) with either saved Direct or CoT fallback; the corresponding saved hybrids scored 265/300 (88.3%) and 262/300 (87.3%). ReAct's ten-question measured pilot cost was $0.00786570 ($0.00078657/question), projecting $0.23597100 for 300; actual ReAct spend was $0.33895905. Combined experiment spend was $0.44115525. Across the completed runs there were 0 truncations, 0 question errors, 0 skipped questions, and 0 automatic retries. All 22 audit-flagged questions were retained.

See [report.md](report.md) for the full question-level RRF flips, pilot costs, detailed spend/token totals, setup deviations, and artifact inventory. Results and raw call ledgers are saved in the two experiment subdirectories.
