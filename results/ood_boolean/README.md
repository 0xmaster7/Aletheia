# OOD and Boolean follow-up artifacts

This directory contains the requested Task 0 audit, Task 1 paraphrase OOD evaluation, and Task 2 Boolean failure analysis/fix run. The complete report is [`report.md`](report.md). Raw prompts, retrieved contexts, responses, usage tokens, and per-question scoring are retained under the corresponding task directories.

## Frozen provenance

- Branch: `scratch/ood-boolean`, created from `scratch/extensions`.
- Code revision at run start: `ecb373a7e41d92cdcdeebc43c27b6267d662e610`.
- Task 1 prompt/operator/scorer/config hashes and run configuration: `task1/frozen_manifest.json`.
- Task 2b policy/operator/request hashes and run configuration: `task2b/frozen_manifest.json`.
- Exact source-question provenance for every paraphrase (all 300, including repeats): `task1/paraphrase_source_ids.csv`; paired-comparison policy and original IDs: `task1/matched_comparison_plan.json`.
- API calls and per-call prompt/completion usage: `task1/calls.jsonl`, `task1/paraphrase_generation_calls*.jsonl`, and `task2b/calls.jsonl`.
- Task 2a uses only saved run files and is detailed in `task2a/failures.json`.

## Reproduction commands

Run from the repository root with the original pinned environment and `OPENAI_API_KEY` configured. `--run` modes make paid API calls. Do not execute them unless intentionally reproducing the calls and budgeted run. Outputs here are the already completed run; rerunning can overwrite local artifacts and cannot guarantee identical model responses.

## External inputs and caches

The raw source artifacts are deliberately not bundled here. Restore these files at the same repository-relative locations before rebuilding requests or repeating the offline audits:

- `results/prof_feedback/pf-20261006-hybrid300-final-01/requests.jsonl`, `scored_calls.json`, and `frozen_manifest.json`, plus `logs/prof_feedback/pf-20261006-hybrid300-final-01/calls.jsonl` for the final-v3 confirmatory sample and its answer/usage records.
- `results/prof_feedback/pf-20261004-phaseA-05/gold_support.json` for the audited labels and source facts used to build the paraphrases.

The pinned MemoryAgentBench `Conflict_Resolution` Arrow cache must also exist at `~/.cache/huggingface/datasets/ai-hyz___memory_agent_bench/default/0.0.0/7ea066982b140a19337e17e60d45d4076e042faf/memory_agent_bench-Conflict_Resolution.arrow`. Expected Arrow and extracted-context SHA-256 values are recorded in the frozen manifests. The local SentenceTransformers cache must contain `all-MiniLM-L6-v2`; exact token counting needs `tiktoken` and its `o200k_base` tokenizer data. Install the packages pinned in `requirements.txt` and the additional runtime dependencies `pyarrow`, `tiktoken`, and `torch` in a compatible Python environment. Paid reproduction additionally requires an `OPENAI_API_KEY` supplied through the environment. Saved-result inspection and offline scoring do not require the key.

Task 1 deterministic sample construction and checks:

```sh
python3 results/ood_boolean/task1/build_paraphrases.py
```

Task 1 request preparation, paid run, then offline scoring:

```sh
python3 results/ood_boolean/task1/evaluate_ood.py --prepare
python3 results/ood_boolean/task1/evaluate_ood.py --run
python3 results/ood_boolean/task1/evaluate_ood.py --score
```

Task 2a is an offline audit of `results/prof_feedback/pf-20261006-hybrid300-final-01`; its saved analysis is already in `task2a/`. Task 2b request preparation, one paid run, and offline scoring:

```sh
python3 results/ood_boolean/task2b/prepare_requests.py
python3 results/ood_boolean/task2b/run_fix.py --run
python3 results/ood_boolean/task2b/run_fix.py --score
```

No tests or new experiments were run during final report assembly. Actual completed run totals: Task 1 $0.25178565 (including $0.00481005 for two rejected paraphrase-generation drafts), Task 2b $0.03454425, combined $0.28632990. See `report.md` for usage-derived accounting and deviations.
