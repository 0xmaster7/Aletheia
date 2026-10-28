# Repository map

## Current paper and review entry points

- `README.md`: current final-paper results and system overview.
- `docs/evaluation_report.md`: Tables I–VI and their interpretation.
- `docs/REPRODUCIBILITY.md`: current artifact map, frozen hashes, and offline checks.
- `docs/Aletheia_Project_Summary.md`: current architecture and main findings.
- `docs/FINAL_PAPER_AUDIT.md`: code/results alignment and faculty-package scope.
- `paper/Aletheia_IEEE_Access_Updated.docx`: authoritative supplied manuscript, included locally in the faculty ZIP.
- `paper/aletheia_paper.md` and `paper/aletheia_paper.tex`: current manuscript pointers; the old full drafts are archived.
- `paper/diagrams/`: current native-router architecture and result diagrams.

## Source and saved evidence

- `scripts/lib/_pipeline.py`: native MiniLM routing, BM25, extraction, operators, and other historical pipeline helpers.
- `scripts/lib/evaluation_scorer.py`: frozen fixed scorer.
- `scripts/lib/prof_feedback_final.py`: separate older LLM-planner implementation.
- `scripts/experiments/`: original experiment drivers, including the legacy first-60 runner.
- `scripts/analysis/`: historical planner/control/matched-baseline runners and diagnostics.
- `scripts/validation/`: dataset, saved-result, and operator checks.
- `data/synthetic_benchmark.json`: released 13,425-question benchmark.
- `results/prof_feedback/router-final300-native-20261016/`: local own-K native evaluation.
- `results/prof_feedback/router-final300-native-k80-20261016/`: local shared-K native evaluation.
- `results/router_runs/pf-20261016-router-confirmatory-01/`: local routing replay, confirmatory evaluation and hybrids.
- `results/poc_results/` and `results/poc_results_backup_imported/`: FactConsolidation, ablations, legacy and imported results.
- `results/prof_feedback/pf-*/`: frozen sampled question sets and older control/planner evidence.
- `results/extensions/` and `results/ood_boolean/`: older planner extensions, preserved with their own manifests.
- `benchmarks/latency/` and `logs/latency/`: routing-only latency benchmark and saved log.
- `logs/`: saved execution and usage logs.

## Historical material

- `archive/legacy_planner_docs/`: superseded evaluation, reproduction and summary documents.
- `archive/manuscript_drafts/`: original pre-v20 Markdown/LaTeX drafts with obsolete claims.
- `archive/`: other reviewed one-off utilities.

The local faculty ZIP includes newer local-only results and the Word manuscript; a GitHub source download does not. Dependency files and `.env.example` support setup, while credentials, Git metadata and external caches are excluded.
