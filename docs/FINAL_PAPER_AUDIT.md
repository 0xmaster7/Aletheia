# Final paper and local submission audit

Checked on 2026-10-28 against the revised local `Aletheia_IEEE_Access_Updated.docx`. Its title scopes determinism to the operator layer; its six numerical tables match the earlier supplied attachment. The earlier manuscript is preserved locally under `archive/manuscript_drafts/Aletheia_IEEE_Access_Final_older_title.docx`.

## Status

The Desktop checkout has GitHub remote `https://github.com/0xmaster7/Aletheia.git`. The frozen native experiments used source commit `a3a9c839c6ae9d6758a82f4c459073ec796fc8df`; the current documentation describes those native runs. The working folder also contains newer untracked native-router experiment artifacts and a pre-existing change in `scripts/lib/_lf.py`; those local files are not implied by the GitHub commit history.

The README, evaluation report, reproducibility guide, project summary, repository map, TODO list, and Mermaid diagrams now follow the supplied paper's native-router results. The current Word paper is included locally as `paper/Aletheia_IEEE_Access_Updated.docx`. It is copied unchanged from the revised local Downloads manuscript and is not published by the documentation/code-verifier GitHub update.

## Checks performed without API calls

- All 300 own-K saved score rows were independently rescored with the repository's fixed scorer: 152 correct, 251 answered, 49 abstained, one truncated response counted as incorrect.
- All 300 K=80 saved score rows were rescored: 151 correct, 253 answered, 47 abstained, one truncated response counted as incorrect.
- All 300 confirmatory saved answer rows were rescored: 152 correct, 242 answered, 58 abstained, three truncated responses counted as incorrect.
- Both 300-question confirmatory hybrid outputs were rescored: 193 correct with Direct fallback and 192 with CoT fallback. Selection was checked to use the router answer when present and fallback on abstention.
- Final own-K and K=80 question-ID sets match; the confirmatory IDs do not overlap the final sample.
- Saved final/confirmatory intent selections yield 240/300 and 247/300 correct routes.
- The saved adaptive FactConsolidation 262k result reports 81/100.
- The released dataset passes its existing 13,425-row structural validator; the legacy saved 60-row run matches the first 60 released questions.
- The request/control files needed for saved-score inspection are present locally, including all three native-router call ledgers and the original feedback inputs. The current manuscript's six numerical tables were used to update the README.
- Current pipeline, fixed scorer, and candidate prompt hashes match the frozen experiment manifests. The K=80 package's saved artifact hashes match their recorded values.
- Twenty-four cases check the actual pipeline's paper-described operator behavior and own-K retrieval boundary without importing a model or calling an API: freshness, historical/default/ordinal/fallback selection, normalized deduplication, Boolean latest/ever/negation, empty outputs, and query clean-up.

Run the read-only checks with `python scripts/validation/check_final_paper_results.py` and `python scripts/validation/check_paper_operators.py`.

Historical router/control costs were checked against all five 300-call ledgers and independently recalculated from saved token counts. Latency values match the saved aggregate log and benchmark script. All included confirmatory manifest input hashes, cached MiniLM snapshot file hashes, and the pinned Arrow dataset hash were checked locally. See [LOCAL_EVIDENCE_AUDIT.md](LOCAL_EVIDENCE_AUDIT.md).

These checks inspect/recompute saved outcomes; they do not constitute a fresh model evaluation or validate every benchmark gold label. No new paid experiments were run.

## Documentation and reproduction limits

The folder was not fully aligned before this audit. The older evaluation report, reproducibility guide and project summary have moved to `archive/legacy_planner_docs/`, with native-router replacements at their original `docs/` paths. The pre-v20 Markdown/LaTeX manuscripts have moved to `archive/manuscript_drafts/`; `paper/aletheia_paper.md` and `.tex` now identify the authoritative supplied Word paper instead of presenting obsolete claims as current. The architecture/result diagrams and stale paper-writing TODO were corrected. Existing result-package documentation remains attached to its original historical experiments.

The core pipeline/operators and fixed scorer were not changed: their frozen hashes and saved results remain intact. Only descriptive text in two legacy runner/validator docstrings was clarified. The code still intentionally contains older planner and other experiment paths; they are identified as separate systems in the current guide rather than removed or presented as the paper's native route.

Use `results/prof_feedback/router-final300-native-20261016/scores_fixed.json`, not that folder's initial `scores.json`: the initial runner counted 16 Boolean abstentions as correct. The confirmatory paper hybrids are in `results/router_runs/pf-20261016-router-confirmatory-01/r3_summary.json`; the own-K package's evidence-gated hybrid is a different diagnostic.

The three newer native-router result packages are local/untracked and are included in the faculty ZIP. A GitHub source download alone does not include them. They are:

- `results/prof_feedback/router-final300-native-20261016/`
- `results/prof_feedback/router-final300-native-k80-20261016/`
- `results/router_runs/pf-20261016-router-confirmatory-01/`

The supplied ZIP is a source, saved-evidence, and manuscript snapshot. It does not include the Conda environment, external Hugging Face dataset/model caches, or API credentials. Some saved runner instructions refer to temporary Python environments that no longer exist. Full model reruns require recreating the documented environment, obtaining the pinned data/model files, and deliberately configuring paid API access. Saved-score review uses the included JSON/JSONL evidence without an API key.

The pre-existing `_lf.py` modification enabling `ALETHEIA_DISABLE_TRACING=1` is retained locally and in the faculty source snapshot. Other existing local modifications are preserved. The documentation/code-verifier GitHub update does not upload those local result packages, the tracing shim change, or the manuscript.

## ZIP contents

The package includes the current README, this audit, LICENSE, dependency files, `.env.example`, the current Word manuscript, and the `archive`, `benchmarks`, `data`, `docs`, `logs`, `paper`, `results`, and `scripts` trees, including local-only results.

It excludes `.env`, Git metadata, editor settings, Python caches, virtual environments, macOS metadata, local personal notes/report extracts, and one-off root-level manuscript-generation helpers. No credential file is needed to review the source and saved results.
