# Aletheia follow-up work

## Completed for the supplied final paper

- [x] Include the final Word manuscript in the local faculty package.
- [x] Align the README, evaluation report, reproducibility guide and project summary with native-router results.
- [x] Update the architecture and result diagrams to the paper's figures.
- [x] Archive superseded planner documentation and manuscript drafts.
- [x] Check frozen source hashes, saved-score totals, and paper-described operator behavior offline.

## Future research, not implemented or claimed by the paper

- [ ] Add a calibrated OOD rejection threshold to the embedding router.
- [ ] Evaluate deterministic extraction such as spaCy/regex against the LLM extractor.
- [ ] Compare chunking alternatives and stronger extractors on the synthetic benchmark.
- [ ] Build independent out-of-distribution questions and broaden evaluation beyond the sampled templated sets.
- [ ] Measure end-to-end latency and sequential fallback deployment cost.
- [ ] Investigate routing/retrieval/extraction errors with component-level analysis.

Future changes should be evaluated as new experiments rather than silently changing the frozen implementation or saved outputs.
