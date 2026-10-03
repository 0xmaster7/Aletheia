# Aletheia: Project To-Do List

## Code & Architecture Tasks
- [ ] **Implement OOD (Out-of-Distribution) Rejection:** Add a minimum cosine similarity threshold to the Semantic Router. If a query (e.g., "How is the weather in Delhi?") scores below the threshold across all intents, the system should gracefully reject it rather than forcing it down an irrelevant pathway.
- [ ] **Investigate API-Free Extraction:** Research whether we can completely remove the OpenAI API dependency. Explore using purely deterministic NLP methods (e.g., spaCy Named Entity Recognition, Regex, or local lightweight NER models) to extract the structured entities from the unstructured BM25 chunks.

## Documentation
- [ ] **Write Research Paper:** Draft the final paper outlining the baseline flaws, the Semantic Router architecture, the deterministic execution layer, and the 262k extreme-noise ablation results.
