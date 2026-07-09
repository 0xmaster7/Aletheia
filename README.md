# Research Notes: Memory Conflict Resolution in Agentic Memory Systems

Notes on prior art before starting implementation. Goal: build a **fully local, non-LLM-extraction version of deterministic conflict resolution**, plus a **question-type router** that handles cases the current best paper doesn't.

---

## Key paper: "Don't Ask the LLM to Track Freshness"

**Link:** https://arxiv.org/abs/2606.01435 (also available as HTML: https://arxiv.org/html/2606.01435v1)
**Date:** May 2026

### The problem it addresses
When an agent's memory contains contradictory facts (e.g. "works at Google" written in January, "works at Meta" written in June), the system needs to figure out which is current when answering a question. Every major memory system — Mem0, Zep/Graphiti, MemGPT, Cognee, HippoRAG-v2 — performs badly at this, even when explicitly told which fact is newer.

### The core finding
Letting the LLM judge "which fact is more recent/correct" fails because the LLM's own background knowledge quietly overrides the in-context instruction — it isn't a reading failure, it's a bias-override failure. This gets worse as more facts are added to context.

### The fix
Split the task into two steps instead of one:
1. **LLM step (kept narrow):** filter retrieved facts down to only the ones relevant to the question. No judgment, no ranking — just relevance filtering.
2. **Code step (zero LLM):** apply `max(candidates, key=serial_number_or_timestamp)` to pick the newest fact. Plain Python, deterministic, no model call.

For multi-hop questions (where the answer requires chasing a chain of facts, e.g. "where does the author of this book currently live"), they apply the same two-step logic at each link in the chain.

### Results
- Single-hop conflict resolution: **78–94.8%** accuracy depending on backbone model, vs. the previous best published system's **54%** (HippoRAG-v2).
- Mem0 and Zep/Graphiti score **18%** and **7%** respectively on the same task, despite Zep being purpose-built for temporal/time-aware memory.
- The deterministic approach stays stable as more facts are added to context; LLM-judgment approaches degrade.
- Multi-hop is still much harder for everyone — best results are far below single-hop.

### Stated limitations (this is the actual opportunity)
The paper explicitly does **not** solve:
- **Yes/No questions** ("Do I still work at Google?") — needs a match/compare step, not a lookup.
- **"What was it before?" questions** ("Where did I work before Meta?") — `max()` throws away exactly the fact these questions need; needs full history + second-highest (or targeted) retrieval instead.
- **Aggregation/counting questions** ("How many jobs have I had?") — needs the full list of facts, not the single current winner.
- All experiments use cloud LLMs only (GPT-4o-mini, GPT-4o, o4-mini) for the extraction step — no local/open-weight model is tested.

---

## Benchmarks in this space

| Benchmark | What it tests | Link |
|---|---|---|
| **MemoryAgentBench** (ICLR 2026) | 4 competencies: Accurate Retrieval, Test-Time Learning, Long-Range Understanding, Conflict Resolution (includes the FactConsolidation task used above) | https://github.com/HUST-AI-HYZ/MemoryAgentBench |
| **STALE** | 400 expert-validated conflict scenarios (1,200 queries) testing whether agents notice memories are outdated and act on it | https://arxiv.org/abs/2605.06527 |
| **LongMemEval** | Long-context (up to ~1.5M tokens) memory QA, temporal-reasoning heavy | (standard long-mem benchmark, referenced across the above papers) |
| **LoCoMo** | Long multi-session conversational memory QA — general retrieval accuracy, not conflict-specific | (standard benchmark, used by Mem0/Zep/Hindsight) |
| **Supersede** | Diagnoses the same memory-update gap; notes no existing benchmark is *trainable* (reward = supersession-correctness) — purely diagnostic so far | https://arxiv.org/abs/2606.27472 |

---

## What this project is trying to add

1. **Local extraction instead of cloud LLM.** Replace the GPT-4o-mini filtering step with a local pipeline (spaCy NER/dependency parsing, or a small local model) — test whether the deterministic `max()` win survives with zero API calls and zero cost per query.
2. **A question-type router.** Classify incoming questions into: current-state (→ `max()`), historical (→ full-history + targeted retrieval), yes/no (→ compare-against-max), aggregation (→ full scan). This is the piece the source paper explicitly left unbuilt.
3. **Latency/cost instrumentation.** Report tokens-per-query and p50/p95 latency alongside accuracy — the source paper doesn't report cost/latency comparisons in these terms.

## Known shortcomings of our approach (to track honestly as we build)

- **Question-type classification is itself an open problem** — if it's wrong, the correct operator gets applied to the wrong question. This is the actual hard part of the project, not the operators themselves.
- **Local extraction will likely be less accurate than GPT-4o-class extraction** at pulling clean entities/relations out of messy natural language — the open question is *how much* accuracy is lost, not whether some is lost.
- **"Before" questions require disambiguating which fact "before" refers to** when an entity has multiple attributes changing at different times — not solved by second-max alone in all cases.
- **No trainable benchmark exists yet** for this task (per Supersede) — we're limited to evaluating frozen pipelines, not fine-tuning against a reward signal, at least initially.
- **Multi-hop conflict resolution is still poorly solved industry-wide** (best published accuracy ≤6% in some settings) — likely out of scope to fully solve in this project; single-hop + the four question types above is the realistic target.