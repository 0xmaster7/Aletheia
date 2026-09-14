# Aletheia: A Deterministic Semantic Routing Architecture for Zero-Hallucination Memory Resolution in Long-Context Language Models

**Authors:** Keshav Nanda (23BCE2249), Dhanalakshmi S (23BCE2275)

## Abstract
As the context windows of Large Language Models (LLMs) scale to handle hundreds of thousands of tokens, their efficacy in resolving conflicting facts within massive memory streams remains a critical bottleneck. Standard retrieval-augmented generation (RAG) pipelines, fundamentally optimized for isolated "freshness" retrieval, categorically fail at processing complex conversational intents such as historical timeline tracking, statistical aggregation, and boolean state validation. Furthermore, state-of-the-art neural architectures suffer from inherent "calculation hallucinations" when tasked with mathematical reasoning over extracted entities. In this paper, we introduce Aletheia, a resilient memory resolution architecture designed to decouple intent routing, entity extraction, and mathematical reasoning. By integrating a sub-10ms Semantic Routing Engine, Aletheia categorizes incoming queries across multi-dimensional intent vectors, directing them toward an Adaptive Operator Layer. This layer completely offloads mathematical operations (e.g., sorting, counting, and logic gating) from neural weights to deterministic Python functions, thereby eliminating calculation hallucinations entirely. In our rigorous evaluation against extreme memory conflicts, Aletheia achieved a 93.3% routing accuracy on synthetic multi-intent queries. Furthermore, it maintained an unprecedented 81.0% end-to-end accuracy across an extreme 262,000-token context window, drastically outperforming standard baseline retrieval systems, which collapsed to 56.0% under identical noise thresholds. Our findings present a functional leap toward deterministic reliability in long-context conversational agents.

## 1. Introduction
The rapid evolution of Large Language Models (LLMs) has been marked by an unprecedented expansion in context window capacities. Contemporary architectures are now capable of ingesting millions of tokens, theoretically enabling them to act as vast, persistent memory repositories. However, the theoretical capacity to ingest tokens does not linearly correlate with the practical ability to retrieve, synthesize, and reason over heavily conflicting facts buried within that noise. 

Standard Baseline retrieval architectures, typically consisting of BM25 sparse retrieval combined with a generative LLM prompt, are fundamentally optimized for "freshness"—the ability to retrieve the most recent state of an entity. While these systems demonstrate moderate efficacy at low context sizes, they suffer from massive performance degradation as token noise increases. The "lost in the middle" phenomenon exacerbates this, causing models to hallucinate or entirely miss facts located in the center of the context window.

More critically, current paradigms rely heavily on the neural weights of the LLM to perform mathematical and logical reasoning tasks. When a user queries a system to count the unique occurrences of a fact or sort them chronologically, the LLM treats these mathematical tasks as probabilistic language generation tasks. This reliance introduces severe "calculation hallucinations," where the system outputs confidently incorrect mathematical operations. Furthermore, standard RAG pipelines fail entirely when faced with complex conversational intents, defaulting to freshness rather than tracking historical states or aggregating data.

To address these systemic flaws, we propose Aletheia, a novel architecture that enforces a structural separation between semantic intent classification, entity extraction, and mathematical execution. 

## 2. Background and Related Work

### 2.1 Limitations of Long-Context LLMs
Recent advancements in Transformer architectures have pushed context limits from 4,000 tokens to over 2 million. Despite this, empirical studies demonstrate that retrieval accuracy significantly degrades as the context grows. This degradation is most pronounced when the context contains adversarial noise or multiple conflicting states of the same entity (e.g., a person changing professions over time).

### 2.2 Retrieval-Augmented Generation (RAG) and Freshness
RAG was introduced to ground LLM responses in external, verifiable knowledge. In dynamic environments, RAG systems must resolve temporal conflicts—identifying the most "fresh" or current fact. Datasets like MemoryAgentBench evaluate this freshness capability. However, focusing solely on freshness creates an architectural blind spot: the inability to recall historical states or aggregate previous facts.

### 2.3 Mathematical Reasoning in Neural Networks
Neural networks, by design, are probabilistic pattern matchers. While they excel at linguistic modeling, they are notoriously unreliable at deterministic mathematical operations such as sorting arrays, deduplicating sets, or executing boolean logic gates. Aletheia builds upon the premise of neuro-symbolic AI by stripping the neural network of its mathematical responsibilities and offloading them to symbolic, deterministic code.

## 3. Proposed Architecture: Aletheia
Aletheia is constructed upon three primary components working sequentially to resolve complex memory conflicts without hallucination.

> **[INSERT DIAGRAM 1: `diagrams/diagram1_architecture.mmd` HERE - Flowchart of Aletheia Architecture]**

### 3.1 Semantic Routing Engine
Standard RAG architectures process all queries identically. Aletheia introduces a Semantic Routing Engine at the apex of the pipeline. Utilizing a local CPU-bound vector embedding model (*sentence-transformers/all-MiniLM-L6-v2*), incoming queries are embedded into a dense vector space and compared against predefined semantic clusters using cosine similarity. The router classifies queries into four distinct intents:
* **Freshness:** Queries demanding the most recent state.
* **Historical:** Queries demanding a previous or original state.
* **Aggregation:** Queries requiring a count or summation of states.
* **Boolean:** Queries validating a specific condition (True/False).

Operating at sub-10ms latency, the routing engine achieved a 93.3% routing accuracy on highly complex adversarial queries.

### 3.2 Entity Extraction (BM25 + LLM)
Once the intent is classified, the system utilizes BM25 sparse retrieval to extract the Top-K relevant chunks from the massive memory stream. The LLM is then invoked in a strictly constrained "reader" capacity. Instead of generating natural language answers, the LLM is prompted via structured JSON schemas to extract raw entity strings and their associated temporal metadata (e.g., serial numbers). This completely strips the LLM of calculation duties.

### 3.3 Adaptive Operator Layer
The extracted entities are passed into the Adaptive Operator Layer, which dynamically executes deterministic Python functions based on the chosen route.

**Freshness and Historical Operations:**
For freshness queries, the operator calculates the maximum serial number. For historical queries, it evaluates timeline offset logic (e.g., "previous") and calculates the minimum serial number. This prevents LLM hallucination of temporal order.

**Aggregation Operations:**
For queries asking "how many," the operator deduplicates the array of extracted entities and returns the integer length, avoiding probabilistic counting failures.

**Boolean Logic Gates:**
For validation queries, the operator checks for the existence of the target entity across the extracted timeline, returning a deterministic True or False.

## 4. Experimental Setup

### 4.1 Datasets
We evaluated Aletheia against two distinct benchmarks to measure both noise resilience and intent versatility:
1. **Standard Freshness Benchmark (MemoryAgentBench):** 100 queries strictly testing freshness resolution across escalating noise windows (32k, 64k, and 262k tokens).
2. **Synthetic Complex Intent Benchmark:** A custom 60-question adversarial dataset generated across 20 heavily conflicting entities, specifically designed to test Historical, Aggregation, and Boolean logic capabilities.

### 4.2 Baseline Architecture
The baseline architecture consisted of standard BM25 sparse retrieval paired with a GPT-4o-mini generation prompt designed to natively process the retrieved chunks without the aid of a semantic router or deterministic mathematical operators.

## 5. Results and Discussion

### 5.1 Resilience to Extreme Memory Conflicts
Aletheia demonstrated unprecedented resilience to extreme memory conflicts. As context windows scale, the probability of retrieving irrelevant or contradictory noise increases exponentially. 

> **[INSERT DIAGRAM 2: `diagrams/diagram2_noise_resilience.mmd` HERE - Graph comparing 32k, 64k, 262k accuracy]**

**Table 1: System Accuracy Under Escalating Noise Thresholds**
| Context Window | Baseline (BM25) | Aletheia | Delta |
| :--- | :--- | :--- | :--- |
| 32,000 Tokens (Medium) | 70.0% | 77.0% | +7.0% |
| 64,000 Tokens (High) | 75.0% | 81.0% | +6.0% |
| 262,000 Tokens (Extreme) | 56.0% | **81.0%** | **+25.0%** |

While the baseline BM25 architecture severely degraded from 75.0% at 64k tokens to a failing 56.0% at 262k tokens, Aletheia maintained a robust 81.0% accuracy. This performance delta (+25.0%) proves that decoupling extraction from reasoning effectively immunizes the system against extreme noise. 

### 5.2 Complex Intent Resolution
Standard baselines scored 0% on complex intents, as their architectures are inherently hardcoded to return a single "fresh" fact. They fundamentally lack the structural capacity to count variables or validate boolean conditions. 

> **[INSERT DIAGRAM 3: `diagrams/diagram3_complex_intents.mmd` HERE - Chart showing performance by intent]**

**Table 2: Synthetic Benchmark Performance on Multi-Intent Queries**
| Intent Category | Original Baseline | Aletheia |
| :--- | :--- | :--- |
| Boolean (Logic) | 0% | 85.0% |
| Historical Tracking | 0% | 15.0% |
| Statistical Aggregation | 0% | 10.0% |

Aletheia successfully evaluated Boolean logic with an 85.0% accuracy. Furthermore, it established a functional foundation for Historical (15.0%) and Aggregation (10.0%) queries, providing a pathway to resolving queries that baseline systems cannot even attempt.

### 5.3 Error Analysis of Complex Intents
While the routing accuracy was exceptionally high (93.3%), the overall accuracy for Historical and Aggregation queries remained bottlenecked at 10-15%. Error analysis reveals that the failure point does not lie in the deterministic math, but rather in the LLM Entity Extraction step. When parsing 262,000 tokens of noise, the LLM frequently failed to extract all valid historical entities, starving the execution layer of the comprehensive data required to perform an accurate count or historical sort. 

## 6. Limitations and Future Work
The primary limitation of Aletheia is the recall bottleneck of the LLM during the extraction phase. Future work will focus on optimizing BM25 chunking strategies (e.g., semantic chunking versus fixed-size chunking) to present cleaner context to the LLM. Additionally, we propose investigating API-free, purely deterministic Natural Language Processing (NLP) techniques, such as spaCy Named Entity Recognition (NER), to replace the LLM extraction step entirely, further cementing the deterministic nature of the pipeline.

## 7. Conclusion
Aletheia successfully bridges the gap between theoretical RAG architectures and highly functional, conversational memory agents. By structurally enforcing a strict separation between semantic intent routing, constrained entity extraction, and deterministic mathematical execution, the system achieves unprecedented resilience to extreme noise limits. Aletheia completely eliminates mathematical hallucinations, proving that the future of reliable AI memory resolution lies not in larger neural networks, but in the intelligent orchestration of neuro-symbolic architectures.
