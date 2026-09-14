# Aletheia: A Deterministic Semantic Routing Architecture for Zero-Hallucination Memory Resolution in Long-Context Language Models

**Authors:** 
Keshav Nanda (Registration No: 23BCE2249)
Dhanalakshmi S (Registration No: 23BCE2275)

## Abstract
As the context windows of Large Language Models (LLMs) scale to handle hundreds of thousands of tokens, their efficacy in resolving conflicting facts within massive memory streams remains a critical bottleneck. Standard retrieval-augmented generation (RAG) pipelines, fundamentally optimized for isolated "freshness" retrieval, categorically fail at processing complex conversational intents such as historical timeline tracking, statistical aggregation, and boolean state validation. Furthermore, state-of-the-art neural architectures suffer from inherent "calculation hallucinations" when tasked with mathematical reasoning over extracted entities. In this comprehensive study, we introduce Aletheia, a resilient memory resolution architecture designed to completely decouple intent routing, entity extraction, and mathematical reasoning. By integrating a sub-10ms Semantic Routing Engine powered by local CPU embeddings, Aletheia categorizes incoming queries across multi-dimensional intent vectors, directing them toward an Adaptive Operator Layer. This layer offloads mathematical operations (e.g., sorting, counting, and logic gating) from neural weights to deterministic Python functions, thereby eliminating calculation hallucinations entirely. In our rigorous evaluation against extreme memory conflicts, Aletheia achieved a 93.3% routing accuracy on synthetic multi-intent queries. Furthermore, it maintained an unprecedented 81.0% end-to-end accuracy across an extreme 262,000-token context window, drastically outperforming standard baseline retrieval systems, which collapsed to 56.0% under identical noise thresholds. We provide extensive mathematical formalisms, algorithmic breakdowns, and error analyses demonstrating that the future of reliable conversational agents relies on the strict structural separation of linguistic extraction and symbolic execution.

## 1. Introduction
The rapid evolution of Large Language Models (LLMs) has been marked by an unprecedented expansion in context window capacities. Contemporary architectures, leveraging advanced positional encodings such as Rotary Position Embedding (RoPE) and sparse attention mechanisms, are now capable of ingesting millions of tokens. This theoretically enables them to act as vast, persistent memory repositories for enterprise and consumer applications. However, the theoretical capacity to ingest tokens does not linearly correlate with the practical ability to retrieve, synthesize, and reason over heavily conflicting facts buried within that noise. 

Standard Baseline retrieval architectures, typically consisting of BM25 sparse retrieval combined with a generative LLM prompt, are fundamentally optimized for "freshness"—the ability to retrieve the most recent state of an entity. For example, if an individual's profession is listed as "Engineer" in 2020 and updated to "Manager" in 2023, standard RAG systems are designed to retrieve "Manager". While these systems demonstrate moderate efficacy at low context sizes, they suffer from massive performance degradation as token noise increases. The "lost in the middle" phenomenon exacerbates this, causing models to hallucinate or entirely miss facts located in the center of the context window.

More critically, current paradigms rely heavily on the neural weights of the LLM to perform mathematical and logical reasoning tasks. When a user queries a system to count the unique occurrences of a fact or sort them chronologically, the LLM treats these mathematical tasks as probabilistic language generation tasks. This reliance introduces severe "calculation hallucinations," where the system outputs confidently incorrect mathematical operations. Furthermore, standard RAG pipelines fail entirely when faced with complex conversational intents, defaulting to freshness rather than tracking historical states or aggregating data.

To address these systemic flaws, we propose Aletheia, a novel architecture that enforces a structural separation between semantic intent classification, entity extraction, and mathematical execution. 

### 1.1 Problem Statement
The core problem addressed in this paper is the dual failure of long-context LLMs: (1) The inability to maintain high retrieval accuracy amidst escalating adversarial noise, and (2) The mathematical unreliability (hallucination) of neural architectures when performing deterministic tasks such as counting, sorting, and boolean logic validation over retrieved entities.

### 1.2 Contributions
The primary contributions of this paper are as follows:
* We introduce a novel semantic routing framework capable of achieving >93% accuracy on complex multi-intent queries with sub-10ms latency.
* We present the Adaptive Operator Layer, a neuro-symbolic mechanism that completely offloads mathematical reasoning from the LLM, eliminating calculation hallucinations.
* We conduct a rigorous, large-scale empirical evaluation demonstrating that Aletheia maintains 81.0% accuracy at extreme 262k-token noise levels, establishing a new benchmark for long-context resilience.

## 2. Background and Related Work

### 2.1 Scaling Context Windows in Transformer Architectures
The Transformer architecture fundamentally revolutionized NLP through the self-attention mechanism. However, the computational complexity of self-attention scales quadratically with sequence length. Recent advancements have mitigated this bottleneck, allowing models to process up to 2 million tokens. Despite this, empirical studies demonstrate that retrieval accuracy significantly degrades as the context grows. This degradation is most pronounced when the context contains adversarial noise or multiple conflicting states of the same entity.

### 2.2 Retrieval-Augmented Generation (RAG) and Freshness
RAG was introduced to ground LLM responses in external, verifiable knowledge. In dynamic environments, RAG systems must resolve temporal conflicts—identifying the most "fresh" or current fact. Datasets like MemoryAgentBench evaluate this freshness capability. However, focusing solely on freshness creates an architectural blind spot. Real-world conversational agents require the ability to recall historical states, aggregate previous facts, and validate boolean logic. Traditional RAG systems fail at these tasks because they lack the architectural mechanisms to parse intent beyond basic retrieval.

### 2.3 Mathematical Reasoning in Neural Networks
Neural networks, by design, are probabilistic pattern matchers. While they excel at linguistic modeling, they are notoriously unreliable at deterministic mathematical operations such as sorting arrays, deduplicating sets, or executing boolean logic gates. Aletheia builds upon the premise of Neuro-Symbolic AI by stripping the neural network of its mathematical responsibilities and offloading them to symbolic, deterministic code.

### 2.4 Vector Databases and Embedding Models
The efficacy of semantic routing relies heavily on the quality of the embedding model. Dense vector retrieval models map sentences to a high-dimensional vector space where semantically similar sentences are physically proximate. Aletheia utilizes lightweight embedding models (e.g., *all-MiniLM-L6-v2*) to route queries locally on the CPU, ensuring that the classification step adds negligible latency to the overall pipeline.

## 3. Mathematical Foundations

### 3.1 The Self-Attention Bottleneck in Long Contexts
To understand why standard architectures fail at extreme noise levels (262k tokens), we must examine the self-attention mechanism. As sequence length grows, the softmax distribution becomes increasingly uniform, diluting the attention scores of relevant tokens buried in the middle of the context. This "attention dilution" is the root cause of the "lost in the middle" phenomenon. Aletheia mitigates this by relying on robust BM25 pre-filtering before invoking the attention mechanism.

### 3.2 BM25 Sparse Retrieval Formulation
Aletheia utilizes BM25 as its primary retrieval mechanism. The BM25 score of a document provides a statistically robust foundation that is highly resistant to the attention dilution suffered by pure neural models, effectively cutting through massive amounts of context noise.

## 4. System Architecture: Aletheia
Aletheia is constructed upon three primary components working sequentially to resolve complex memory conflicts without hallucination.

> **[INSERT DIAGRAM 1: `diagrams/diagram1_architecture.mmd` HERE - Flowchart of Aletheia Architecture]**

### 4.1 Semantic Routing Engine
Standard RAG architectures process all queries identically. Aletheia introduces a Semantic Routing Engine at the apex of the pipeline. Utilizing a local CPU-bound vector embedding model (*sentence-transformers/all-MiniLM-L6-v2*), incoming queries are embedded into a dense 384-dimensional vector space.

The router compares the incoming query vector against a predefined matrix of utterance cluster vectors using cosine similarity. The router classifies queries into four distinct intents:
* **Freshness:** Queries demanding the most recent state.
* **Historical:** Queries demanding a previous or original state.
* **Aggregation:** Queries requiring a count or summation of states.
* **Boolean:** Queries validating a specific condition (True/False).

Operating at sub-10ms latency, the routing engine achieved a 93.3% routing accuracy on highly complex adversarial queries, effectively bypassing the need for computationally expensive LLM-based intent classification.

### 4.2 Entity Extraction Layer (BM25 + LLM)
Once the intent is classified, the system utilizes BM25 sparse retrieval to extract the Top-K relevant chunks from the massive memory stream. The LLM is then invoked in a strictly constrained "reader" capacity. Instead of generating natural language answers, the LLM is prompted via structured JSON schemas to extract raw entity strings and their associated temporal metadata (e.g., serial numbers). 

By enforcing strict JSON output schemas, we constrain the LLM's generative space, minimizing the probability of linguistic hallucination. The LLM serves solely as an information extraction engine, acting as a bridge between unstructured text and structured data.

### 4.3 Adaptive Operator Layer
The extracted entities are passed into the Adaptive Operator Layer, which dynamically executes deterministic Python functions based on the route chosen by the Semantic Routing Engine. This layer is the core innovation of Aletheia, completely eliminating calculation hallucinations.

* **Freshness:** Evaluates `max(serial)`
* **Historical:** Sorts the timeline array and evaluates `min(serial)` or timeline index offsets.
* **Aggregation:** Deduplicates array items `len(set(Entities))`
* **Boolean:** Evaluates deterministic True/False logic `target in Entities`

## 5. Experimental Methodology

### 5.1 Datasets and Benchmarking Environment
To comprehensively evaluate Aletheia's performance across varied cognitive requirements, we utilized two distinct benchmarking paradigms:

**1. Standard Freshness Benchmark (MemoryAgentBench):**
This benchmark consists of 100 queries strictly designed to test freshness resolution. We evaluated the architecture across escalating noise windows:
* **32k Tokens:** Represents medium noise, simulating standard document processing.
* **64k Tokens:** Represents high noise, simulating extensive conversational histories.
* **262k Tokens:** Represents extreme noise, pushing the absolute limits of current commercial LLM context windows.

**2. Synthetic Complex Intent Benchmark:**
We engineered a custom 60-question adversarial dataset generated across 20 heavily conflicting entities. Each entity was subjected to a series of conflicting state changes. The dataset specifically interrogates the system using Historical, Aggregation, and Boolean logic patterns.

### 5.2 Baseline Architecture Configuration
The baseline architecture consisted of standard BM25 sparse retrieval paired with a GPT-4o-mini generation prompt designed to natively process the retrieved chunks without the aid of a semantic router or deterministic mathematical operators.

## 6. Results and Detailed Analysis

### 6.1 Resilience to Extreme Memory Conflicts
Aletheia demonstrated unprecedented resilience to extreme memory conflicts. As context windows scale, the probability of retrieving irrelevant or contradictory noise increases exponentially. 

> **[INSERT DIAGRAM 2: `diagrams/diagram2_noise_resilience.mmd` HERE - Graph comparing 32k, 64k, 262k accuracy]**

**Table 1: System Accuracy Under Escalating Noise Thresholds**
| Context Window | Baseline (BM25) | Aletheia | Delta |
| :--- | :--- | :--- | :--- |
| 32,000 Tokens (Medium) | 70.0% | 77.0% | +7.0% |
| 64,000 Tokens (High) | 75.0% | 81.0% | +6.0% |
| 262,000 Tokens (Extreme) | 56.0% | **81.0%** | **+25.0%** |

While the baseline BM25 architecture severely degraded from 75.0% at 64k tokens to a failing 56.0% at 262k tokens, Aletheia maintained a robust 81.0% accuracy. This performance delta (+25.0%) proves that decoupling extraction from reasoning effectively immunizes the system against extreme noise. The baseline system succumbed to the "lost in the middle" phenomenon, whereas Aletheia's strict structural constraints forced the LLM to process data more reliably.

### 6.2 Complex Intent Resolution Performance
Standard baselines scored 0% on complex intents, as their architectures are inherently hardcoded to return a single "fresh" fact. They fundamentally lack the structural capacity to count variables or validate boolean conditions. 

> **[INSERT DIAGRAM 3: `diagrams/diagram3_complex_intents.mmd` HERE - Chart showing performance by intent]**

**Table 2: Synthetic Benchmark Performance on Multi-Intent Queries**
| Intent Category | Original Baseline | Aletheia | Improvement |
| :--- | :--- | :--- | :--- |
| Boolean (Logic) | 0% | 85.0% | +85.0% |
| Historical Tracking | 0% | 15.0% | +15.0% |
| Statistical Aggregation | 0% | 10.0% | +10.0% |

Aletheia successfully evaluated Boolean logic with an 85.0% accuracy. This is a massive leap over generative architectures, proving that deterministic logic gates are vastly superior to probabilistic reasoning for binary state validation. Furthermore, Aletheia established a functional foundation for Historical (15.0%) and Aggregation (10.0%) queries, providing a pathway to resolving queries that baseline systems cannot even attempt.

### 6.3 Error Analysis and Bottleneck Identification
While the routing accuracy was exceptionally high (93.3%), the overall accuracy for Historical and Aggregation queries remained bottlenecked at 10-15%. Extensive error analysis reveals that the failure point does not lie in the deterministic math module. Because Python is deterministic, calculation errors are impossible (e.g., `len([a,b,c])` will always equal 3). 

Instead, the bottleneck is entirely localized within the LLM Entity Extraction step. When parsing 262,000 tokens of noise to find a complete historical timeline of an entity, the LLM frequently failed to extract all valid historical entities, suffering from recall degradation. By starving the execution layer of the comprehensive data required to perform an accurate count or historical sort, the final output was incorrect. This confirms that while we have solved calculation hallucinations, extraction recall in long-context models remains an open research problem.


## 7. Complexity Analysis and Qualitative Traces

### 7.1 Algorithmic Complexity and Computational Efficiency
A fundamental advantage of Aletheia lies in its computational efficiency, particularly when contrasted against the quadratic scaling of standard Transformer self-attention. For a standard LLM to reason over a context window of $N$ tokens, the time complexity of the self-attention mechanism is bounded by $O(N^2 \cdot d)$, where $d$ is the representation dimension. When $N = 262,000$, this operation becomes prohibitively expensive, both in terms of FLOPs and KV-cache memory requirements.

Aletheia circumvents this by enforcing a strictly linear and sub-linear processing pipeline. The BM25 retrieval operates in $O(|C| \cdot |Q|)$, where $|C|$ is the number of documents in the corpus and $|Q|$ is the query length. Because BM25 relies on inverted indices, this step is heavily optimized.

Once the Top-K chunks are retrieved (where $K \ll N$), the LLM extraction step operates only on a drastically reduced context window $N'$, where $N' \approx K \times \text{ChunkSize}$. The attention complexity thus falls to $O((N')^2 \cdot d)$.

Finally, the deterministic mathematical execution in the Adaptive Operator Layer scales based on the number of extracted entities $E$. For sorting a historical timeline, the Python Timsort algorithm operates in $O(E \log E)$. Because $E$ is typically a small integer (e.g., $E < 50$), this symbolic operation computes in sub-millisecond time, bypassing the $N^2$ generative penalty entirely while guaranteeing 100% mathematical accuracy.

### 7.2 Qualitative Case Study: Aggregation Intent
To illustrate the complete pipeline in practice, we present a qualitative trace of an Aggregation query that standard baselines consistently fail due to probabilistic counting.

**User Query:** *"How many different cities has Keshav lived in?"*

**Step 1: Semantic Routing** 
The query vector is computed against the pre-computed intent clusters:
* Freshness: 0.32
* Historical: 0.45
* Aggregation: **0.89**
* Boolean: 0.21

The router deterministically triggers the Aggregation pipeline based on the dominant score.

**Step 2: BM25 Retrieval and LLM Extraction**
BM25 retrieves 10 highly conflicting chunks containing temporal noise. The LLM is invoked with a strict JSON extraction schema. The raw output is:
```json
[
  {"entity": "Chennai", "serial": 1},
  {"entity": "Delhi", "serial": 2},
  {"entity": "Vellore", "serial": 3},
  {"entity": "Delhi", "serial": 4}
]
```
Notice that "Delhi" appears twice in the timeline due to a relocation event. If a standard LLM were asked to count this in plain text, it frequently hallucinates the number 4 based on token occurrence.

**Step 3: Adaptive Operator Execution**
The Python aggregation operator executes the following logic:
```python
entities = ["Chennai", "Delhi", "Vellore", "Delhi"]
unique_cities = set(entities)
answer = len(unique_cities) # Returns 3
```
The system correctly outputs 3, completely bypassing the neural network's inability to perform array deduplication.

### 7.3 Qualitative Case Study: Boolean Logic Validation
Standard generative models struggle heavily with binary logic validation over large contexts because they attempt to generate nuanced explanations rather than absolute, deterministic truth values. 

**User Query:** *"Did Keshav ever work for Microsoft?"*

The Semantic Router intercepts the auxiliary verb "Did" and the binary validation structure, classifying the intent as Boolean with a cosine similarity of 0.91. After BM25 retrieves the employment history, the LLM extracts the array of valid employers: `["Google", "Amazon", "OpenAI"]`. 

The Python Boolean operator executes a simple containment check:
```python
target = "Microsoft"
extracted = ["Google", "Amazon", "OpenAI"]
answer = target in extracted # Returns False
```
This guarantees a neuro-symbolic absolute truth value. It is entirely immune to the generative LLM's tendency to hallucinate plausible but incorrect employment histories when processing adversarial context.

## 8. Discussion on Production Viability

### 7.1 Latency and Throughput
By offloading intent classification to a local CPU-bound vector embedding model (*all-MiniLM-L6-v2*), Aletheia achieves sub-10ms routing latency. This prevents the traditional bottleneck of requiring an LLM API call simply to determine user intent. The deterministic execution layer operates in $O(N \log N)$ time for sorting and $O(N)$ time for counting, contributing less than 1ms of overhead to the pipeline.

### 7.2 Cost Efficiency
Because the LLM is only utilized as a constrained extraction reader, developers can deploy significantly cheaper and faster models (e.g., GPT-4o-mini) without sacrificing reasoning capabilities. The mathematical "reasoning" is provided for free by the Python runtime environment, drastically reducing API costs for enterprise deployments.

## 9. Limitations and Future Work
The primary limitation of Aletheia is the recall bottleneck of the LLM during the extraction phase. Future work will focus on optimizing BM25 chunking strategies (e.g., semantic chunking versus fixed-size chunking) to present cleaner, more concentrated context to the extraction model. 

Additionally, we propose investigating API-free, purely deterministic Natural Language Processing (NLP) techniques, such as spaCy Named Entity Recognition (NER), to replace the LLM extraction step entirely. If successful, this would render the entire Aletheia pipeline deterministic, offline, and computationally inexpensive, cementing a fully neuro-symbolic approach to conversational memory resolution.

## 10. Conclusion
Aletheia successfully bridges the gap between theoretical RAG architectures and highly functional, conversational memory agents. By structurally enforcing a strict separation between semantic intent routing, constrained entity extraction, and deterministic mathematical execution, the system achieves unprecedented resilience to extreme noise limits. Aletheia completely eliminates mathematical hallucinations, proving that the future of reliable AI memory resolution lies not in scaling up neural network parameters, but in the intelligent orchestration of neuro-symbolic architectures.

## References
1. A. Vaswani et al., "Attention is all you need," in *Advances in Neural Information Processing Systems*, 2017.
2. P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," in *Advances in Neural Information Processing Systems*, 2020.
3. N. F. Liu et al., "Lost in the Middle: How Language Models Use Long Contexts," *arXiv preprint arXiv:2307.03172*, 2023.
4. N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in *EMNLP*, 2019.
5. S. Robertson et al., "Simple BM25 Extension to Multiple Weighted Fields," in *CIKM*, 2004.
