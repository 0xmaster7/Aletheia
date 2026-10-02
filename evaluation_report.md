# Aletheia: Memory Conflict Resolution Evaluation

## 1. Standard "Freshness" Benchmark Results
This test was run using the standard `MemoryAgentBench` dataset (which strictly tests finding the **newest/most recent fact** when there are conflicts).

**Context Lengths Tested:**
*   **32k tokens** (~25,000 words / medium noise)
*   **64k tokens** (~50,000 words / high noise)
*   **262k tokens** (~200,000 words / extreme noise)

### Results: Aletheia (Semantic Router) vs. Baseline (BM25)
| Context Length | Baseline (BM25) Accuracy | Aletheia (Semantic Router) Accuracy |
| :--- | :--- | :--- |
| **32k** | 70.00% | **77.00%** |
| **64k** | 75.00% | **81.00%** |
| **262k** | 56.00% | **81.00%** |

**Analysis:**
Across all tested context lengths (32k - 262k), the Aletheia Semantic Router architecture clearly outperforms standard retrieval. Notably, while the baseline degrades heavily under extreme noise limits (dropping to 56.00% at 262k), Aletheia maintains a highly resilient 81.00% accuracy.

---

## 2. Synthetic Benchmark Comparison (Complex Intents)
Because the original author's repository relied entirely on a hardcoded `max(serial)` Python function, it was only capable of answering the "Freshness" questions above. 

We generated a custom **60-question Synthetic Benchmark** across 20 heavily-conflicting entities to test how Aletheia performs against complex real-world queries (Historical, Aggregation, and Boolean) compared to the original author's architecture.

### Benchmark Results
| Question Intent | Original Author's Repo (Projected) | Aletheia (Semantic Router) | Notes |
| :--- | :--- | :--- | :--- |
| **Historical** (e.g. "What was the initial value?") | **0%** | **15.0%** (3/20) | Original repo fails completely as it blindly returns the newest fact instead of the oldest. |
| **Aggregation** (e.g. "How many unique values?") | **0%** | **10.0%** (2/20) | Original repo fails completely as it only returns one single text string instead of performing a count. |
| **Boolean** (e.g. "Is it true they lived here?") | **0%** | **85.0%** (17/20) | Original repo returns raw fact text rather than evaluating logical True/False. Aletheia's logic gate excels here. |
| **Overall Synthetic Score** | **0%** | **36.6%** (22/60) | |

**Analysis:**
While the overall accuracy on Historical and Aggregation questions in Aletheia is currently low (10-15%) and indicates room for further research in chunking/LLM parsing limits, it still represents a functional leap over the original author's architecture (0%). Aletheia successfully identifies the intent and attempts the correct mathematical operation (`min()`, `count()`), whereas the original repository is incapable of handling anything outside of `max(serial)`. Aletheia's Boolean logic gate is highly successful (85%).
