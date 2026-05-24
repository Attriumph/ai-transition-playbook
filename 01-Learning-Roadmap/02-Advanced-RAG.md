# 🔍 Advanced Retrieval-Augmented Generation (RAG)

## 🏗️ The Production-Grade RAG Architecture
Naive vector search (converting text to embeddings and performing cosine similarity) suffers from semantic mismatch, vocabulary mismatch, and lacks structural reasoning. Production-grade RAG systems require a hybrid approach combining **exact match** (sparse keywords) with **concept matching** (dense vectors), optimized via reranking layers and semantic compressors.

---

## 🗺️ Core Curriculum & Architectural Deep Dives

### 1. Hybrid Search & Reciprocal Rank Fusion (RRF)
To combine results from sparse retrievers (BM25/FTS5) and dense retrievers (Vector embeddings) without having to normalize their raw, disparate scores, we use **Reciprocal Rank Fusion (RRF)**.

#### 🧮 Mathematical Formulation
The RRF score for a document $d$ across a set of retrieval channels $R$ is formulated as:

$$\text{RRF}(d) = \sum_{r \in R} \frac{w_r}{k + \text{rank}_r(d)}$$

Where:
* **$d$**: A target document inside the candidate pool.
* **$R$**: The set of retrieval models (e.g., $R = \{\text{BM25}, \text{Dense Vector}\}$).
* **$\text{rank}_r(d)$**: The position index of document $d$ in the output list of retriever $r$ (1-indexed). If a document is absent from a retriever's list, its contribution is $0$.
* **$k$**: A smoothing parameter (typically set to **$60$**). This prevents high-ranked items from overwhelmingly dominating the score and rewards consensus.
* **$w_r$**: Optional weight assigned to retriever $r$ (defaults to $1.0$).

---

### 💻 Python Implementation: Hybrid RRF Merger
Here is a pure Python implementation demonstrating how to merge search results from a keyword ranker and a vector search:

```python
from typing import List, Dict, Any

def reciprocal_rank_fusion(
    sparse_results: List[str], 
    dense_results: List[str], 
    k: int = 60,
    sparse_weight: float = 1.0,
    dense_weight: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Computes Reciprocal Rank Fusion on sparse and dense retrieval lists.
    Returns a sorted list of dictionaries containing doc_id and RRF score.
    """
    rrf_scores: Dict[str, float] = {}

    # Helper function to compute rank scores
    def apply_rrf(results: List[str], weight: float):
        for rank, doc_id in enumerate(results, 1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            # Apply RRF formula: weight * (1 / (k + rank))
            rrf_scores[doc_id] += weight * (1.0 / (k + rank))

    # Score both pipelines
    apply_rrf(sparse_results, sparse_weight)
    apply_rrf(dense_results, dense_weight)

    # Sort documents by score in descending order
    sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    
    return [{"doc_id": doc_id, "rrf_score": score} for doc_id, score in sorted_docs]

# Example Usage
if __name__ == "__main__":
    # Document IDs retrieved via keyword search
    sparse = ["doc_A", "doc_B", "doc_C", "doc_D"]
    # Document IDs retrieved via dense vector search
    dense  = ["doc_C", "doc_E", "doc_A"]
    
    merged = reciprocal_rank_fusion(sparse, dense, k=60)
    print("Fused Results:")
    for rank, doc in enumerate(merged, 1):
        print(f"Rank {rank}: {doc['doc_id']} (Score: {doc['rrf_score']:.5f})")
```

---

### 2. Multi-Vector and Late-Interaction Retrievers (ColBERT)
Traditional dense embeddings collapse an entire document into a single vector (e.g., 1536 dimensions). This causes loss of granular context in long documents. 

**ColBERT (Contextualized Late Interaction over BERT)** addresses this by preserving token-level representation:
* **Token-Level Encoding**: Every token in the query and document receives a low-dimensional vector.
* **Late Interaction Query Math**: 
  Instead of a single cosine dot-product, the similarity is computed as the sum of maximum similarities (MaxSim) between every query token vector $E_q$ and all document token vectors $E_d$:

$$\text{MaxSim}(Q, D) = \sum_{i \in Q} \max_{j \in D} \left( E_{q_i} \cdot E_{d_j}^T \right)$$

This allows the model to align fine-grained terms (e.g., matching a query's "async" directly to a document's "asyncio" token) without losing global query context.

---

### 3. Parsing & Chunking Strategies

#### A. Semantic Chunking
Instead of split-points based on character lengths, semantic chunking calculates embeddings for every sentence, computes a sliding cosine similarity across sentence boundaries, and splits the document only when similarity drops below a dynamic percentile threshold:

```
Sentence 1 ===[ similarity: 0.95 ]=== Sentence 2 ===[ similarity: 0.42 (SPLIT) ]=== Sentence 3
```

#### B. Parent-Child & Small-to-Large Retrieval
* **Concept**:
  * **Child Chunks** ($100$ tokens) are indexed in the Vector DB for high semantic matching precision.
  * **Parent Chunks** ($1000$ tokens) are stored in a Document Store (SQL/NoSQL).
  * **Retrieval**: Match vector queries against Child Chunks, but fetch and feed the complete Parent Chunk to the LLM context window to retain maximum context.

---

## 🛠️ Practical Drills & Competency Benchmarks

- [ ] **Drill 1**: Build a local SQL FTS5 table to index code docstrings, write a Python script that computes dense embeddings, and run the `reciprocal_rank_fusion` method to merge the outputs.
- [ ] **Drill 2**: Implement a semantic chunker using SentenceTransformers, calculate the difference gradient between sentences, and plot split points.
- [ ] **Drill 3**: Implement a dynamic context compression loop that uses an open-source model (like Llama-3) to extract and summarize key terms before sending retrieved text to Claude.
