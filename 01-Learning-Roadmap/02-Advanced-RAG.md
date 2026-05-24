# 🔍 Advanced Retrieval-Augmented Generation (RAG)

## 🏗️ Beyond Naive RAG
Naive vector retrieval (chunking documents, embedding them, and performing cosine similarity) is brittle in production. Production RAG systems require sophisticated query preprocessing, high-recall retrieval, precision filtering, reranking, and post-retrieval synthesis.

---

## 🗺️ Core Curriculum

### 1. Advanced Chunking & Parsing Strategies
* **Document Parsing**:
  * Handling multi-modal documents (PDFs with tables/charts) using partition engines like Unstructured or LlamaParse.
  * Hierarchical parsing to build node trees of documents.
* **Chunking Algorithms**:
  * **Recursive Character Text Splitting**: The baseline standard.
  * **Semantic Chunking**: Splitting text based on embedding drift/shifts between sentences to keep coherent ideas together.
  * **Parent-Child & Small-to-Large Retrieval**: Creating small sub-chunks for vector matching, but returning the larger parent chunk to the LLM to preserve context.

### 2. Multi-Vector and Late-Interaction Retrievers
* **Hybrid Search (Sparse + Dense)**:
  * Combining exact-match BM25/keyword indices with vector embeddings.
  * Merging results using Reciprocal Rank Fusion (RRF) with normalized weights.
* **ColBERT & Late Interaction**:
  * Understanding ColBERT's approach of generating token-level embeddings instead of a single document vector.
  * Utilizing late interaction to evaluate similarities between query token and document token matrices, optimizing recall for long and complex documents.

### 3. Query Transformation & Pre-Retrieval
* **Query Rewriting**: Using light, fine-tuned LLMs to translate user intent (e.g., chat history context) into standalone, optimized search queries.
* **Query Expansion / HyDE (Hypothetical Document Embeddings)**:
  * Generating a hypothetical answer first, then using that answer's embedding to search the vector database.
* **Query Routing**: Analyzing queries to route them dynamically to specific data sources (e.g., Vector DB vs. SQL DB vs. Web Search).

### 4. Post-Retrieval Optimization & Reranking
* **Cross-Encoder Rerankers**:
  * Using cross-encoders (e.g., Cohere Rerank, BGE-Reranker) to evaluate the query and the candidate documents jointly, providing high-precision scoring.
* **Metadata Filtering**:
  * Extracting metadata (dates, departments, tags) dynamically from user queries to apply pre-filtering or post-filtering, drastically reducing search space.
* **Context Compressors**:
  * Shrinking retrieved context chunks using token-level compression (LLMLingua) to remove fluff and optimize context window economics.

---

## 🛠️ Practical Drills & Tracker

- [ ] **Drill 1**: Build a hybrid search pipeline from scratch utilizing `sqlite3` for FTS5 (BM25) search and `qdrant-client` for dense vector search. Combine using RRF.
- [ ] **Drill 2**: Implement a semantic chunker using `numpy` to calculate cosine differences between sliding windows of sentences, setting dynamic break thresholds.
- [ ] **Drill 3**: Create a parent-child retriever using `LlamaIndex` or `LangChain` where chunks are $100$ tokens, but the retrieved content returned to the LLM context is $1000$ tokens of surrounding paragraph block.
- [ ] **Drill 4**: Benchmarking RAG performance: Use a dataset and test accuracy improvements when inserting a Cohere Rerank step after raw vector retrieval.

---

## 📚 Resources
* [LlamaIndex Production RAG Guide](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)
* [Qdrant Hybrid Search & RRF Tutorial](https://qdrant.tech/documentation/concepts/hybrid-search/)
* [ColBERT: Efficient and Effective Late Interaction](https://arxiv.org/abs/2004.12832)
