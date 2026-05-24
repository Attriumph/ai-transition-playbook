# 📊 LLMOps, Observability & Evaluation

## 🎯 The Engineering Approach to LLMs
If you cannot measure your LLM applications, you cannot optimize them. Deploying LLM systems to production requires transition from ad-hoc prompting to strict regression testing, tracing every token, monitoring latency, and auditing costs.

---

## 🗺️ Core Curriculum

### 1. Tracing & Observability
* **Tracing Execution Graphs**:
  * Logging every LLM invocation, tool call, and state transition within complex cyclic agent systems.
  * Capturing input/output token counts, latency breakdowns, and system metadata.
* **LangSmith / Phoenix**:
  * Integrating SDK tracing into LangGraph and LlamaIndex.
  * Analyzing trace traces to diagnose slow operations, high cost, or context bloating.

### 2. LLM-Assisted Evaluations (LLM-as-a-Judge)
* **Continuous Evaluation (CI/CD)**:
  * Running evaluation suites automatically before merging changes to prompts or agent architecture.
* **Ragas Framework**:
  * **Faithfulness**: Is the generated answer grounded *only* in the retrieved context? (Hallucination detection).
  * **Answer Relevance**: Does the generated answer directly address the user query?
  * **Context Recall & Precision**: Did the retrieval system fetch the correct chunks?
* **Promptfoo**:
  * Running local assertions (JSON validations, semantic similarity checks, red-teaming tests) using lightweight, fast-executing pipelines.

### 3. Context & Token Optimization
* **Prompt Caching**:
  * Structuring system prompts and context retrievals to maximize prompt cache hits (Anthropic, OpenAI, DeepSeek specs).
  * Understanding prefix matches and segment alignments.
* **Semantic Caching**:
  * Building a local semantic cache (e.g., using Redis Vector Search or GPTCache) to intercept similar queries before hitting public APIs, keeping latency at sub-10ms.

---

## 🛠️ Practical Drills & Tracker

- [ ] **Drill 1**: Configure LangSmith or Arize Phoenix tracing on a local multi-agent system. Identify the exact node causing the largest latency bottle-neck.
- [ ] **Drill 2**: Create a custom dataset of 20 evaluation queries. Write an evaluation script using Ragas to measure Faithfulness and Context Recall.
- [ ] **Drill 3**: Implement prompt caching on an Anthropic Claude call, verifying via API response headers that the prompt cache was successfully hit.
- [ ] **Drill 4**: Build a semantic cache in Python using a local vector model (like SentenceTransformers) and redis-py, returning cached answers for queries with a cosine similarity > 0.93.

---

## 📚 Resources
* [LangSmith Documentation](https://docs.smith.langchain.com/)
* [Ragas Evaluation Metrics Guide](https://docs.ragas.io/en/stable/concepts/metrics/index.html)
* [Promptfoo Getting Started](https://www.promptfoo.dev/docs/getting-started)
* [Anthropic Prompt Caching Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
