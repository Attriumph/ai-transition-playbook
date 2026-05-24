# 💻 Live Coding Drills: Self-Correction & Core Agent Primitives

This section contains coding drills designed to build the "muscle memory" needed for hands-on AI engineering coding interviews. I focus on implementing robust, self-correcting logic, JSON sanitizers, and token rate limiters from scratch without using heavy agent frameworks.

---

## 🛠️ Drill 1: The Self-Correcting Python Execution Loop
* **Objective**: Write a pure Python class `SelfCorrectingAgent` that receives a natural language prompt, writes a Python script to solve it, runs the script in a secure subprocess, parses the stack trace if it crashes, and recursively prompts itself to fix the error until it completes successfully or hits a retry threshold.
* **Why it matters**: Demonstrates cyclic state graphs, shell execution, error parsing, and recursion limits.

### Reference Signature:
```python
import subprocess
import json
import sys

class SelfCorrectingAgent:
    def __init__(self, model_client, max_retries: int = 3):
        self.client = model_client
        self.max_retries = max_retries
        
    def execute_and_correct(self, task_description: str) -> str:
        # Loop starts here:
        # 1. Ask LLM to generate code block
        # 2. Extract code block via regex
        # 3. Write to temporary file
        # 4. Run subprocess: python temp.py
        # 5. If exit code != 0, capture stderr and recurse with error log
        pass
```

---

## 🛠️ Drill 2: Robust JSON Repair & Parsing Agent
* **Objective**: LLMs often output markdown-wrapped JSON or invalid JSON (missing trailing commas, double-quoted keys inside single-quoted strings, unescaped characters). Write a robust parser that cleans and extracts the JSON block from raw LLM text, and if `json.loads` still fails, uses a lightweight LLM call to correct *only* the syntax errors.
* **Why it matters**: Crucial for production deterministic routing.

### Reference Code Sketch:
```python
import re
import json

def extract_and_parse_json(raw_llm_output: str, llm_fix_fn=None) -> dict:
    # 1. Regex find ```json ... ``` or first { to last }
    # 2. Replace common JSON faults (e.g., trailing commas, bad string wrapping)
    # 3. If json.loads fails and llm_fix_fn is present, feed syntax exception to it
    pass
```

---

## 🛠️ Drill 3: Token Bucket Rate Limiter for LLM Endpoints
* **Objective**: Implement a thread-safe token bucket rate limiter in Python that tracks two separate constraints: Requests-Per-Minute (RPM) and Tokens-Per-Minute (TPM) before dispatching requests to external API clients.
* **Why it matters**: Production agent systems will easily hit rate limits if a parallel Map-Reduce workflow is triggered.

---

## 🛠️ Drill 4: Local RAG Matcher from Raw Arrays
* **Objective**: Using only `numpy` and `math`, implement a complete semantic search searcher. Given a list of sentences, calculate their embeddings using a local library, normalize the matrices, and perform cosine similarity matching against a query vector, returning top $K$ items without importing Pinecone, Qdrant, or LangChain.
