# 07 - Open-Source Contribution Roadmap (vLLM)

> **Goal:** Transition from Frontend / API Developer to AI Infrastructure Engineer via high-value open-source contributions.

## Why Open-Source Contribution Beats "Toy Projects"

When interviewing for Senior/Staff AI Infrastructure roles, a "toy" implementation of an inference engine (like a mini-vLLM) is good for personal learning, but it carries low signal for recruiters. A merged PR to a production-grade inference engine like **vLLM** (12k+ stars, used by major enterprises) is the strongest possible signal on a resume.

Given a background in **Frontend, API Development, and Distributed Systems concepts**, the strategy is to attack the codebase from the "outside in," starting with familiar API layers and moving deeper into scheduling and memory management, avoiding CUDA/kernel code until much later.

---

## The 3-Month Realistic Contribution Roadmap

### Month 1: Learn the Concepts & Read the Python Layer

**Objective:** Understand the architecture of vLLM without writing code.

1. **Week 1: Theoretical Foundation**
   - Read `05-AI-Inference-Systems.md`. Understand KV Cache, PagedAttention, and Continuous Batching conceptually.
2. **Week 2: Run vLLM Locally**
   - Install vLLM (`pip install vllm`).
   - Run the OpenAI-compatible server locally with a small model (e.g., `Qwen/Qwen2-0.5B`).
   - Send requests using standard `curl` or the OpenAI Python client.
3. **Week 3: Codebase Tour — The API Layer**
   - Explore `vllm/entrypoints/openai/`. 
   - Observe that the API server is written in **FastAPI**—this is directly applicable to existing API development skills.
4. **Week 4: Codebase Tour — The Scheduler**
   - Explore `vllm/core/scheduler.py` and `vllm/core/block_manager.py`.
   - Observe that these are pure Python modules managing queues and states—this maps directly to distributed systems knowledge.

### Month 2: Your First PR (The "Comfort Zone" Strategy)

**Objective:** Get a PR merged by leveraging your existing API development expertise.

**The Strategy:** vLLM's `v1/chat/completions` API server frequently needs bug fixes, parameter support updates, and test coverage. This is pure Python/FastAPI work.

1. **Find an Issue:**
   - Search the vLLM GitHub issues for the label `good first issue`.
   - Search for keywords like `api`, `openai`, `server`, or `fastapi`.
2. **Target PR Types:**
   - **Bug Fixes:** E.g., fixing missing fields in the JSON response when streaming is enabled.
   - **Documentation:** Clarifying API deployment parameters in the docs.
   - **Testing:** Adding `pytest` coverage for edge cases in the API endpoints.
3. **Execution:**
   - Comment on the issue to claim it.
   - Write the fix, ensure tests pass, and submit the PR.
   - A single merged PR allows you to legitimately claim: **"Open-Source Contributor to vLLM (API Server layer)"**.

### Month 3: Moving Deeper into Systems Infrastructure

**Objective:** Apply distributed systems knowledge to the core orchestration logic.

Once comfortable with the repository's CI/CD and review process, target the Python-based infrastructure layers:

1. **The Scheduler (`vllm/core/scheduler.py`)**
   - Look for issues related to request prioritization, preemption logic, or batch construction.
2. **The Memory Manager (`vllm/core/block_manager.py`)**
   - Look for issues related to KV cache block allocation, freeing, and swapping to CPU RAM.
3. **Observability & Metrics**
   - vLLM exposes Prometheus metrics. Look for issues related to tracking new metrics (e.g., Time-To-First-Token histograms, GPU utilization tracking).

---

## Summary of the Mental Model

```text
Your Current Skills                        Target Identity
      │                                          │
      ▼                                          ▼
[Frontend & API Dev]                 [AI Infra Engineer]
      │                                          ▲
      │── Month 1: Run vLLM + Read Python Code ──│
      │── Month 2: Contribute to FastAPI Layer ──│
      │── Month 3: Contribute to Scheduler/Mem ──│
      │── Month 4+: Learn Triton/CUDA kernels ───│
```

**Key Takeaway:** You do not need to be a C++/CUDA expert to contribute to the world's leading AI infrastructure project. The orchestration, API, and scheduling layers are written in Python and desperately need engineers with standard backend/systems expertise.
