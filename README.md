# 🧠 AI Transition Playbook: Distributed Systems Expert ➡️ Applied AI Builder

> **Mission**: Documenting my transition from Staff Software Engineer (Distributed Systems, Infrastructure, High-Throughput Services) to Senior/Staff Applied AI Engineer. Building in public, grounding LLMs in solid software engineering practices, and mastering the 2026 AI Engineer tech stack.

---

## 🧭 Master Dashboard

### 🎯 The Core Thesis
AI Engineering is transitioning from simple "prompt engineering wrappers" to complex, deterministic, stateful distributed systems. The next generation of AI systems requires orchestrating millions of tokens, managing complex multi-agent execution graphs, building resilient retrieval pipelines, and setting up strict, reproducible evaluations. 

My background in distributed consensus (Raft/Paxos), state machines, event-driven architectures, and high-performance caching is not obsolete—it is the *exact* foundation needed to build robust, production-grade Applied AI.

---

## 📈 Transition Tracks & Directory Map

| Track | Directory / Resource | Description | Status |
| :--- | :--- | :--- | :--- |
| **01** | [**Learning Roadmap**](file:///Users/lingquan/Projects/ai-transition-playbook/01-Learning-Roadmap) | Tech stack curriculum: Agentic Workflows, Advanced RAG, MCP, and LLMOps. | 🏃‍♂️ In Progress |
| **02** | [**Architecture Notes**](file:///Users/lingquan/Projects/ai-transition-playbook/02-Architecture-Notes) | High-level technical reflections (e.g., Context Window Economics, Multi-Agent Patterns). | 📝 Drafting |
| **03** | [**Portfolio Projects**](file:///Users/lingquan/Projects/ai-transition-playbook/03-Portfolio-Projects) | Production-grade codebases. Flagship: `repo-migrator-agent`. | 🛠️ Scaffolding |
| **04** | [**Interview Prep**](file:///Users/lingquan/Projects/ai-transition-playbook/04-Interview-Prep) | Staff AI System Design mockups, Live Coding drills, and my AI Vision. | 🎯 Active |
| **05** | [**Dev Logs**](file:///Users/lingquan/Projects/ai-transition-playbook/05-Dev-Logs) | Weekly chronological build logs tracking insights, failures, and breakthroughs. | 🪵 Week 1 |

---

## 🏗️ Active Portfolios

### ⚡ Primary Systems Portfolio: `resilient-ai-gateway`
* **Path**: [`03-Portfolio-Projects/resilient-ai-gateway/`](file:///Users/lingquan/Projects/ai-transition-playbook/03-Portfolio-Projects/resilient-ai-gateway)
* **Stack**: FastAPI (ASGI) + Redis (Sliding Token Buckets) + Stateful Circuit Breaker + Stream failovers.
* **Problem Statement**: Native LLM API endpoints are slow, lack failover capabilities during provider outages, and enforce strict, easily-tripped Requests-Per-Minute and Tokens-Per-Minute limits.
* **Solution**: A high-throughput reverse-proxy gateway that handles distributed token-bucket rate limiting, sub-millisecond semantic caching, and a stateful circuit breaker that hot-swaps streams in flight (<50ms failover) if a provider crashes.

### 🎙️ Secondary Product Portfolio: `prepHub-orchestrator`
* **Path**: [`03-Portfolio-Projects/prepHub-orchestrator/`](file:///Users/lingquan/Projects/ai-transition-playbook/03-Portfolio-Projects/prepHub-orchestrator)
* **Stack**: OpenAI Realtime API (WebRTC) + LangGraph Multi-Agent Engine + pgvector (Supabase).
* **Problem Statement**: Standard mock interview platforms rely on slow, linear asynchronous text interfaces, leading to unnatural conversational delays and generic, unchallenging coaching metrics.
* **Solution**: A high-speed, sub-500ms conversational audio broker built with OpenAI Realtime WebRTC, driven by a stateful multi-agent LangGraph system that dynamically researches target companies using web-search tools and aggregates STAR performance scoring.

---

## 🛠️ Tech Stack Competency Tracker

- [ ] **Agentic Workflows**: LangGraph (State Graph, Redux-like reducers, Human-in-the-loop), Temporal.io for durable agent execution.
- [ ] **Retrieval-Augmented Generation (RAG)**: Hybrid Search (Sparse/Dense), Dense Vector DBs (Qdrant/pgvector), Cohere Rerank, Cohere/ColBERT multi-vector retrieval.
- [ ] **Protocols**: Model Context Protocol (MCP) servers, custom tool definitions, transport layers (SSE vs Stdio).
- [ ] **LLMOps & Evaluation**: LangSmith for tracing, Ragas/Promptfoo for LLM-assisted evaluations, prompt caching optimization, semantic caching.

---

## 🪵 Latest Build Logs
* **Week 1**: Exploring MCP protocol limits and SSE transports. See [`05-Dev-Logs/week-01-mcp-exploration.md`](file:///Users/lingquan/Projects/ai-transition-playbook/05-Dev-Logs/week-01-mcp-exploration.md)
* **Goal**: Establish a baseline multi-agent orchestrator with error self-correction.

---
*Follow my journey live. Built with rigor, documented in public.*
