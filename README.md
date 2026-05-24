# 🧠 AI Transition Playbook: Distributed Systems Expert ➡️ AI Infrastructure & Applied AI

> **Mission**: Documenting my transition from Staff Software Engineer (Distributed Systems, Infrastructure, High-Throughput Services) to **Senior/Staff AI Infrastructure Engineer**. Building in public, translating deep systems expertise into the highest-moat AI skillsets of 2026.

---

## 🧭 Master Dashboard

### 🎯 The Core Thesis
The 2026 AI industry has bifurcated into two tiers:

1. **Application-layer AI** (RAG wrappers, chatbots, prompt chains) — increasingly automatable by AI agents themselves.
2. **Infrastructure-layer AI** (inference engines, GPU orchestration, distributed training, model serving) — requires deep systems knowledge that AI agents **cannot** replicate.

My background in distributed consensus (Raft/Paxos), state machines, event-driven architectures, and high-performance caching maps **directly** onto the hardest, highest-value problems in AI today: KV cache memory management, continuous batching schedulers, tensor/pipeline parallelism, and fault-tolerant distributed training.

**This playbook follows a dual-track strategy:**

| Track | Focus | Moat Level |
| :--- | :--- | :--- |
| 🔧 **AI Infrastructure** (Primary) | Inference optimization, GPU systems, serving engines | 🏔️ Very High |
| 🧠 **Applied AI** (Secondary) | Multimodal agents, RAG, LLMOps/Evaluation | 🏕️ Moderate |

---

## 📈 Transition Tracks & Directory Map

| Track | Directory / Resource | Description | Status |
| :--- | :--- | :--- | :--- |
| **01** | [**Learning Roadmap**](./01-Learning-Roadmap) | Dual-track curriculum: AI Infra (Inference, GPU, Distributed Training) + Applied AI (Agents, RAG, MCP, LLMOps). | 🏃‍♂️ In Progress |
| **02** | [**Architecture Notes**](./02-Architecture-Notes) | Technical deep-dives: PagedAttention internals, KV Cache economics, Multi-Agent failure modes. | 📝 Drafting |
| **03** | [**Portfolio Projects**](./03-Portfolio-Projects) | Production-grade codebases demonstrating both tracks. | 🛠️ Scaffolding |
| **04** | [**Interview Prep**](./04-Interview-Prep) | Staff AI System Design mockups, Live Coding drills, and technical vision. | 🎯 Active |
| **05** | [**Dev Logs**](./05-Dev-Logs) | Weekly chronological build logs tracking insights, failures, and breakthroughs. | 🪵 Week 1 |

---

## 🏗️ Active Portfolios

### 🔥 Track A — AI Infra: `mini-inference-engine`
* **Path**: [`03-Portfolio-Projects/mini-inference-engine/`](./03-Portfolio-Projects/mini-inference-engine)
* **Stack**: Python 3.11 + PyTorch + Triton + FastAPI.
* **Problem Statement**: Naive LLM inference wastes 60-80% of GPU memory through pre-allocation fragmentation and achieves only ~30% GPU utilization with static batching.
* **Solution**: A pedagogical but production-grade implementation of the three core algorithms powering vLLM and SGLang: **PagedAttention** (OS-style virtual memory for KV cache), **Continuous Batching** (iteration-level request scheduling), and a **KV Cache Memory Calculator**. Includes benchmarks proving the throughput gains.
* **Why this matters**: This project directly translates distributed systems expertise (memory management, scheduling, resource allocation) into the #1 most in-demand AI Infra skillset of 2026.

### 🎙️ Track B — Applied AI: `prepHub-orchestrator` (Multimodal Vision Enhanced)
* **Path**: [`03-Portfolio-Projects/prepHub-orchestrator/`](./03-Portfolio-Projects/prepHub-orchestrator)
* **Stack**: OpenAI Realtime API (WebRTC) + Vision API + LangGraph Multi-Agent Engine + pgvector (Supabase).
* **Problem Statement**: Standard mock interview platforms rely on slow, linear asynchronous text interfaces. They also fail to evaluate visual components of an interview, such as whiteboard system designs or coding screencasts.
* **Solution**: An advanced, sub-500ms conversational audio broker built with WebRTC. It features a **Multimodal Interview Agent** capable of "seeing" user-uploaded diagrams or screen captures in real-time. The system uses a stateful multi-agent LangGraph to dynamically evaluate visual architectures alongside verbal explanations, creating an uncompromising, realistic "Senior/Staff" interview experience.

---

## 📚 Learning Roadmap Overview

### 🔧 Track A: AI Infrastructure (High Moat)
| Module | File | Core Topics |
| :--- | :--- | :--- |
| **05** | [AI Inference Systems](./01-Learning-Roadmap/05-AI-Inference-Systems.md) | KV Cache math, PagedAttention, Continuous Batching, Speculative Decoding, Quantization |
| **06** | [GPU Systems & Distributed Training](./01-Learning-Roadmap/06-GPU-Systems-and-Distributed-Training.md) | CUDA/Triton, Tensor/Pipeline/Data Parallelism, NCCL, Mixed Precision, Fault Tolerance |

### 🧠 Track B: Applied AI (Complementary)
| Module | File | Core Topics |
| :--- | :--- | :--- |
| **01** | [Agentic Workflows](./01-Learning-Roadmap/01-Agentic-Workflows.md) | ReAct (POMDP), DSPy, MCTS, LangGraph + Temporal architecture |
| **02** | [Advanced RAG](./01-Learning-Roadmap/02-Advanced-RAG.md) | HNSW math, ColBERT/PLAID, Hybrid Search (BM25+SPLADE), Cross-Encoder Reranking |
| **03** | [Protocols & MCP](./01-Learning-Roadmap/03-Protocols-MCP.md) | JSON-RPC spec, Stdio vs SSE, Security, Tool-RAG scaling |
| **04** | [LLMOps & Evaluation](./01-Learning-Roadmap/04-LLMOps-and-Eval.md) | RAGAS metrics, LLM-as-Judge bias, OpenTelemetry, KV Cache economics, CI/CD for LLMs |

---

## 🛠️ Tech Stack Competency Tracker

### AI Infrastructure (Primary)
- [ ] **Inference Optimization**: KV Cache management, PagedAttention, Continuous Batching, Speculative Decoding, FP8/INT4 Quantization.
- [ ] **GPU Systems**: CUDA/Triton kernels, memory hierarchy (HBM/SRAM), Roofline Model analysis.
- [ ] **Distributed Training**: Tensor Parallelism (Megatron), Pipeline Parallelism (GPipe), FSDP/ZeRO, NCCL AllReduce.
- [ ] **Serving Frameworks**: vLLM, SGLang (RadixAttention), TensorRT-LLM, Triton Inference Server.
- [ ] **Orchestration**: Ray, Kubernetes GPU scheduling, model routing, cost-per-token optimization.

### Applied AI (Secondary)
- [ ] **Multimodal AI**: Vision-language models (VLM), real-time audio (WebRTC integrations), spatial/layout grounding.
- [ ] **Agentic Workflows**: LangGraph (State Graph, Redux-like reducers, Human-in-the-loop), Temporal.io for durable agent execution.
- [ ] **Retrieval-Augmented Generation (RAG)**: Hybrid Search (Sparse/Dense), Dense Vector DBs (Qdrant/pgvector), ColBERT multi-vector retrieval.
- [ ] **Protocols**: Model Context Protocol (MCP) servers, custom tool definitions, transport layers (SSE vs Stdio).
- [ ] **LLMOps & Evaluation**: LangSmith for tracing, RAGAS/DeepEval for evaluations, prompt caching optimization.

---

## 🎯 Open-Source Contribution Targets

Contributing to these projects is the ultimate proof of AI Infra competence:

| Project | Contribution Focus | Difficulty |
| :--- | :--- | :--- |
| [**vLLM**](https://github.com/vllm-project/vllm) | Model support, test coverage, PagedAttention improvements | ⭐⭐⭐ |
| [**SGLang**](https://github.com/sgl-project/sglang) | RadixAttention, structured generation, DeepSeek optimizations | ⭐⭐⭐ |
| [**Ray**](https://github.com/ray-project/ray) | Ray Serve for model serving, GPU scheduling | ⭐⭐ |

---

## 🪵 Latest Build Logs
* **Week 1**: Exploring MCP protocol limits and SSE transports. See [`05-Dev-Logs/week-01-mcp-exploration.md`](./05-Dev-Logs/week-01-mcp-exploration.md)
* **Goal**: Establish a baseline multi-agent orchestrator with error self-correction.

---
*Follow my journey live. Built with rigor, documented in public.*
