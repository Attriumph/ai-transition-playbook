# 🔄 Agentic Workflows: State Machines & Durable Execution

## 🌌 The 2026 AI Agent Landscape
Simple linear LLM chains are legacy. Modern agentic systems are modeled as **Cyclic Directed Graphs** (State Machines) and robust, distributed workflows. Building production-grade agents requires addressing the core challenges of distributed systems: network partitions, state consistency, long-running transactions, and human intervention (Human-in-the-Loop).

---

## 🗺️ Core Curriculum

### 1. LangGraph & State Machines
Moving away from agent models that hide internal loops (like simple ReAct loops) to explicit, compile-time checked state transition graphs.
* **State & Reducers**:
  * Implementing schema-based central states.
  * Leveraging custom Redux-like reducers to control state updates (appending messages, updating tool lists, tracking token counts).
* **Controlled Cycles**:
  * Modeling complex routing logic via conditional edges based on tool execution status or LLM classifications.
  * Mitigating runaway loops using strict loop counts, max token budgets, and cost policies.
* **Human-in-the-Loop (HITL)**:
  * Interrupt patterns: Pausing execution prior to sensitive actions (e.g., executing writes, payment processing, deleting files).
  * State Editing / Time Travel: Rewinding state, editing message history, and resuming agent executions from historical checkpointers.
* **Persistence & Threading**:
  * Designing multi-tenant checkpointers (using Postgres or Redis) to enable cross-session state persistence.

### 2. Durable Agent Execution with Temporal.io
Why do we need Temporal for agents? When an agent needs to execute a task that takes hours or days (e.g., waiting for code review, processing long-running migrations), LangGraph memory alone is insufficient if the container restarts.
* **Durable Sagas Pattern**:
  * Using Temporal workflows to wrap LangGraph execution steps.
  * Ensuring fault-tolerant, retryable tool executions.
  * Managing state hydration and dehydration across system crashes.
* **Long-Running Waits**:
  * Combining Temporal signals and queries to model agents that pause for human approval, external webhooks, or asynchronous CI/CD runs.

### 3. State Management & Consistency Models
* **Memory Architectures**:
  * Short-term: Ephemeral chat history (in-memory state graph).
  * Long-term: Vector-backed semantic memory + key-value user profile stores.
* **Concurrency Control**:
  * Handling race conditions when multiple users or agents write to the same thread (Optimistic locking vs. queueing).

---

## 🛠️ Practical Drills & Tracker

- [ ] **Drill 1**: Build a ReAct agent using only pure Python dictionary state, custom routers, and manual loop checking (No framework).
- [ ] **Drill 2**: Implement a multi-agent LangGraph setup with a Coordinator and two specialized Workers (Editor, Tester) with Human-in-the-loop validation on the compilation step.
- [ ] **Drill 3**: Wrap a LangGraph agent inside a Temporal.io workflow. Trigger a server reboot mid-workflow to verify that the agent resumes from the exact state without loss.
- [ ] **Drill 4**: Implement a "Time Travel" UI mock that allows reverting the agent's state graph back $N$ steps, modifying the state variables, and re-executing.

---

## 📚 Reference Architectures & Resources
* [LangGraph Documentation (Conceptual Guides)](https://langchainai.github.io/langgraph/concepts/)
* [Temporal.io: Designing Durable Workflows](https://docs.temporal.io/)
* [The Actor Model vs. LLM Agents](https://arxiv.org/abs/2402.03578) (Reflections on state isolation)
