# 🪵 Week 01 Dev Log: Pushing MCP & LangGraph Boundaries

## 📅 Chronology: May 23, 2026

## 🎯 Weekly Goals
1. Establish the "Build in Public" scaffolding to track my Applied AI Staff journey.
2. Complete research on the **Model Context Protocol (MCP)** limits, specifically investigating SSE (Server-Sent Events) vs Stdio transport overheads.
3. Architecture design for `repo-migrator-agent`.

---

## 💡 Key Architectural Revelations

### 1. The Real Cost of Local Stdio vs. SSE
When writing MCP servers, the default quickstart recommends **Stdio transport** (child processes writing JSON-RPC to standard output). 
* **The Good**: It's highly secure and requires no open ports. When the parent CLI tool exits, the server process terminates instantly.
* **The Bad**: In a micro-services or multi-agent orchestrator setup, launching processes for every file read or compile step introduces massive initialization delays (especially with Node/Python cold starts).
* **The Shift**: Moving forward, the flagship `repo-migrator-agent` compiler tools will run via an **SSE-based HTTP server** inside a persistent Docker container. This gives us sub-millisecond JSON-RPC round-trip times and enables robust container scaling.

### 2. Guarding AST-Level Read Tools
When feeding code bases to LLMs, sending entire raw files is a waste of money and pollutes the context window.
* **Insight**: Built a custom file parser tool schema in the `mcp-server-fs` server. Instead of a simple `read_file(path)` tool, I designed:
  * `read_symbol_ast(path, symbol_name)`: Returns only the target class or method definition.
  * `find_references(path, regex_query)`: Mimics classical LSP behavior to trace dependencies across the project.

---

## 🛠️ Build Failures & How I Solved Them
* **Problem**: The LLM frequently generated invalid JSON schema syntax when making tool calls through MCP, especially when double-quoting array arguments inside a nested bash call.
* **Solution**: Implemented an explicit validation node in the LangGraph coordinator immediately following the LLM generation step. This node intercepts tool calls, parses them, repairs minor quoting errors, and returns an instructive syntax error message to the LLM *before* executing the tool, preventing the orchestrator from crashing.

---

## 📊 Plans for Week 2
- [ ] Write the core Node/TypeScript codebase for `mcp-servers/mcp-server-fs`.
- [ ] Hook up Qdrant vector database and write the multi-vector parsing utility in Python.
- [ ] Record the first demo of a self-correcting import refactoring task.
