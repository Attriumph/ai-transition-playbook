# 🔌 Model Context Protocol (MCP): The Universal Interface

## 💡 The Paradigm Shift
As LLMs shift from chatbots to autonomous agents, tool use has historically been highly fragmented. Every framework had its own format. The **Model Context Protocol (MCP)**, open-sourced by Anthropic, introduces a standardized, JSON-RPC-based client-server architecture for LLM tools, resources, and prompts. It operates similarly to the Language Server Protocol (LSP) for editors.

---

## 🗺️ Core Curriculum

### 1. Architectural Blueprint
MCP decouples the **LLM Client** (which acts as the orchestrator and orchestrates prompt routing) from the **MCP Servers** (which expose localized resources, tools, and custom prompt templates).
* **Key Components**:
  * **MCP Host/Client**: Connects to one or more MCP servers. Examples: Claude Desktop, Cursor, Custom orchestrator.
  * **MCP Server**: Lightweight services exposing endpoints.
  * **JSON-RPC 2.0**: The protocol transport layer message format.

### 2. The Core Primitives
MCP defines three primary interfaces that servers can expose:
* **Resources**: Readable data sources (e.g., local files, database tables, git repositories, API responses).
* **Tools**: Executable actions with JSON Schema definitions (e.g., running shell commands, editing a file, compiling code, triggering webhooks).
* **Prompts**: Standardized prompt templates with user arguments (e.g., code-review template, refactor template).

### 3. Transport Protocols
* **Stdio Transport**: 
  * Local communication.
  * Host launches the server process via stdin/stdout.
  * Perfect for local developer agents and desktop applications.
* **Server-Sent Events (SSE) Transport**:
  * Over-the-air communication.
  * Client connects via HTTP POST and receives streamed SSE responses.
  * Required for cloud-native agents, remote databases, and distributed MCP servers.

### 4. Security & Isolation
* **Sandboxing**: Restricting MCP tools from executing arbitrary commands directly on the host machine.
* **Authentication**: Securing SSE endpoints with Bearer tokens and mutual TLS (mTLS).
* **Permission Model**: Designing user-approval gates before executing mutating tools.

---

## 🛠️ Practical Drills & Tracker

- [ ] **Drill 1**: Build a basic Python MCP server using the `@mcp` decorator that exposes a single resource: `/system/metrics` returning live CPU, Memory, and Disk usage of the host machine.
- [ ] **Drill 2**: Build an SSE-based MCP server in Node.js/TypeScript that connects to a local SQLite database. Deploy it locally and access it using Claude Desktop by editing the `claude_desktop_config.json`.
- [ ] **Drill 3**: Create a custom MCP client (in Python) that connects to a local Stdio MCP server, parses its JSON schema tools, executes tool calls outputted by an LLM, and sends back the result.
- [ ] **Drill 4**: Build a secure execution sandbox for an MCP shell executor using a Docker container, routing shell commands from the MCP client to the containerized environment.

---

## 📚 Resources
* [Model Context Protocol Specification & Docs](https://modelcontextprotocol.io/)
* [Anthropic Quickstart: Building MCP Servers](https://github.com/modelcontextprotocol/quickstart)
* [Awesome MCP: Curated List of Servers](https://github.com/punkpeye/awesome-mcp)
