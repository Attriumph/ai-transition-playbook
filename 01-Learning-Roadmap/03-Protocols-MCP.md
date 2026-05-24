# 🔌 Model Context Protocol (MCP): The Universal Interface

## 💡 The Unified Tool Standard
Historically, connecting LLMs to custom enterprise data sources and local developer tools required writing fragile, framework-specific adapter layers (LangChain tools, LlamaIndex data loaders, custom OpenAI functions). 

The **Model Context Protocol (MCP)** standardizes this interface. Operative like the Language Server Protocol (LSP) in IDEs, MCP defines a structured, JSON-RPC 2.0-based contract separating the **LLM Orchestration Client** from isolated **Context & Tool Servers**.

---

## 🗺️ Core Curriculum & Architectural Deep Dives

### 1. JSON-RPC 2.0 Protocol Handshake & Execution Frames
MCP clients and servers communicate by exchange of standard JSON-RPC 2.0 messages. Let's study the raw protocol packets for a tool execution request and response.

#### 🛰️ Client Tool Discovery Request
When the client boots, it queries the server to fetch all available tools and their respective JSON Schema argument expectations:

```json
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "id": 1
}
```

#### 📥 Server Discovery Response
The server returns a structured array of schemas:

```json
{
  "jsonrpc": "2.0",
  "result": {
    "tools": [
      {
        "name": "parse_python_ast",
        "description": "Extracts high-level syntax nodes (classes, function names) from a Python file.",
        "inputSchema": {
          "type": "object",
          "properties": {
            "path": {
              "type": "string",
              "description": "Absolute filesystem path to the Python file."
            }
          },
          "required": ["path"]
        }
      }
    ]
  },
  "id": 1
}
```

#### 🛰️ Client Execution Request (Invoking Tool)
When the LLM decides to execute the tool, the client serializes the parameters and sends the invocation call:

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "parse_python_ast",
    "arguments": {
      "path": "/Users/lingquan/Projects/ai-transition-playbook/orchestrator/main.py"
    }
  },
  "id": 2
}
```

---

### 2. Transport Layer Deep Dive: Stdio vs. SSE
MCP isolates transport logic from the protocol schema. There are two standard transport layers:

#### A. Stdio (Standard Input/Output)
* **Under the Hood**: The client spawns the server process locally (e.g., via `subprocess.Popen`) and sends raw JSON lines via stdin, reading response frames via stdout.
* **Best For**: Local IDE extensions (Cursor, VSCode), desktop assistant runtimes (Claude Desktop).
* **Limitations**: Highly coupled; cannot easily run in cloud container architectures or connect to remote infrastructure without SSH tunneling.

#### B. SSE (Server-Sent Events) HTTP Transport
* **Under the Hood**: The server runs as a standard web server. The client establishes a persistent HTTP connection using SSE to receive events streamed from the server. Client writes to the server using standard HTTP POST requests.
* **Best For**: Distributed AI microservices, cloud-native deployments, secure sandboxed execution environments.

```
+------------+       HTTP POST /messages (Client Sends Data)      +------------+
|            | -------------------------------------------------> |            |
| MCP Client |                                                    | MCP Server |
|            | <------------------------------------------------- |            |
+------------+   Persistent HTTP /sse Stream (Server Sends Data)  +------------+
```

---

### 3. Security, Sandboxing & Zero-Trust Architectures
Giving an LLM access to tools (like shell execution or database writes) introduces significant security risks (Prompt Injection exploits).
* **Network Isolation**: Run remote SSE MCP servers inside strictly sandboxed Docker containers with disabled outbound internet access unless explicitly required.
* **Mutual TLS (mTLS)**: Force cryptographic validation on both client and server sides to secure remote SSE channels.
* **Write Gates**: Implement a tokenized cryptographic validation loop. When a tool requests a mutating change (e.g., `write_file` or `delete_record`), the client interrupts the graph and requires a signed user approval before returning the response payload.

---

## 🛠️ Practical Drills & Competency Benchmarks

- [ ] **Drill 1**: Read the complete Model Context Protocol JSON-RPC specification. Write a simple Python TCP socket server that manually handles client handshakes without utilizing the official SDK.
- [ ] **Drill 2**: Build an SSE-based Node.js or Python FastMCP server, expose it over a local port, and connect to it using a standard client web interface to monitor JSON-RPC payload events.
- [ ] **Drill 3**: Implement a local "sandbox file explorer" tool that strictly restricts the directory path argument of all filesystem operations to a designated sub-folder (jailbreak prevention).
- [ ] **Drill 4**: Integrate your custom `fs_server.py` directly into Claude Desktop by configuring the local `claude_desktop_config.json` file.
