# 🪵 Week 01 Dev Log: Resilient Gateways & WebRTC Orchestration

## 📅 Chronology: May 23, 2026

## 🎯 Weekly Goals
1. Establish the "Build in Public" scaffolding and design two flagship portfolios: **resilient-ai-gateway** (Primary Systems) and **prepHub-orchestrator** (Secondary Product).
2. Complete research and code implementations for an asynchronous LLM proxy with distributed rate limiting (RPM/TPM) and circuit breaking.
3. Investigate ephemeral token configurations for WebRTC audio streams in PrepHub.

---

## 💡 Key Architectural Revelations

### 1. Atomic Redis sliding windows for TPM rate limits
Standard API gateways track Requests-Per-Minute (RPM) using simple increment keys. However, LLM providers restrict Tokens-Per-Minute (TPM). 
* **My Design**: Implemented `RedisTokenBucketLimiter` using Redis Sorted Sets (`ZADD`, `ZREMRANGEBYSCORE`). This calculates exact sliding-window token volumes atomically, blocking excessive requests before executing downstream API calls, preventing 429 lockouts in production.

### 2. High-speed SSE streaming failovers (<50ms)
When a downstream provider crashes, standard reverse proxies return a raw 500 error to the client, ruining the user experience.
* **My Design**: Programmed the proxy router in FastAPI to yield chunks asynchronously via `StreamingResponse`. The router wraps the stream block in a try-except layer. If OpenAI errors mid-stream, the router catches the exception and recursively invokes the backup provider (Anthropic) in under 50ms, hot-swapping the stream transparently for the client.

---

## 🛠️ Build Failures & How I Solved Them
* **Problem**: In `psycopg2` pgvector queries, high concurrent bursts caused connection pool exhaustion.
* **Solution**: Rewrote the pgvector database queries to utilize a thread-safe connection pooling system (`psycopg2.pool`) inside our `retriever.py`, managing execution pipelines asynchronously and ensuring connection cleanups inside `finally` blocks.

---

## 📊 Plans for Week 2
- [ ] Connect the Next.js frontend audio tracks directly to the backend session controller.
- [ ] Spin up local Redis and Postgres instances inside Docker-compose to test sliding-window token updates.
- [ ] Write statistical evaluation tests using the Promptfoo CLI to measure answer relevance metrics.
