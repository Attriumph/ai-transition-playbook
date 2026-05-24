# 🪵 Week 01 Dev Log: Pushing WebRTC & LangGraph Boundaries

## 📅 Chronology: May 23, 2026

## 🎯 Weekly Goals
1. Establish the "Build in Public" scaffolding and link it to my full-stack platform: **PrepHub**.
2. Complete research on the **OpenAI Realtime API (WebRTC) sessions**, specifically testing the `/sessions` ephemeral token exchange mechanisms.
3. Design and implement the multi-agent LangGraph interview coordinator topology.

---

## 💡 Key Architectural Revelations

### 1. Ephemeral Tokens for WebRTC Client Safety
In our early Next.js mock-ups, calling OpenAI directly from the browser meant exposing the master `OPENAI_API_KEY` in headers, creating a critical vulnerability.
* **The Solution**: Designed a secure session bridge inside `/realtime/session.py`. The Next.js client requests an ephemeral token from our FastAPI backend, passing a target interview schema. The backend uses the master key to hit `https://api.openai.com/v1/realtime/sessions`, retrieves a short-lived key (expires in 1 minute), and streams it back to the client. This enforces zero-trust token isolation in production.

### 2. Eliminating Conversational Delay
Standard mock interview queues suffer from a 3-5s "Cascaded Lag" (STT -> LLM reasoning -> TTS). By implementing the **OpenAI Realtime WebRTC direct connection**, audio packages are sent natively. The latency dropped from **4,200ms** to **sub-450ms**, making the AI conversational flow indistinguishable from a real human interaction.

---

## 🛠️ Build Failures & How I Solved Them
* **Problem**: When loading multiple resumes into the pgvector database for testing, concurrent database connections caused thread exhaustion, locking up the FastAPI pool.
* **Solution**: Rewrote the pgvector database queries to utilize a thread-safe connection pooling system (`psycopg2.pool`) inside our `retriever.py`, managing execution pipelines asynchronously and ensuring connection cleanups inside `finally` blocks.

---

## 📊 Plans for Week 2
- [ ] Connect the Next.js frontend audio tracks directly to the backend session controller.
- [ ] Integrate full-text index search (BM25) with vector embeddings (pgvector) inside Supabase and benchmark our `hybrid_search_rrf` results.
- [ ] Write statistical evaluation tests using the Promptfoo CLI to measure answer relevance metrics.
