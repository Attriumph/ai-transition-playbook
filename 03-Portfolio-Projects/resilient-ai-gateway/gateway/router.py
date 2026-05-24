import os
import httpx
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from gateway.limiter import RedisTokenBucketLimiter
from gateway.circuit_breaker import StatefulCircuitBreaker, BreakerState

# Initialize APIRouter
router = APIRouter(prefix="/v1", tags=["gateway"])

# Initialize Breaker globally for upstream model provider state tracking
openai_breaker = StatefulCircuitBreaker(failure_threshold=3, recovery_timeout=30)

# Mock Redis interface for rate limiting demo purposes
class MockRedis:
    def zremrangebyscore(self, *args): pass
    def zcard(self, *args): return 0
    def zrange(self, *args, **kwargs): return []
    def zadd(self, *args, **kwargs): pass
    def expire(self, *args): pass

mock_redis = MockRedis()
limiter = RedisTokenBucketLimiter(mock_redis)

async def mock_embedding_similarity(query: str) -> float:
    """
    Mock vector semantic matching. In production, this computes cosine similarity 
    between query and Redis semantic cache.
    """
    return 0.88 # Below threshold to execute proxy calls

@router.post("/chat/completions")
async def chat_completions_proxy(request: Request):
    """
    Reverse Proxy endpoint intercepting chat completions.
    1. Checks Token Bucket Rate Limiter (RPM & TPM).
    2. Checks Semantic Cache.
    3. Leverages Circuit Breaker to route to OpenAI (Primary) or Anthropic (Backup).
    """
    body = await request.json()
    user_query = body.get("messages", [{}])[-1].get("content", "")
    
    # --- 1. Evaluate Rate Limits (Estimated Token count) ---
    estimated_tokens = len(user_query.split()) * 2 # Standard heuristic
    limited, reason = limiter.is_rate_limited(
        client_id="user_prod_api_key",
        estimated_tokens=estimated_tokens,
        rpm_limit=100,
        tpm_limit=50000
    )
    if limited:
        raise HTTPException(status_code=429, detail=reason)

    # --- 2. Evaluate Semantic Cache ---
    similarity = await mock_embedding_similarity(user_query)
    if similarity >= 0.96:
        # Expected semantic hit, bypass downstream calls
        # Returns cached payload mock
        async def cached_generator():
            yield b"data: {\"choices\": [{\"delta\": {\"content\": \"(Cached Hit Response)\"}}]}\n\n"
            yield b"data: [DONE]\n\n"
        return StreamingResponse(cached_generator(), media_type="text/event-stream")

    # --- 3. Evaluate Upstream Routing State ---
    active_route = openai_breaker.get_route()
    
    openai_url = "https://api.openai.com/v1/chat/completions"
    anthropic_url = "https://api.anthropic.com/v1/messages" # Or local vLLM endpoint
    
    primary_key = os.environ.get("OPENAI_API_KEY", "mock-key")
    backup_key = os.environ.get("ANTHROPIC_API_KEY", "mock-key")

    async def stream_completions(url: str, api_key: str, is_fallback: bool = False):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Format payload structure dynamically to match upstream expectations
        payload = {
            "model": "gpt-4o" if not is_fallback else "claude-3-5-sonnet",
            "messages": body.get("messages"),
            "stream": True
        }

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", url, headers=headers, json=payload, timeout=5.0) as resp:
                    if resp.status_code != 200:
                        raise httpx.HTTPStatusError("Upstream Outage", request=None, response=resp)
                        
                    async for chunk in resp.iter_bytes():
                        yield chunk
                        
            # If request finishes successfully, report health to the breaker
            if not is_fallback:
                openai_breaker.log_success()
                
        except (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException) as exc:
            if not is_fallback:
                # Log the failure and trip the circuit breaker
                openai_breaker.log_failure()
                print(f"⚠️ [Proxy Route] Primary failed: {str(exc)}. Triggering instant failover reroute...")
                
                # Dynamic Failover: Recurse stream completion targeting the backup route in under 50ms
                async for fallback_chunk in stream_completions(anthropic_url, backup_key, is_fallback=True):
                    yield fallback_chunk
            else:
                yield b"data: {\"error\": \"All upstream providers are currently unavailable.\"}\n\n"

    # Route request dynamically based on breaker health
    if active_route == BreakerState.OPEN:
        print("🛡️ [Circuit Breaker] Primary is OPEN. Bypassing directly to backup route.")
        return StreamingResponse(
            stream_completions(anthropic_url, backup_key, is_fallback=True),
            media_type="text/event-stream"
        )
    else:
        print("🟢 [Circuit Breaker] Primary is CLOSED/HALF-OPEN. Routing to primary endpoint.")
        return StreamingResponse(
            stream_completions(openai_url, primary_key, is_fallback=False),
            media_type="text/event-stream"
        )
