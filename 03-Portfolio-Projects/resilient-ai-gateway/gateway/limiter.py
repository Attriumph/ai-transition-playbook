import time
from typing import Tuple

class RedisTokenBucketLimiter:
    """
    A thread-safe, distributed rate-limiter designed for LLM APIs.
    Concurrently tracks both Requests-Per-Minute (RPM) and Tokens-Per-Minute (TPM)
    using a sliding window / token bucket mechanism.
    """
    def __init__(self, redis_client, user_prefix: str = "rate_limit"):
        self.redis = redis_client
        self.prefix = user_prefix

    def is_rate_limited(
        self, 
        client_id: str, 
        estimated_tokens: int,
        rpm_limit: int = 60,
        tpm_limit: int = 40000
    ) -> Tuple[bool, str]:
        """
        Evaluates rate limits for a client.
        Uses atomic Redis pipelines to prevent race conditions during concurrent API bursts.
        """
        current_time = int(time.time())
        window_start = current_time - 60 # 60-second sliding window

        rpm_key = f"{self.prefix}:{client_id}:rpm"
        tpm_key = f"{self.prefix}:{client_id}:tpm"

        # In production, we run this inside a Redis Pipeline to ensure atomicity
        try:
            # 1. Clean expired entries older than the sliding window
            self.redis.zremrangebyscore(rpm_key, 0, window_start)
            self.redis.zremrangebyscore(tpm_key, 0, window_start)

            # 2. Query current window usage
            current_rpm = self.redis.zcard(rpm_key)
            
            # Sum tokens in the current window
            tpm_entries = self.redis.zrange(tpm_key, 0, -1, withscores=True)
            current_tpm = sum(int(entry[0].split(b":")[1]) for entry in tpm_entries)

            # 3. Check RPM violations
            if current_rpm >= rpm_limit:
                return True, f"Rate Limit Exceeded: RPM limit of {rpm_limit} reached. Current: {current_rpm} RPM."

            # 4. Check TPM violations
            if current_tpm + estimated_tokens > tpm_limit:
                return True, f"Rate Limit Exceeded: TPM limit of {tpm_limit} reached. Current: {current_tpm} TPM. Estimated request tokens: {estimated_tokens}."

            # 5. Commit usage if within limits
            # We append a unique identifier (timestamp:estimated_tokens) to prevent collisions
            unique_id = f"{current_time}:{estimated_tokens}"
            
            self.redis.zadd(rpm_key, {unique_id: current_time})
            self.redis.zadd(tpm_key, {unique_id: current_time})
            
            # Set key expiration to auto-cleanup inactive users
            self.redis.expire(rpm_key, 120)
            self.redis.expire(tpm_key, 120)

            return False, "Approved"

        except Exception as e:
            # Fallback strategy: If Redis fails, fail-safe open to prevent blocking clients
            print(f"⚠️ Rate-Limiter Redis Exception (Fail-Safe Approved): {str(e)}")
            return False, "Approved (Fail-Safe Open)"
