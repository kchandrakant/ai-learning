# Step 6: Rate Limiting & Quotas

## Why Rate Limit?

- Prevent abuse
- Manage costs
- Ensure fair access
- Protect infrastructure

## Token Bucket Algorithm

```python
import time
from threading import Lock

class TokenBucket:
    def __init__(self, rate: float, capacity: int):
        self.rate = rate          # tokens per second
        self.capacity = capacity  # max tokens
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = Lock()
    
    def acquire(self, tokens: int = 1) -> bool:
        with self.lock:
            now = time.time()
            # Add tokens based on time passed
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
```

## Per-User Quotas

```python
from collections import defaultdict

class QuotaManager:
    def __init__(self):
        self.usage = defaultdict(lambda: {"tokens": 0, "requests": 0})
        self.limits = {
            "free": {"tokens": 10000, "requests": 100},
            "pro": {"tokens": 100000, "requests": 1000},
        }
    
    def check_quota(self, user_id: str, tier: str, tokens: int) -> bool:
        limits = self.limits[tier]
        usage = self.usage[user_id]
        
        if usage["tokens"] + tokens > limits["tokens"]:
            return False
        if usage["requests"] >= limits["requests"]:
            return False
        return True
    
    def record_usage(self, user_id: str, tokens: int):
        self.usage[user_id]["tokens"] += tokens
        self.usage[user_id]["requests"] += 1
```

## FastAPI Middleware

```python
from fastapi import HTTPException, Request

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    user_id = request.headers.get("X-User-ID", "anonymous")
    
    if not rate_limiter.acquire(user_id):
        raise HTTPException(429, "Rate limit exceeded")
    
    return await call_next(request)
```

## Files

- `rate_limiting.py` - Rate limiting implementations
