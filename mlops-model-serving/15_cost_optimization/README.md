# Step 15: Cost Optimization

## Why It Matters

LLM inference is expensive. Optimization directly impacts margins.

## Cost Components

```
Total Cost = Compute + Storage + Network + API Calls

Compute:   GPU hours, CPU time
Storage:   Model weights, caches
Network:   Data transfer
API Calls: External LLM usage
```

## Optimization Strategies

### 1. Model Routing

Route to cheaper models when possible:

```python
class ModelRouter:
    def __init__(self):
        self.cheap_model = "gpt-3.5-turbo"
        self.expensive_model = "gpt-4"
    
    def route(self, query: str, complexity: str) -> str:
        if complexity == "simple":
            return self.cheap_model
        return self.expensive_model
    
    def estimate_complexity(self, query: str) -> str:
        # Simple heuristic or classifier
        if len(query) < 100 and "?" in query:
            return "simple"
        return "complex"
```

### 2. Caching

Don't recompute identical requests:

```python
import hashlib
from functools import lru_cache

def cache_key(messages: list) -> str:
    content = str(messages)
    return hashlib.md5(content.encode()).hexdigest()

@lru_cache(maxsize=10000)
def cached_generate(cache_key: str, messages_json: str):
    messages = json.loads(messages_json)
    return generate(messages)
```

### 3. Prompt Optimization

Shorter prompts = lower cost:

```python
def optimize_prompt(prompt: str) -> str:
    # Remove redundant whitespace
    prompt = " ".join(prompt.split())
    # Truncate if too long
    if count_tokens(prompt) > MAX_TOKENS:
        prompt = truncate_to_tokens(prompt, MAX_TOKENS)
    return prompt
```

### 4. Spot Instances

Use spot/preemptible instances for batch workloads:

```yaml
# Kubernetes spot node pool
nodeSelector:
  cloud.google.com/gke-spot: "true"
tolerations:
  - key: "cloud.google.com/gke-spot"
    operator: "Equal"
    value: "true"
    effect: "NoSchedule"
```

### 5. Auto-Scaling

Scale to zero when idle:

```python
# Scale based on queue depth
if queue_depth == 0 and idle_time > 300:
    scale_to_zero()
elif queue_depth > threshold:
    scale_up()
```

## Cost Monitoring

```python
class CostTracker:
    def __init__(self):
        self.costs = {}
    
    def record(self, model: str, input_tokens: int, output_tokens: int):
        pricing = PRICING[model]
        cost = (
            input_tokens * pricing["input"] / 1000 +
            output_tokens * pricing["output"] / 1000
        )
        self.costs[model] = self.costs.get(model, 0) + cost
    
    def get_daily_cost(self) -> float:
        return sum(self.costs.values())
```

## Files

- `cost_optimization.py` - Cost tracking and optimization
