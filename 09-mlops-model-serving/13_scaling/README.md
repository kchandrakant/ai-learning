# Step 13: Scaling Strategies

## Scaling Dimensions

```
Vertical:   Bigger machines (more GPU memory)
Horizontal: More machines (more replicas)
```

## Horizontal Scaling

```python
# Auto-scale based on queue depth
class AutoScaler:
    def __init__(self, min_replicas=1, max_replicas=10):
        self.min = min_replicas
        self.max = max_replicas
    
    def calculate_replicas(self, queue_depth: int, current: int) -> int:
        target_per_replica = 10  # requests per replica
        desired = max(1, queue_depth // target_per_replica)
        desired = max(self.min, min(self.max, desired))
        return desired
```

## Load Balancing

```yaml
# Kubernetes Service with session affinity
apiVersion: v1
kind: Service
metadata:
  name: llm-api
spec:
  sessionAffinity: ClientIP  # Sticky sessions for caching
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 3600
```

## Multi-Region

```
Region A: Primary
Region B: Failover
Region C: Read replicas

Route based on:
- Latency (nearest region)
- Capacity (least loaded)
- Compliance (data residency)
```

## Scaling Considerations

| Factor | Impact |
|--------|--------|
| Cold start | Slow scale-up for GPU pods |
| Model loading | 30-60s to load model |
| KV cache | Memory grows with batch |
| Cost | GPUs are expensive |

## Files

- `scaling.py` - Scaling strategies
