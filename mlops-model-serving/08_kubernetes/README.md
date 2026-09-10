# Step 8: Kubernetes Deployment

## Why Kubernetes?

- Container orchestration
- Auto-scaling
- Self-healing
- Resource management

## Basic Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: llm-api
  template:
    metadata:
      labels:
        app: llm-api
    spec:
      containers:
      - name: llm-api
        image: llm-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
```

## GPU Scheduling

```yaml
resources:
  limits:
    nvidia.com/gpu: 1  # Request 1 GPU

nodeSelector:
  gpu-type: a100       # Specific GPU type

tolerations:
- key: "nvidia.com/gpu"
  operator: "Exists"
  effect: "NoSchedule"
```

## Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-api
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: llm-api
spec:
  selector:
    app: llm-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Files

- `kubernetes/` - Complete K8s manifests
