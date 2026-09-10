# Step 16: CI/CD for ML

## Why CI/CD?

- Reliable deployments
- Quick rollbacks
- Automated testing
- Audit trail

## Model Versioning

```
models/
├── llama-2-7b/
│   ├── v1.0.0/
│   ├── v1.0.1/
│   └── latest -> v1.0.1
```

## GitHub Actions Pipeline

```yaml
name: Deploy LLM API

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: pytest tests/

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build and push
        run: |
          docker build -t llm-api:${{ github.sha }} .
          docker push llm-api:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy canary
        run: |
          kubectl set image deployment/llm-api \
            llm-api=llm-api:${{ github.sha }} \
            --record
      
      - name: Health check
        run: ./scripts/health_check.sh
      
      - name: Promote or rollback
        run: |
          if ./scripts/verify_canary.sh; then
            kubectl rollout resume deployment/llm-api
          else
            kubectl rollout undo deployment/llm-api
          fi
```

## Canary Deployment

```yaml
# Deploy to 10% of traffic first
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
spec:
  http:
  - route:
    - destination:
        host: llm-api
        subset: stable
      weight: 90
    - destination:
        host: llm-api
        subset: canary
      weight: 10
```

## Rollback

```bash
# Quick rollback
kubectl rollout undo deployment/llm-api

# Rollback to specific version
kubectl rollout undo deployment/llm-api --to-revision=2
```

## Files

- `.github/workflows/` - CI/CD pipelines
- `scripts/` - Deployment scripts
