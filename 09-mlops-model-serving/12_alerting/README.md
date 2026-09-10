# Step 12: Alerting & Incident Response

## SLOs and SLAs

Define what "good" looks like:

```yaml
SLOs:
  availability: 99.9%
  latency_p99: 5s
  error_rate: < 1%
```

## Alert Rules (Prometheus)

```yaml
groups:
- name: llm-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(llm_requests_total{status="error"}[5m]) > 0.01
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"

  - alert: HighLatency
    expr: histogram_quantile(0.99, llm_request_latency_seconds) > 5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "P99 latency above 5s"

  - alert: LowGPUUtilization
    expr: gpu_utilization_percent < 20
    for: 15m
    labels:
      severity: info
    annotations:
      summary: "GPU underutilized - consider scaling down"
```

## Runbook Template

```markdown
# Alert: HighErrorRate

## Symptoms
- Error rate > 1%
- Users experiencing failures

## Diagnosis
1. Check error logs: `kubectl logs -l app=llm-api`
2. Check GPU status: `nvidia-smi`
3. Check memory: `kubectl top pods`

## Resolution
1. If OOM: Restart pods, reduce batch size
2. If model error: Roll back to previous version
3. If rate limit: Scale up replicas

## Escalation
- Page on-call if not resolved in 15 minutes
```

## Files

- `alerting.py` - Alert configuration
- `runbooks/` - Incident runbooks
