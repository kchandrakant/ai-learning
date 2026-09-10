# Step 15: Monitoring & Audit

## Why Monitoring Matters

You can't secure what you can't see.

```
Agent actions without audit = Black box
Agent actions with audit = Accountability
```

## Security Logging

### Structured Logging

```python
import structlog

logger = structlog.get_logger()

class AuditLogger:
    def log_action(self, agent_id: str, action: str, resource: str, 
                   result: str, context: dict = None):
        logger.info(
            "agent_action",
            agent_id=agent_id,
            action=action,
            resource=resource,
            result=result,
            timestamp=datetime.utcnow().isoformat(),
            context=context or {}
        )
    
    def log_auth_event(self, event_type: str, agent_id: str, 
                       success: bool, details: dict = None):
        log_func = logger.info if success else logger.warning
        log_func(
            "auth_event",
            event_type=event_type,
            agent_id=agent_id,
            success=success,
            timestamp=datetime.utcnow().isoformat(),
            details=details or {}
        )
```

### What to Log

```python
AUDIT_EVENTS = {
    # Authentication
    "auth.login": "Agent authenticated",
    "auth.logout": "Agent session ended",
    "auth.failed": "Authentication failed",
    "auth.token_refresh": "Token refreshed",
    
    # Authorization
    "authz.granted": "Permission granted",
    "authz.denied": "Permission denied",
    "authz.escalation": "Privilege escalation attempted",
    
    # Actions
    "action.tool_call": "Agent called tool",
    "action.api_call": "Agent made API call",
    "action.data_access": "Agent accessed data",
    "action.data_modify": "Agent modified data",
    
    # Security
    "security.anomaly": "Anomalous behavior detected",
    "security.injection": "Potential injection attempt",
    "security.rate_limit": "Rate limit exceeded"
}
```

## Audit Trail Requirements

### Immutable Logs

```python
import hashlib

class ImmutableAuditLog:
    def __init__(self):
        self.previous_hash = "0" * 64
    
    def append(self, event: dict):
        # Add chain hash
        event["previous_hash"] = self.previous_hash
        event_json = json.dumps(event, sort_keys=True)
        event["hash"] = hashlib.sha256(event_json.encode()).hexdigest()
        
        # Store immutably
        self.storage.append(event)
        self.previous_hash = event["hash"]
    
    def verify_chain(self) -> bool:
        """Verify audit log integrity."""
        prev_hash = "0" * 64
        for event in self.storage.read_all():
            if event["previous_hash"] != prev_hash:
                return False
            
            stored_hash = event.pop("hash")
            computed_hash = hashlib.sha256(
                json.dumps(event, sort_keys=True).encode()
            ).hexdigest()
            
            if stored_hash != computed_hash:
                return False
            
            prev_hash = stored_hash
        
        return True
```

### Retention Policies

```yaml
# Log retention configuration
retention:
  security_events: 7 years  # Compliance requirement
  auth_events: 2 years
  action_events: 1 year
  debug_logs: 30 days
```

## Anomaly Detection

```python
from collections import defaultdict
from datetime import timedelta

class AnomalyDetector:
    def __init__(self):
        self.baselines = defaultdict(lambda: {"count": 0, "avg_rate": 0})
    
    def check_action(self, agent_id: str, action: str) -> bool:
        key = f"{agent_id}:{action}"
        baseline = self.baselines[key]
        
        # Check rate anomaly
        current_rate = self.get_current_rate(agent_id, action)
        if current_rate > baseline["avg_rate"] * 3:
            self.alert(
                "rate_anomaly",
                agent_id=agent_id,
                action=action,
                current_rate=current_rate,
                baseline_rate=baseline["avg_rate"]
            )
            return False
        
        # Check time anomaly (actions outside normal hours)
        if not self.is_normal_hours(agent_id):
            self.alert(
                "time_anomaly",
                agent_id=agent_id,
                action=action
            )
        
        return True
    
    def alert(self, alert_type: str, **context):
        logger.warning(
            "security_anomaly",
            alert_type=alert_type,
            **context
        )
        # Send to SIEM, PagerDuty, etc.
```

## Metrics & Dashboards

### Key Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Authentication metrics
auth_attempts = Counter(
    'agent_auth_attempts_total',
    'Total authentication attempts',
    ['agent_id', 'result']
)

# Action metrics
action_duration = Histogram(
    'agent_action_duration_seconds',
    'Time spent on agent actions',
    ['agent_id', 'action']
)

# Security metrics
security_events = Counter(
    'agent_security_events_total',
    'Security events by type',
    ['event_type', 'severity']
)

# Active sessions
active_sessions = Gauge(
    'agent_active_sessions',
    'Number of active agent sessions'
)
```

### Dashboard Queries

```promql
# Failed auth rate
rate(agent_auth_attempts_total{result="failed"}[5m])

# High-risk actions per agent
sum by (agent_id) (agent_security_events_total{severity="high"})

# P95 action latency
histogram_quantile(0.95, rate(agent_action_duration_seconds_bucket[5m]))
```

## Incident Response

```python
class IncidentResponder:
    async def handle_security_event(self, event: dict):
        severity = self.assess_severity(event)
        
        if severity == "critical":
            # Immediate action
            await self.revoke_agent_tokens(event["agent_id"])
            await self.notify_security_team(event)
            await self.isolate_agent(event["agent_id"])
        
        elif severity == "high":
            # Investigation needed
            await self.notify_security_team(event)
            await self.increase_monitoring(event["agent_id"])
        
        else:
            # Log for review
            await self.queue_for_review(event)
```

## Compliance

| Requirement | Implementation |
|-------------|----------------|
| SOC 2 | Immutable audit logs, access controls |
| GDPR | Data access logging, retention limits |
| HIPAA | PHI access audit, encryption |
| PCI DSS | Cardholder data logging |

## Files

- `audit_logger.py` - Structured audit logging
- `anomaly_detector.py` - Behavioral analysis
- `metrics.py` - Prometheus metrics
- `incident_response.py` - Automated response

## Key Takeaways

1. Log all security-relevant events
2. Make audit logs immutable
3. Detect anomalies automatically
4. Metric dashboards for visibility
5. Automate incident response

## What's Next?

Step 16: **Threat Modeling** — understanding AI-specific attacks.
