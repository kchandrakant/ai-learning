# Step 14: Protocol Security

## Security Challenges

Each protocol has different security models:
- MCP: Local (stdio) vs remote (HTTP)
- A2A: Cross-organization trust
- ACP: Kubernetes RBAC

## MCP Security

### Local Servers (stdio)

Relatively secure — runs in user context.

**Risks:**
- Malicious server code
- Excessive permissions

**Mitigations:**
```json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["server.py"],
      "env": {
        "ALLOWED_PATHS": "/home/user/projects"
      }
    }
  }
}
```

### Remote Servers (HTTP)

**Risks:**
- Unauthorized access
- Data interception
- Server impersonation

**Mitigations:**
```python
# Always use HTTPS
server_url = "https://mcp.example.com"

# API key authentication
headers = {"Authorization": f"Bearer {api_key}"}

# Verify server certificate
client = httpx.AsyncClient(verify=True)
```

## A2A Security

### Agent Identity

```json
{
  "authentication": {
    "schemes": ["bearer", "oauth2"],
    "oauth2": {
      "authorizationUrl": "https://auth.example.com/authorize",
      "tokenUrl": "https://auth.example.com/token",
      "scopes": {
        "tasks:send": "Send tasks to this agent",
        "tasks:read": "Read task status"
      }
    }
  }
}
```

### Task Authentication

```python
@app.post("/tasks/send")
async def send_task(
    request: dict,
    authorization: str = Header(...)
):
    # Verify token
    token = authorization.replace("Bearer ", "")
    claims = verify_jwt(token)
    
    # Check scopes
    if "tasks:send" not in claims["scopes"]:
        raise HTTPException(403, "Insufficient scope")
    
    # Check rate limits per client
    if is_rate_limited(claims["client_id"]):
        raise HTTPException(429, "Rate limited")
    
    return await process_task(request)
```

### Cross-Organization Trust

```
Organization A                     Organization B
      │                                  │
      │ 1. Discovery                     │
      │──────────────────────────────────▶│
      │                                  │
      │ 2. Trust establishment           │
      │ (OAuth, mutual TLS, or manual)   │
      │◀─────────────────────────────────│
      │                                  │
      │ 3. Authenticated tasks           │
      │──────────────────────────────────▶│
```

## ACP Security

Kubernetes-native security:

```yaml
# Service Account
apiVersion: v1
kind: ServiceAccount
metadata:
  name: research-agent
  namespace: agents

---
# Role
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: agent-role
  namespace: agents
rules:
  - apiGroups: [""]
    resources: ["secrets"]
    verbs: ["get"]
    resourceNames: ["llm-api-key"]

---
# RoleBinding
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: agent-role-binding
  namespace: agents
subjects:
  - kind: ServiceAccount
    name: research-agent
roleRef:
  kind: Role
  name: agent-role
  apiGroup: rbac.authorization.k8s.io
```

## Security Patterns

### Token Validation

```python
from jose import jwt

def verify_token(token: str, audience: str) -> dict:
    try:
        # Fetch JWKS
        jwks = fetch_jwks(JWKS_URL)
        
        # Decode and verify
        claims = jwt.decode(
            token,
            jwks,
            algorithms=["RS256"],
            audience=audience,
            issuer=TRUSTED_ISSUER
        )
        
        # Check expiration
        if claims["exp"] < time.time():
            raise ValueError("Token expired")
        
        return claims
        
    except Exception as e:
        raise HTTPException(401, f"Invalid token: {e}")
```

### Input Validation

```python
from pydantic import BaseModel, validator

class TaskRequest(BaseModel):
    id: str
    message: dict
    
    @validator("id")
    def validate_id(cls, v):
        if not v.isalnum() or len(v) > 50:
            raise ValueError("Invalid task ID")
        return v
    
    @validator("message")
    def validate_message(cls, v):
        if not v.get("parts"):
            raise ValueError("Message must have parts")
        total_size = sum(len(str(p)) for p in v["parts"])
        if total_size > 100_000:
            raise ValueError("Message too large")
        return v
```

### Audit Logging

```python
import structlog

logger = structlog.get_logger()

async def send_task(request: TaskRequest, client_id: str):
    # Log request
    logger.info(
        "task_received",
        task_id=request.id,
        client_id=client_id,
        message_size=len(str(request.message))
    )
    
    try:
        result = await process_task(request)
        logger.info("task_completed", task_id=request.id)
        return result
    except Exception as e:
        logger.error("task_failed", task_id=request.id, error=str(e))
        raise
```

### Rate Limiting

```python
from slowapi import Limiter

limiter = Limiter(key_func=get_client_id)

@app.post("/tasks/send")
@limiter.limit("100/minute")
async def send_task(request: TaskRequest):
    ...
```

## Security Checklist

- [ ] Use HTTPS for all remote communication
- [ ] Implement authentication (OAuth, API keys)
- [ ] Validate all inputs
- [ ] Rate limit requests
- [ ] Audit log all actions
- [ ] Sanitize outputs (no credential leaks)
- [ ] Use least-privilege access
- [ ] Rotate credentials regularly

## Files

- `auth_middleware.py` - Authentication patterns
- `input_validation.py` - Validation examples
- `audit_logging.py` - Logging setup

## Key Takeaways

1. Each protocol needs appropriate security
2. Always authenticate remote connections
3. Validate all inputs
4. Log everything for audit
5. Use least-privilege access

## What's Next?

Step 15: **Future of Agent Protocols** — where the landscape is heading.
