# Step 7: Access Control Patterns

## Access Control Models

### RBAC (Role-Based Access Control)

Assign permissions to roles, assign roles to users/agents.

```python
# Define roles
ROLES = {
    "reader": ["read:documents"],
    "editor": ["read:documents", "write:documents"],
    "admin": ["read:documents", "write:documents", "delete:documents", "manage:users"]
}

# Check permission
def check_permission(agent_role: str, required_permission: str) -> bool:
    return required_permission in ROLES.get(agent_role, [])

# Usage
if check_permission(agent.role, "write:documents"):
    # Allow action
```

**Pros:** Simple, easy to audit
**Cons:** Coarse-grained, role explosion

### ABAC (Attribute-Based Access Control)

Decisions based on attributes of subject, resource, action, environment.

```python
def check_access(subject: dict, resource: dict, action: str, environment: dict) -> bool:
    # Policy: Sales can read customer data during business hours
    if action == "read" and resource["type"] == "customer_data":
        if subject["department"] == "sales":
            if 9 <= environment["hour"] <= 17:
                return True
    return False

# Usage
allowed = check_access(
    subject={"id": "agent-1", "department": "sales"},
    resource={"type": "customer_data", "id": "cust-123"},
    action="read",
    environment={"hour": 14, "day": "Monday"}
)
```

**Pros:** Flexible, context-aware
**Cons:** Complex to manage, hard to audit

### Capability-Based Security

Tokens that grant specific capabilities.

```python
# Capability token
capability = {
    "resource": "file://documents/report.pdf",
    "actions": ["read"],
    "expires": "2024-12-31T23:59:59Z",
    "holder": "agent-123"
}

# Verify capability
def verify_capability(token: dict, resource: str, action: str) -> bool:
    if token["resource"] != resource:
        return False
    if action not in token["actions"]:
        return False
    if datetime.fromisoformat(token["expires"]) < datetime.now():
        return False
    return True
```

**Pros:** Fine-grained, minimal authority
**Cons:** Token management complexity

## Policy Engines

### Open Policy Agent (OPA)

```rego
# policy.rego
package authz

default allow = false

allow {
    input.action == "read"
    input.subject.role == "analyst"
    input.resource.classification != "top_secret"
}

allow {
    input.subject.role == "admin"
}
```

```python
from opa_client import OPAClient

opa = OPAClient("http://localhost:8181")

decision = opa.check_policy(
    "authz/allow",
    {
        "subject": {"role": "analyst"},
        "action": "read",
        "resource": {"classification": "confidential"}
    }
)
# decision.result == True
```

### AWS Cedar

```cedar
// Policy
permit(
    principal == Agent::"research-agent",
    action == Action::"read",
    resource in Folder::"research-docs"
);
```

## Least Privilege for Agents

```python
class LeastPrivilegeAgent:
    """Agent that requests minimal permissions."""
    
    async def execute_task(self, task: str):
        # Analyze task to determine needed permissions
        required_perms = self.analyze_permissions(task)
        
        # Request only what's needed
        token = await self.request_scoped_token(required_perms)
        
        # Execute with limited token
        with self.use_token(token):
            result = await self.perform_actions(task)
        
        # Token automatically expires after task
        return result
    
    def analyze_permissions(self, task: str) -> list:
        """Determine minimum permissions for task."""
        perms = []
        if "read" in task.lower():
            perms.append("read:data")
        if "send email" in task.lower():
            perms.append("send:email")
        return perms
```

## Comparison

| Model | Granularity | Complexity | Best For |
|-------|-------------|------------|----------|
| RBAC | Coarse | Low | Simple apps |
| ABAC | Fine | High | Complex policies |
| Capability | Very fine | Medium | Distributed systems |

## Files

- `rbac.py` - Role-based access control
- `abac.py` - Attribute-based policies
- `capability.py` - Capability tokens
- `opa_integration.py` - OPA policy engine

## Key Takeaways

1. RBAC is simple but coarse
2. ABAC is flexible but complex
3. Capabilities provide fine-grained control
4. Policy engines externalize decisions
5. Agents should use least privilege

## What's Next?

Step 8: **Per-Task Authorization** — dynamic, just-in-time permissions.
