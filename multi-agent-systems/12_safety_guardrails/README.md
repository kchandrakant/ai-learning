# Step 12: Safety & Guardrails

## Why Safety Matters

Agents can take real actions with real consequences:
- Delete files
- Send emails
- Make API calls
- Execute code
- Spend money

**Without guardrails, agents are dangerous.**

## Defense Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Safety Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Layer 1: Input Validation    ─── Sanitize inputs          │
│   Layer 2: Action Allowlisting ─── Restrict capabilities    │
│   Layer 3: Confirmation Gates  ─── Human approval           │
│   Layer 4: Rate Limiting       ─── Prevent runaway          │
│   Layer 5: Sandboxing          ─── Isolate execution        │
│   Layer 6: Monitoring          ─── Detect anomalies         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Layer 1: Input Validation

```python
import re
from typing import Any

class InputValidator:
    def __init__(self):
        self.patterns = {
            "sql_injection": r"('|--|;|DROP|DELETE|INSERT|UPDATE)",
            "path_traversal": r"\.\./|\.\.\\",
            "command_injection": r"[;&|`$]",
        }
    
    def validate(self, input_data: Any) -> tuple[bool, str]:
        """Validate input for common attack patterns."""
        
        if isinstance(input_data, str):
            for name, pattern in self.patterns.items():
                if re.search(pattern, input_data, re.IGNORECASE):
                    return False, f"Blocked: {name} pattern detected"
        
        return True, "Valid"
    
    def sanitize(self, input_data: str) -> str:
        """Remove potentially dangerous characters."""
        # Remove null bytes
        sanitized = input_data.replace('\x00', '')
        # Limit length
        sanitized = sanitized[:10000]
        return sanitized
```

## Layer 2: Action Allowlisting

```python
class ActionGuard:
    def __init__(self):
        self.allowed_actions = set()
        self.blocked_actions = set()
        self.action_rules: dict[str, Callable] = {}
    
    def allow(self, action: str):
        self.allowed_actions.add(action)
    
    def block(self, action: str):
        self.blocked_actions.add(action)
    
    def add_rule(self, action: str, rule: Callable[[dict], bool]):
        """Add conditional rule for an action."""
        self.action_rules[action] = rule
    
    def check(self, action: str, args: dict) -> tuple[bool, str]:
        # Explicit blocks
        if action in self.blocked_actions:
            return False, f"Action '{action}' is blocked"
        
        # Allowlist mode
        if self.allowed_actions and action not in self.allowed_actions:
            return False, f"Action '{action}' is not in allowlist"
        
        # Custom rules
        if action in self.action_rules:
            if not self.action_rules[action](args):
                return False, f"Action '{action}' failed rule check"
        
        return True, "Allowed"

# Usage
guard = ActionGuard()
guard.allow("read_file")
guard.allow("write_file")
guard.block("delete_file")
guard.block("execute_shell")

# Conditional rules
guard.add_rule("write_file", lambda args: not args["path"].startswith("/etc"))
```

## Layer 3: Confirmation Gates

```python
class ConfirmationGate:
    def __init__(self, risk_assessor):
        self.risk_assessor = risk_assessor
        self.auto_approve_low_risk = True
    
    def assess_and_confirm(self, action: str, args: dict) -> bool:
        risk_level = self.risk_assessor.assess(action, args)
        
        if risk_level == "low" and self.auto_approve_low_risk:
            return True
        
        if risk_level == "high":
            # Always require human confirmation for high-risk
            return self.request_human_approval(action, args, risk_level)
        
        if risk_level == "medium":
            # Show warning, auto-approve with logging
            self.log_warning(action, args)
            return True
        
        return True
    
    def request_human_approval(self, action: str, args: dict, risk: str) -> bool:
        print(f"\n⚠️  HIGH RISK ACTION REQUIRES APPROVAL")
        print(f"Action: {action}")
        print(f"Arguments: {args}")
        print(f"Risk level: {risk}")
        
        response = input("Approve? (yes/no): ")
        return response.lower() == "yes"

class RiskAssessor:
    def __init__(self):
        self.high_risk_actions = {"delete", "execute", "send_email", "api_post"}
        self.high_risk_paths = {"/etc", "/system", "~/.ssh"}
    
    def assess(self, action: str, args: dict) -> str:
        # High risk actions
        if action in self.high_risk_actions:
            return "high"
        
        # High risk paths
        path = args.get("path", "")
        if any(path.startswith(p) for p in self.high_risk_paths):
            return "high"
        
        # Check for sudo/admin
        if "sudo" in str(args).lower():
            return "high"
        
        return "low"
```

## Layer 4: Rate Limiting

```python
from collections import defaultdict
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self):
        self.limits = {
            "llm_calls": {"count": 100, "window": timedelta(minutes=1)},
            "tool_calls": {"count": 50, "window": timedelta(minutes=1)},
            "api_requests": {"count": 10, "window": timedelta(minutes=1)},
        }
        self.counters = defaultdict(list)
    
    def check(self, action_type: str) -> bool:
        if action_type not in self.limits:
            return True
        
        limit = self.limits[action_type]
        now = datetime.now()
        window_start = now - limit["window"]
        
        # Clean old entries
        self.counters[action_type] = [
            t for t in self.counters[action_type] if t > window_start
        ]
        
        # Check limit
        if len(self.counters[action_type]) >= limit["count"]:
            return False
        
        # Record this call
        self.counters[action_type].append(now)
        return True
    
    def get_status(self) -> dict:
        return {
            action: {
                "used": len(times),
                "limit": self.limits[action]["count"]
            }
            for action, times in self.counters.items()
        }
```

## Layer 5: Sandboxing

```python
import subprocess
import tempfile
import os

class Sandbox:
    def __init__(self, 
                 allowed_paths: list[str] = None,
                 max_execution_time: int = 30,
                 max_memory_mb: int = 512):
        self.allowed_paths = allowed_paths or []
        self.max_execution_time = max_execution_time
        self.max_memory_mb = max_memory_mb
    
    def execute_code(self, code: str, language: str = "python") -> dict:
        """Execute code in isolated environment."""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write code to temp file
            code_file = os.path.join(tmpdir, f"code.{language}")
            with open(code_file, 'w') as f:
                f.write(code)
            
            # Run in subprocess with limits
            try:
                result = subprocess.run(
                    ["python", code_file],
                    capture_output=True,
                    timeout=self.max_execution_time,
                    cwd=tmpdir,
                    env={"PATH": "/usr/bin"}  # Minimal PATH
                )
                
                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout.decode()[:10000],
                    "stderr": result.stderr.decode()[:10000],
                    "returncode": result.returncode
                }
                
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": f"Execution timed out after {self.max_execution_time}s"
                }
    
    def check_file_access(self, path: str, operation: str) -> bool:
        """Check if file access is allowed."""
        abs_path = os.path.abspath(path)
        
        for allowed in self.allowed_paths:
            if abs_path.startswith(os.path.abspath(allowed)):
                return True
        
        return False
```

## Layer 6: Anomaly Detection

```python
class AnomalyDetector:
    def __init__(self):
        self.baselines = {}
        self.thresholds = {
            "actions_per_minute": 30,
            "errors_per_minute": 5,
            "cost_per_session": 1.00,  # dollars
            "consecutive_failures": 3
        }
        self.current_session = {
            "action_count": 0,
            "error_count": 0,
            "total_cost": 0,
            "consecutive_failures": 0
        }
    
    def record_action(self, success: bool, cost: float = 0):
        self.current_session["action_count"] += 1
        self.current_session["total_cost"] += cost
        
        if not success:
            self.current_session["error_count"] += 1
            self.current_session["consecutive_failures"] += 1
        else:
            self.current_session["consecutive_failures"] = 0
    
    def check_anomalies(self) -> list[str]:
        anomalies = []
        
        if self.current_session["consecutive_failures"] >= self.thresholds["consecutive_failures"]:
            anomalies.append("Too many consecutive failures")
        
        if self.current_session["total_cost"] >= self.thresholds["cost_per_session"]:
            anomalies.append("Cost threshold exceeded")
        
        return anomalies
    
    def should_halt(self) -> bool:
        return len(self.check_anomalies()) > 0
```

## Complete Safety System

```python
class SafetySystem:
    def __init__(self):
        self.input_validator = InputValidator()
        self.action_guard = ActionGuard()
        self.confirmation_gate = ConfirmationGate(RiskAssessor())
        self.rate_limiter = RateLimiter()
        self.sandbox = Sandbox(allowed_paths=["./workspace"])
        self.anomaly_detector = AnomalyDetector()
    
    def pre_action_check(self, action: str, args: dict) -> tuple[bool, str]:
        """Run all safety checks before an action."""
        
        # Check for anomalies
        if self.anomaly_detector.should_halt():
            return False, "Anomaly detected - halting"
        
        # Validate inputs
        for key, value in args.items():
            valid, msg = self.input_validator.validate(value)
            if not valid:
                return False, msg
        
        # Check action allowlist
        allowed, msg = self.action_guard.check(action, args)
        if not allowed:
            return False, msg
        
        # Rate limiting
        if not self.rate_limiter.check("tool_calls"):
            return False, "Rate limit exceeded"
        
        # Confirmation for risky actions
        if not self.confirmation_gate.assess_and_confirm(action, args):
            return False, "User declined action"
        
        return True, "All checks passed"
```

## Files

- `safety_guardrails.py` - Complete safety system

## Key Takeaways

1. Defense in depth: multiple safety layers
2. Allowlist preferred over blocklist
3. Human approval for high-risk actions
4. Rate limit to prevent runaway
5. Sandbox code execution
6. Monitor for anomalies

## Demo Projects

With fundamentals complete, try building:
1. **Research Assistant** — Multi-agent research workflow
2. **Coding Agent** — Code generation with safety
3. **Customer Support** — Agent with handoff to human
