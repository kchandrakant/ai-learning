# Module 5: Human-in-the-Loop Design

Integrating human oversight effectively in agent systems.

## Overview

The goal isn't full autonomy—it's appropriate autonomy. Well-designed human-in-the-loop patterns let agents handle routine work while escalating high-stakes decisions.

## Key Topics

### When to Pause for Human Input
- High-cost actions (financial transactions, data deletion)
- Irreversible operations (sending emails, publishing content)
- Ambiguous situations (conflicting instructions, unclear intent)
- Policy violations (detected or near-boundary)

### Approval Patterns

**Pre-approval (synchronous):**
```
Agent: "I need to delete 500 files. Approve?"
User: "Yes"
Agent: proceeds
```

**Post-review (async):**
```
Agent: executes action
System: queues for review
User: reviews and confirms/reverts
```

**Guardrail (automatic):**
```
Agent: attempts action
System: checks against rules → blocks if risky
Agent: notified, adjusts approach
```

### Async Approval Patterns
- 202 Accepted responses (action pending approval)
- Webhook callbacks on approval
- Timeout with default action
- Escalation chains (user → manager → admin)

### UI/UX for Human-Agent Collaboration
- Clear action explanations
- Diff views for changes
- One-click approve/reject
- Batch approval for similar actions
- Audit trail visibility

### Escalation Policies
```yaml
escalation_policy:
  - condition: cost > $100
    action: require_approval
    approver: user
  - condition: affects_production
    action: require_approval  
    approver: admin
  - condition: unknown_tool
    action: block
```

## Exercises

1. Implement a pre-approval gate for file deletions
2. Build an async approval workflow with webhooks
3. Design an escalation policy system
4. Create a review UI for agent actions

## Key Insight

The best human-in-the-loop systems are invisible for routine work but surface exactly when needed. Design for the 95% case (autonomous) while handling the 5% case (human needed) gracefully.
