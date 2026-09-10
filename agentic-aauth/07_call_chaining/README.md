# Module 07: Call Chaining & Delegation

**Time**: ~45 minutes  
**Goal**: Handle multi-hop scenarios where resources act as agents

---

## The Big Idea

Sometimes a resource needs to call **another resource** to fulfill a request:

```
User: "Book my trip to Tokyo"
  ↓
Agent → Booking Service → Payment Processor → Bank
```

The Booking Service must act as an **agent** to call Payment Processor. AAuth handles this with **call chaining**.

---

## The Problem

```
Agent has auth_token for Booking Service
Booking Service needs to call Payment Processor
But Payment Processor doesn't know about the original Agent!
```

Without call chaining:
- Booking Service would need its own separate authorization
- No connection to the original user's consent
- No audit trail back to the original request

---

## Call Chaining Flow

```
Agent        Booking        Payment         PS
  │            │               │             │
  │  auth_token for Booking    │             │
  │───────────▶│               │             │
  │            │               │             │
  │            │ [needs to call Payment]     │
  │            │               │             │
  │            │  resource_token             │
  │            │  (as agent)   │             │
  │            │◀──────────────│             │
  │            │               │             │
  │            │  POST /token                │
  │            │  resource_token             │
  │            │  upstream_token (Booking's auth_token)
  │            │  agent_token (Booking's)    │
  │            │─────────────────────────────▶
  │            │               │             │
  │            │               │  [PS evaluates
  │            │               │   mission context]
  │            │               │             │
  │            │  auth_token for Payment     │
  │            │◀─────────────────────────────
  │            │               │             │
  │            │  HTTPSig w/ auth_token      │
  │            │──────────────▶│             │
  │            │               │             │
  │            │  200 OK       │             │
  │            │◀──────────────│             │
  │            │               │             │
  │  200 OK    │               │             │
  │◀───────────│               │             │
```

---

## Key Concepts

### 1. Resources as Agents

A resource acting as an agent must:
- Have its own agent identity (publish `/.well-known/aauth-agent.json`)
- Sign downstream requests
- Include the upstream token in token requests

### 2. Upstream Token

The `upstream_token` parameter proves the chain of authorization:

```json
{
  "resource_token": "eyJ...",      // From Payment Processor
  "agent_token": "eyJ...",          // Booking Service's agent token
  "upstream_token": "eyJ..."        // Auth token Booking received from Agent
}
```

### 3. The `act` Claim

Auth tokens include delegation chain in the `act` claim:

```json
{
  "agent": "aauth:booking@booking.example",
  "act": {
    "agent": "aauth:assistant@agent.example"
  }
}
```

Nested chains:
```json
{
  "agent": "aauth:payment@payment.example",
  "act": {
    "agent": "aauth:booking@booking.example",
    "act": {
      "agent": "aauth:assistant@agent.example"
    }
  }
}
```

---

## Routing the Token Request

The intermediary determines where to send based on the upstream auth token:

| Upstream token has... | Route to... |
|-----------------------|-------------|
| `mission.approver` | That PS (governed path) |
| `iss` = PS URL | That PS |
| `iss` = AS URL | That AS (no governance) |

**Important**: Mission ensures the PS is in the loop for every hop.

---

## Downstream Scope

Downstream authorization is **not** constrained to be a subset of upstream scope!

Example:
- Agent has `flights.search` scope for Booking
- Booking needs `payment.charge` for Payment Processor
- These are orthogonal — user couldn't directly charge a card

The PS evaluates each hop against mission context, not algebraic scope rules.

---

## Interaction Chaining

When downstream requires user interaction:

```
Agent        Booking        Payment         PS           User
  │            │               │             │             │
  │            │  token request               │             │
  │            │─────────────────────────────▶│             │
  │            │               │             │             │
  │            │  202 (interaction needed)   │             │
  │            │◀─────────────────────────────│             │
  │            │               │             │             │
  │  202 (interaction)         │             │             │
  │◀───────────│               │             │             │
  │            │               │             │             │
  │  [agent directs user]──────│─────────────│────────────▶│
  │            │               │             │             │
  │            │               │             │◀────────────│
  │            │               │             │  [consent]  │
  │            │               │             │             │
```

The interaction requirement propagates back to the original agent.

---

## Sub-Agents

For short-lived workers under a parent agent:

```json
{
  "sub": "aauth:planner.7f3c+search1@vendor.example",
  "parent_agent": "aauth:planner.7f3c@vendor.example"
}
```

Rules:
- Sub-agent identifier: `parent_local+discriminator@domain`
- Max one level deep (sub-agents can't have sub-agents)
- Parent mediates all authorization

---

## What We'll Build

```
07_call_chaining/
├── README.md
├── resource_as_agent.py   # Resource with agent identity
├── call_chain.py          # Multi-hop authorization
├── act_claim.py           # Building delegation chains
└── demo.py                # 3-hop chain demo
```

By the end:
1. Resource that can act as an agent
2. Token request with upstream_token
3. Proper `act` claim construction
4. Multi-hop authorization chain

---

## Let's Start Building

We'll create a multi-hop scenario with proper delegation tracking.

**Say "ready" to begin.**

