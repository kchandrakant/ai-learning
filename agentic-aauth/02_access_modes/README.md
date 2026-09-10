# Module 02: The Four Access Modes

**Time**: ~45 minutes  
**Goal**: Understand how agents access resources with progressive levels of authorization

---

## The Big Idea

AAuth supports four access modes, each adding parties and capabilities. You can adopt incrementally — no coordination required between parties.

```
Mode                    Parties              What You Get
────────────────────────────────────────────────────────────────
Identity-based          Agent + Resource     Replaces API keys
Resource-managed        Agent + Resource     Resource handles auth
PS-asserted (3-party)   + Person Server      User identity & consent
Federated (4-party)     + Access Server      Cross-domain policy
```

---

## The Four Modes

### Mode 1: Identity-Based Access

The simplest mode. The resource verifies the agent's signature and decides based on **who the agent is**.

```
Agent                                    Resource
  │                                         │
  │  HTTPSig w/ agent_token                 │
  │────────────────────────────────────────▶│
  │                                         │ Verify signature
  │                                         │ Check: do I trust this agent?
  │  200 OK                                 │
  │◀────────────────────────────────────────│
```

**Use case**: Replaces API keys. The resource maintains a list of trusted agents.

### Mode 2: Resource-Managed Access (Two-Party)

The resource handles authorization itself — via interaction, OAuth, or internal policy.

```
Agent                                    Resource
  │                                         │
  │  HTTPSig w/ agent_token                 │
  │────────────────────────────────────────▶│
  │                                         │
  │  202 Accepted (interaction required)    │
  │  url="https://resource/consent"         │
  │◀────────────────────────────────────────│
  │                                         │
  │  [user completes interaction]           │
  │                                         │
  │  GET pending URL                        │
  │────────────────────────────────────────▶│
  │                                         │
  │  200 OK + AAuth-Access: opaque-token    │
  │◀────────────────────────────────────────│
```

**Use case**: Resource wraps existing OAuth/consent flows.

### Mode 3: PS-Asserted Access (Three-Party)

The resource issues a **resource token**. The agent's Person Server asserts user identity and consent.

```
Agent                         Resource          PS
  │                              │               │
  │  HTTPSig w/ agent_token      │               │
  │─────────────────────────────▶│               │
  │                              │               │
  │  resource_token (aud=PS)     │               │
  │◀─────────────────────────────│               │
  │                              │               │
  │  POST token_endpoint                         │
  │  w/ resource_token           │               │
  │─────────────────────────────────────────────▶│
  │                              │               │ [consent]
  │  auth_token                  │               │
  │◀─────────────────────────────────────────────│
  │                              │               │
  │  HTTPSig w/ auth_token       │               │
  │─────────────────────────────▶│               │
  │                              │               │
  │  200 OK                      │               │
  │◀─────────────────────────────│               │
```

**Use case**: Any agent's PS can assert identity to any resource (no pre-registration).

### Mode 4: Federated Access (Four-Party)

The resource has its own Access Server. The PS federates with the AS.

```
Agent                    Resource       PS                 AS
  │                         │            │                  │
  │  resource_token         │            │                  │
  │  (aud=AS URL)           │            │                  │
  │◀────────────────────────│            │                  │
  │                         │            │                  │
  │  POST token_endpoint    │            │                  │
  │  w/ resource_token      │            │                  │
  │─────────────────────────────────────▶│                  │
  │                         │            │                  │
  │                         │            │  POST /token     │
  │                         │            │─────────────────▶│
  │                         │            │                  │
  │                         │            │  auth_token      │
  │                         │            │◀─────────────────│
  │                         │            │                  │
  │  auth_token             │            │                  │
  │◀─────────────────────────────────────│                  │
  │                         │            │                  │
  │  HTTPSig w/ auth_token  │            │                  │
  │────────────────────────▶│            │                  │
```

**Use case**: Enterprise resources with their own policy engines.

---

## What We'll Build

```
02_access_modes/
├── README.md
├── mock_resource.py       # Resource server with all 4 modes
├── identity_access.py     # Mode 1: Identity-based
├── resource_managed.py    # Mode 2: Resource handles auth
└── demo.py                # Compare all modes
```

By the end:
1. Implement identity-based access (Mode 1)
2. See how the resource can require interaction (Mode 2)
3. Understand when you need a PS (Mode 3) or AS (Mode 4)

---

## Key Decision: Which Mode?

| I need... | Use Mode |
|-----------|----------|
| Just verify agent identity | 1: Identity-based |
| My own consent/OAuth flow | 2: Resource-managed |
| User identity from any PS | 3: PS-asserted |
| Cross-domain policy enforcement | 4: Federated |

---

## Let's Start Building

We'll focus on **Mode 1** first — it's the foundation. Then we'll progressively add complexity.

**Say "ready" to begin.**

