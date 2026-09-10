# Demo: AAuth in Action

## Overview

This directory contains end-to-end demonstrations of the AAuth protocol, bringing together all the components we build throughout the learning path.

---

## Demos

### 1. Agent Identity Demo
See cryptographic agent identity in action:
- Generate Ed25519 key pair
- Create agent token
- Sign HTTP requests
- Verify signatures

### 2. Four Access Modes Demo
Compare all four resource access modes side-by-side:
- Identity-based (API key replacement)
- Resource-managed (two-party)
- PS-asserted (three-party)
- Federated (four-party)

### 3. Token Flow Visualization
Follow a complete token exchange:
```
Agent Token → Resource → Resource Token → PS → Auth Token → Resource
```

### 4. Mission Lifecycle Demo
Walk through a complete mission:
- Proposal → Clarification → Approval → Execution → Completion

### 5. Call Chaining Demo
Multi-hop authorization in action:
- Agent → Booking → Payment → Bank
- See the `act` claim build up through the chain

### 6. Interactive Consent Flow
User interaction with the Person Server:
- Deferred responses (202 Accepted)
- Polling pattern
- Interaction codes

---

## Running the Demos

```bash
# Agent identity demo
python demo/identity_demo.py

# Access modes comparison
python demo/access_modes_demo.py

# Token flow visualization
python demo/token_flow_demo.py

# Mission lifecycle
python demo/mission_demo.py

# Call chaining
python demo/call_chain_demo.py

# Interactive consent
python demo/consent_demo.py
```

---

## Expected Outputs

Each demo produces:
- **Console output** — Step-by-step explanation of what's happening
- **Token dumps** — Decoded JWTs showing structure
- **Flow diagrams** — ASCII art showing the protocol flow
- **Verification results** — Signature and token verification outcomes

---

## Demo Architecture

All demos use a consistent setup:

```
┌─────────────────────────────────────────────────────────────┐
│                      Demo Environment                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Mock Servers (in-memory):                                  │
│  ├── Agent Provider (issues agent tokens)                   │
│  ├── Resource Server (issues resource tokens)               │
│  ├── Person Server (issues auth tokens, handles consent)    │
│  └── Access Server (policy evaluation)                      │
│                                                              │
│  Agent Client:                                              │
│  ├── Generates keys                                         │
│  ├── Signs requests                                         │
│  └── Orchestrates flows                                     │
│                                                              │
│  Visualization:                                             │
│  ├── Token decoder (shows JWT structure)                    │
│  ├── Flow printer (shows request/response)                  │
│  └── Signature verifier (shows verification steps)          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Files

| File | Description |
|------|-------------|
| `identity_demo.py` | Agent identity creation and verification |
| `access_modes_demo.py` | Compare all four access modes |
| `token_flow_demo.py` | Visualize complete token exchange |
| `mission_demo.py` | Full mission lifecycle |
| `call_chain_demo.py` | Multi-hop authorization |
| `consent_demo.py` | Interactive consent with polling |
| `utils/` | Shared utilities for demos |

---

## What You'll Learn

1. **How agent identity works** — Keys, tokens, signatures
2. **When to use which mode** — Identity vs resource-managed vs PS-asserted vs federated
3. **How tokens flow** — From agent token to auth token
4. **What missions enable** — Governance, audit, clarification
5. **How delegation works** — Call chaining and the act claim

---

## Prerequisites

Complete Modules 00-08 first to build all the components!

Or run demos individually to see concepts in action before implementing them yourself.

---

## Sample Output

```
$ python demo/identity_demo.py

═══════════════════════════════════════════════════════════════
                    AAuth Identity Demo
═══════════════════════════════════════════════════════════════

[1] Generating Ed25519 key pair...
    ✓ Private key: (hidden)
    ✓ Public key:  OKP/Ed25519 x=nWGxne_9WmC6h...

[2] Creating agent token...
    ✓ Agent ID: aauth:demo-agent@localhost
    ✓ Token type: aa-agent+jwt
    
    Decoded payload:
    {
      "iss": "https://localhost:8000",
      "sub": "aauth:demo-agent@localhost",
      "cnf": {"jwk": {"kty": "OKP", "crv": "Ed25519", ...}},
      "ps": "https://localhost:8001",
      "iat": 1730217600,
      "exp": 1730304000
    }

[3] Signing HTTP request...
    Method: GET
    Path: /api/data
    Signature-Input: sig=("@method" "@authority" "@path" ...
    Signature: sig=:nWGxne_9WmC6hP9...:

[4] Verifying signature...
    ✓ Signature valid
    ✓ Agent identity confirmed: aauth:demo-agent@localhost

═══════════════════════════════════════════════════════════════
```

