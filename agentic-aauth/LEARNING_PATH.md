# AAuth Learning Path

A hands-on journey to understand the **AAuth protocol** — the emerging standard for agent-to-resource authorization. We'll build everything together, step by step.

## What is AAuth?

AAuth (Agent Authentication & Authorization) is a new protocol designed specifically for AI agents. Created by Dick Hardt (author of OAuth 2.0), AAuth addresses fundamental limitations of OAuth when applied to autonomous agents.

**Key insight**: OAuth was designed for pre-registered clients with fixed integrations. Agents discover resources at runtime, execute long-running tasks across trust domains, and need authorization decisions mid-task. AAuth is built for this reality.

## Why Not Just Use OAuth?

Before diving into AAuth, we'll briefly understand why OAuth falls short for agents:

| OAuth Limitation | Problem for Agents |
|-----------------|-------------------|
| Client IDs don't travel | A `client_id` at Google is meaningless at GitHub — agents have no portable identity |
| Bearer tokens leak | API keys are shared secrets that eventually get copied somewhere they shouldn't |
| Consent is synchronous | Agents need authorization mid-task; OAuth treats "pending" as an error |
| Scopes are standing permission | `mail.read` looks the same whether summarizing or scraping — no per-call intent |
| No cross-domain identity | SPIFFE/workload identity stops at trust-domain boundaries |

AAuth solves these by giving every agent its own **cryptographic identity** that travels everywhere.

---

## Learning Approach

This is **not** a reading exercise. We build code together:

1. **I explain the concept** — brief theory, what we're building
2. **You implement** — I guide, you write the code
3. **We test and iterate** — see it work, understand why

Each module has a clear goal. We won't move on until you've built something that works.

---

## Module Structure

### Module 00: Why OAuth Falls Short
**Goal**: Understand the limitations that motivated AAuth

- Quick OAuth 2.0 refresher (15 min)
- The 5 problems for agents
- See a concrete example of each limitation
- **Build**: A simple demo showing OAuth's agent limitations

### Module 01: Agent Identity & HTTP Signatures  
**Goal**: Give an agent its own cryptographic identity

- Agent identifiers (`aauth:local@domain`)
- Key pairs and signing
- HTTP Message Signatures (RFC 9421)
- Agent tokens (JWT structure)
- **Build**: Create and verify an agent's identity

### Module 02: The Four Access Modes
**Goal**: Understand how agents access resources progressively

- Identity-based access (replaces API keys)
- Resource-managed access (two-party)
- PS-asserted access (three-party)  
- Federated access (four-party)
- **Build**: Implement identity-based access to a mock resource

### Module 03: Resource Tokens & Auth Tokens
**Goal**: Understand AAuth's token flow

- Resource tokens (what the resource wants)
- Auth tokens (what the agent can do)
- Proof-of-possession binding
- Token verification
- **Build**: Issue and verify resource/auth tokens

### Module 04: Person Server (PS)
**Goal**: Build the user's representative in the protocol

- What a PS does (consent, identity, governance)
- Token endpoint implementation
- User interaction flow (202 Accepted + polling)
- Deferred responses
- **Build**: A minimal Person Server

### Module 05: Missions & Governance
**Goal**: Add governance over agent actions

- Mission creation and approval
- Mission logs (audit trail)
- Permission endpoint (for tool calls)
- Audit endpoint (logging actions)
- Clarification chat
- **Build**: Mission-aware agent governance

### Module 06: Access Server & Federation
**Goal**: Understand enterprise/cross-domain authorization

- When you need an AS vs just a PS
- PS-to-AS federation flow
- Claims requirements
- Trust establishment
- **Build**: Federated authorization flow

### Module 07: Call Chaining & Delegation
**Goal**: Handle multi-hop agent scenarios

- Resources acting as agents
- Upstream/downstream tokens
- The `act` claim for delegation chains
- Sub-agents
- **Build**: Multi-hop authorization chain

### Module 08: Complete AAuth System
**Goal**: Put it all together

- Metadata documents (`.well-known/`)
- JWKS discovery and caching
- Incremental adoption strategies
- **Build**: End-to-end working AAuth system

---

## What We'll Build

By the end, you'll have implemented:

```
┌─────────────────────────────────────────────────────────────┐
│                     Your AAuth System                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    signs requests    ┌──────────────┐        │
│  │  Agent   │ ──────────────────▶  │   Resource   │        │
│  │          │ ◀────────────────── │              │        │
│  └────┬─────┘    resource token    └──────────────┘        │
│       │                                    │                │
│       │ token request                      │                │
│       ▼                                    │                │
│  ┌──────────┐                              │                │
│  │  Person  │    federation       ┌───────┴──────┐        │
│  │  Server  │ ◀─────────────────▶ │   Access     │        │
│  │   (PS)   │                     │   Server     │        │
│  └──────────┘                     └──────────────┘        │
│       │                                                    │
│       │ consent                                            │
│       ▼                                                    │
│  ┌──────────┐                                              │
│  │  Person  │                                              │
│  │  (User)  │                                              │
│  └──────────┘                                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

- Python 3.10+
- Basic understanding of HTTP and REST APIs
- Familiarity with JWTs (we'll review)
- Comfort with cryptographic concepts (signing, verification)

---

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Verify setup
python verify_setup.py

# Start Module 00
cd 00_oauth_limitations
```

---

---

## Demos

After completing the modules, run interactive demonstrations:

```
demo/
├── identity_demo.py       # Agent identity in action
├── access_modes_demo.py   # Compare all 4 modes side-by-side
├── token_flow_demo.py     # Visualize complete token exchange
├── mission_demo.py        # Full mission lifecycle
├── call_chain_demo.py     # Multi-hop authorization
└── consent_demo.py        # Interactive consent flow
```

Run a demo: `python demo/identity_demo.py`

---

## Evolutions & Alternatives

Understand where AAuth fits in the broader landscape:

```
evolutions/
├── oauth_device_flow.py   # OAuth extensions for agents
├── gnap_overview.py       # GNAP comparison
├── spiffe_comparison.py   # Workload identity
├── capabilities_overview.py # Capability-based security
└── mcp_auth_landscape.py  # MCP authorization
```

Learn how AAuth compares to OAuth extensions, GNAP, SPIFFE, and capability-based approaches.

---

## Resources

- [AAuth Protocol Spec](https://www.aauth.dev/) — Official site
- [IETF Draft](https://www.ietf.org/archive/id/draft-hardt-oauth-aauth-protocol-08.html) — Full protocol specification
- [AAuth Explorer](https://explorer.aauth.dev/) — Interactive protocol walkthrough
- [HTTP Message Signatures RFC 9421](https://www.rfc-editor.org/rfc/rfc9421.html)
- [GNAP RFC 9635](https://www.rfc-editor.org/rfc/rfc9635.html) — Grant Negotiation and Authorization
- [SPIFFE](https://spiffe.io/) — Workload Identity Framework

---

## Let's Begin

Ready? We start with **Module 00** to see exactly why OAuth doesn't work for agents — then we build something better.

