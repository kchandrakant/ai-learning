# Agentic Authorization: Evolutions & Alternatives

The landscape of agent authorization is evolving rapidly. AAuth is one approach — here's how it compares to alternatives and what's emerging.

---

## Overview of Approaches

| Approach | Year | Philosophy | Status |
|----------|------|------------|--------|
| OAuth 2.0 + Extensions | 2012+ | Extend existing protocol | Production |
| GNAP | 2024 | Flexible grant negotiation | RFC 9635 |
| AAuth | 2025+ | Agent-native design | IETF Draft |
| SPIFFE/WIMSE | 2016+ | Workload identity | Production (enterprise) |
| Capability-based | 1970s+ | Object capabilities | Research/niche |

---

## Evolution 1: OAuth 2.0 Extensions for Agents

### The Approach
Extend OAuth 2.0 with agent-specific grants:
- **Device Authorization Grant** (RFC 8628) — For devices without browsers
- **Dynamic Client Registration** (RFC 7591) — Register clients on-the-fly
- **DPoP** (RFC 9449) — Proof-of-possession for tokens

### Pros
- Mature ecosystem
- Existing tooling and libraries
- Familiar to developers

### Cons
- Bearer tokens still a risk (DPoP helps but optional)
- Client IDs still don't travel
- No native mission/governance concept
- Consent is still synchronous

### Used By
- MCP (Model Context Protocol) adopted OAuth 2.1

### Files
- `oauth_device_flow.py` — Device authorization for agents
- `oauth_dpop.py` — DPoP proof-of-possession

---

## Evolution 2: GNAP (Grant Negotiation and Authorization Protocol)

### The Approach
Rebuild authorization from scratch with modern assumptions:
- Client identity without pre-registration
- Proof-of-possession by default
- Async authorization flows
- Rich authorization requests

### Pros
- Modern design
- Flexible interaction modes
- Better client identity model

### Cons
- Maximum flexibility = many profiling decisions
- No resource-signed challenges
- No native mission concept
- Limited adoption so far

### How It Differs from AAuth
| Aspect | GNAP | AAuth |
|--------|------|-------|
| Resource involvement | Passive | Active (signs resource tokens) |
| PS/AS separation | Combined | Separated |
| Mission governance | Not native | First-class |
| Federation model | Client-to-AS | PS-to-AS |

### Files
- `gnap_overview.py` — GNAP flow comparison

---

## Evolution 3: SPIFFE & Workload Identity

### The Approach
Give workloads (services, containers) cryptographic identity:
- **SPIFFE** — Secure Production Identity Framework for Everyone
- **SPIRE** — Reference implementation
- **WIMSE** — Workload Identity in Multi-Service Environments

### Pros
- Strong cryptographic identity
- Platform-attested (TEE, TPM support)
- Works in service mesh environments

### Cons
- Stops at trust domain boundaries
- No user-level consent
- Designed for infrastructure, not user-facing agents

### Relationship to AAuth
SPIFFE solves workload-to-workload identity within an enterprise. AAuth extends this to cross-domain agent-to-resource authorization with user governance.

### Files
- `spiffe_comparison.py` — SPIFFE vs AAuth identity

---

## Evolution 4: Capability-Based Security

### The Approach
Instead of identity-based access, pass unforgeable **capabilities**:
- A capability = reference + authority
- Whoever holds it can use it
- Can be attenuated (reduced authority)

### Pros
- Minimal authority principle
- No confused deputy problem
- Natural delegation

### Cons
- Different mental model
- Limited ecosystem
- Hard to audit (who has what?)

### Modern Implementations
- **UCAN** (User Controlled Authorization Networks)
- **Macaroons** (Google)
- **ZCAP-LD** (W3C)

### Files
- `capabilities_overview.py` — Capability-based auth concepts

---

## Evolution 5: Emerging MCP Authorization

### The Approach
MCP (Model Context Protocol) for agent-to-tool communication:
- Initially adopted OAuth 2.1
- Community exploring alternatives
- AAuth influence visible in discussions

### Current State
- MCP servers implement OAuth 2.1
- Device flow for headless agents
- Discussion around mission-like concepts

### Files
- `mcp_auth_landscape.py` — MCP authorization approaches

---

## Comparison Matrix

| Feature | OAuth 2.1 | GNAP | AAuth | SPIFFE |
|---------|-----------|------|-------|--------|
| Agent identity | Per-server | Ephemeral | Universal | Per-domain |
| Proof-of-possession | DPoP (optional) | Default | Required | Yes |
| Resource signs challenges | No | No | Yes | N/A |
| User consent | Sync only | Async possible | Async native | No user |
| Missions/governance | No | No | Yes | No |
| Cross-domain | Limited | Yes | Yes | No |
| Maturity | High | Medium | Low | High |

---

## The Authorization Landscape in 2026

```
                          User Involvement
                                ↑
                                │
           MCP/OAuth ─────────┬─┴─┬───────── AAuth
           (agent-to-service) │   │         (agent governance)
                              │   │
                              │   │
         SPIFFE/WIMSE ────────┼───┤
         (workload identity)  │   │
                              │   │
                              ↓   │
                      Zero Trust  │
                      Infrastructure
                                  │
            ◀─────────────────────┴──────────────────────▶
           Infrastructure                            Application
           (machine-to-machine)                      (user-facing)
```

---

## Files in this Directory

| File | Description |
|------|-------------|
| `oauth_device_flow.py` | OAuth Device Authorization for agents |
| `oauth_dpop.py` | DPoP proof-of-possession |
| `gnap_overview.py` | GNAP flow comparison |
| `spiffe_comparison.py` | SPIFFE vs AAuth |
| `capabilities_overview.py` | Capability-based security |
| `mcp_auth_landscape.py` | MCP authorization approaches |
| `comparison_matrix.py` | Side-by-side comparison tool |

---

## Key Takeaways

1. **OAuth works** but wasn't designed for agents — extensions help but don't fully solve
2. **GNAP is flexible** but requires profiling decisions and lacks resource involvement
3. **SPIFFE is great** for infrastructure but stops at trust boundaries
4. **Capabilities are elegant** but have limited ecosystem
5. **AAuth is agent-native** — designed for the new reality of autonomous agents

---

## What's Next?

The space is evolving. Watch for:
- **Convergence** — Ideas from AAuth appearing in OAuth/GNAP extensions
- **MCP evolution** — Model Context Protocol authorization improvements
- **Enterprise adoption** — SPIFFE + AAuth combinations
- **AI governance** — Mission-like concepts in regulatory frameworks

---

## Prerequisites

Understanding these evolutions helps you:
- Make informed architecture decisions
- Understand why AAuth makes certain choices
- Anticipate where the ecosystem is heading

Complete the AAuth learning path first, then explore alternatives here.

