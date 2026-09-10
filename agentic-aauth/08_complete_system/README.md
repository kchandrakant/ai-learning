# Module 08: Complete AAuth System

**Time**: ~90 minutes  
**Goal**: Put everything together into a working end-to-end system

---

## The Big Idea

We've built all the pieces. Now we assemble them:

```
┌─────────────────────────────────────────────────────────────┐
│                   Complete AAuth System                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐                                           │
│  │    Agent     │  Signs all requests                       │
│  │   Provider   │  Issues agent tokens                      │
│  └──────┬───────┘                                           │
│         │                                                    │
│         │ agent_token                                        │
│         ▼                                                    │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │              │ ──────▶ │              │                  │
│  │    Agent     │         │   Resource   │                  │
│  │              │ ◀────── │              │                  │
│  └──────┬───────┘         └──────┬───────┘                  │
│         │                        │                          │
│         │ resource_token         │                          │
│         ▼                        │                          │
│  ┌──────────────┐                │                          │
│  │    Person    │ ◀──────────────┘ (4-party)               │
│  │    Server    │────────▶┌──────────────┐                  │
│  └──────┬───────┘         │    Access    │                  │
│         │                 │    Server    │                  │
│         │ consent         └──────────────┘                  │
│         ▼                                                    │
│  ┌──────────────┐                                           │
│  │    User      │                                           │
│  └──────────────┘                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## System Components

### 1. Metadata Documents

Every server publishes metadata at well-known URLs:

| Server | URL | Purpose |
|--------|-----|---------|
| Agent Provider | `/.well-known/aauth-agent.json` | Agent identity verification |
| Resource | `/.well-known/aauth-resource.json` | Resource discovery |
| Person Server | `/.well-known/aauth-person.json` | PS capabilities |
| Access Server | `/.well-known/aauth-access.json` | AS capabilities |

### 2. JWKS Endpoints

Every server has a JWKS for token verification:

```
https://server.example/.well-known/jwks.json
```

Caching rules:
- Respect `Cache-Control` headers
- Refresh on unknown `kid`
- Max 1 refresh per minute per issuer
- Discard after 24 hours regardless

### 3. Token Types

| Token | `typ` | Issuer | Purpose |
|-------|-------|--------|---------|
| Agent | `aa-agent+jwt` | Agent Provider | Agent identity |
| Resource | `aa-resource+jwt` | Resource | Access request |
| Auth | `aa-auth+jwt` | PS or AS | Access grant |

---

## The Complete Flow

```
1. Agent obtains agent_token from Agent Provider
   └── Signs with agent's private key
   └── Contains ps claim pointing to Person Server

2. Agent calls Resource
   └── HTTPSig with agent_token
   └── Resource returns resource_token

3. Agent sends resource_token to PS
   └── PS evaluates consent/mission
   └── PS federates with AS if needed
   └── PS returns auth_token

4. Agent calls Resource with auth_token
   └── HTTPSig with auth_token
   └── Resource verifies and grants access
```

---

## Incremental Adoption

AAuth is designed for step-by-step adoption:

### For Resources

1. **Recognize signatures** → Identity-based access
2. **Add interaction flow** → Resource-managed access
3. **Accept PS identity** → PS-asserted access
4. **Deploy AS** → Federated access

### For Agents

1. **Get agent token, sign requests** → Basic identity
2. **Add PS claim** → Enable 3/4-party modes
3. **Add missions** → Full governance

### Adoption Matrix

| Agent has... | Resource supports... | Result |
|--------------|---------------------|--------|
| Agent token | Signatures | Identity-based |
| Agent token | Interaction | Resource-managed |
| Agent token + PS | Resource tokens | PS-asserted |
| Agent token + PS | AS | Federated |
| + Mission | Any | + Governance |

---

## What We'll Build

```
08_complete_system/
├── README.md
├── agent_provider/
│   ├── server.py          # Issues agent tokens
│   └── metadata.py        # .well-known endpoints
├── resource/
│   ├── server.py          # Protected resource
│   ├── auth.py            # Signature verification
│   └── metadata.py        # .well-known endpoints
├── person_server/
│   ├── server.py          # Full PS implementation
│   ├── consent.py         # Consent UI
│   └── metadata.py        # .well-known endpoints
├── access_server/
│   ├── server.py          # Policy evaluation
│   └── metadata.py        # .well-known endpoints
├── agent/
│   ├── client.py          # Agent implementation
│   └── mission.py         # Mission handling
└── demo.py                # End-to-end demonstration
```

---

## Demo Scenarios

We'll implement and test:

### Scenario 1: Identity-Based Access
```
Agent → Resource (no PS needed)
```

### Scenario 2: PS-Asserted Access
```
Agent → Resource → PS → Agent → Resource
```

### Scenario 3: Federated Access
```
Agent → Resource → PS → AS → PS → Agent → Resource
```

### Scenario 4: Mission-Governed Access
```
Agent → PS (mission) → Resource → PS → Agent → Resource
```

### Scenario 5: Call Chaining
```
Agent → Resource1 → Resource2 → PS → Resource2 → Resource1 → Agent
```

---

## Production Considerations

Things we'll discuss but not fully implement:

- **TLS everywhere** — All endpoints HTTPS
- **Key rotation** — Agents and servers rotate keys
- **Token revocation** — Real-time access termination
- **Rate limiting** — Protect against abuse
- **Monitoring** — Audit logs and metrics
- **High availability** — PS/AS redundancy

---

## Success Criteria

By the end, you'll have:

✓ Working Agent Provider issuing agent tokens  
✓ Resource verifying signatures and issuing resource tokens  
✓ Person Server handling consent and issuing auth tokens  
✓ Access Server evaluating policy  
✓ Agent orchestrating the full flow  
✓ Mission-governed authorization  
✓ Multi-hop call chaining  

---

## Let's Build It

This is the capstone module. We'll assemble everything into a working system.

**Say "ready" to begin.**

