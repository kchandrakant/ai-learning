# Module 06: Access Server & Federation

**Time**: ~45 minutes  
**Goal**: Understand enterprise/cross-domain authorization with Access Servers

---

## The Big Idea

When a resource needs **centralized policy enforcement**, it delegates to an Access Server (AS):

- PS handles user consent and identity
- AS handles resource policy and token issuance
- They federate — PS talks to AS, not the agent directly

**Key insight**: The PS is the only entity that calls AS token endpoints.

---

## When Do You Need an AS?

| Scenario | Use |
|----------|-----|
| Simple identity verification | PS-asserted (3-party) |
| Resource applies its own policy | PS-asserted (3-party) |
| Enterprise policy engine | Federated (4-party) |
| Cross-organization access | Federated (4-party) |
| Billing/payment required | Federated (4-party) |

---

## The Federation Flow

```
Agent          Resource          PS                    AS
  │               │               │                     │
  │  POST /authorize              │                     │
  │──────────────▶│               │                     │
  │               │               │                     │
  │  resource_token               │                     │
  │  (aud = AS URL)               │                     │
  │◀──────────────│               │                     │
  │               │               │                     │
  │  POST /token                  │                     │
  │  w/ resource_token            │                     │
  │──────────────────────────────▶│                     │
  │               │               │                     │
  │               │               │  POST /token        │
  │               │               │  resource_token     │
  │               │               │  agent_token        │
  │               │               │────────────────────▶│
  │               │               │                     │
  │               │               │  [AS evaluates      │
  │               │               │   resource policy]  │
  │               │               │                     │
  │               │               │  auth_token         │
  │               │               │◀────────────────────│
  │               │               │                     │
  │  auth_token                   │                     │
  │◀──────────────────────────────│                     │
  │               │               │                     │
  │  HTTPSig w/ auth_token        │                     │
  │──────────────▶│               │                     │
  │               │               │                     │
  │  200 OK       │               │                     │
  │◀──────────────│               │                     │
```

---

## AS Metadata

Published at `/.well-known/aauth-access.json`:

```json
{
  "issuer": "https://as.resource.example",
  "jwks_uri": "https://as.resource.example/.well-known/jwks.json",
  "token_endpoint": "https://as.resource.example/token",
  "scopes_supported": ["data.read", "data.write", "admin"]
}
```

---

## PS-to-AS Token Request

```http
POST /token HTTP/1.1
Host: as.resource.example
Content-Type: application/json
Signature-Input: sig=("@method" "@authority" "@path" "signature-key");created=...
Signature: sig=:...:
Signature-Key: sig=jwks_uri;jwks_uri="https://ps.example/.well-known/jwks.json"

{
  "resource_token": "eyJhbGc...",
  "agent_token": "eyJhbGc..."
}
```

Note: PS uses `jwks_uri` scheme, not `jwt` — the PS authenticates via its JWKS.

---

## AS Response Options

### Direct Grant (200)

```json
{
  "auth_token": "eyJhbGc...",
  "expires_in": 3600
}
```

### Claims Required (202)

AS needs identity claims before deciding:

```http
HTTP/1.1 202 Accepted
Location: /token/pending/xyz
AAuth-Requirement: requirement=claims

{
  "status": "pending",
  "required_claims": ["email", "tenant"]
}
```

PS responds by POSTing claims to the pending URL.

### Interaction Required (202)

AS needs user action (e.g., first-time binding):

```http
HTTP/1.1 202 Accepted
Location: /token/pending/xyz
AAuth-Requirement: requirement=interaction; url="https://as.example/bind"; code="X7K2"

{
  "status": "pending"
}
```

### Payment Required (402)

AS requires billing relationship:

```http
HTTP/1.1 402 Payment Required
Location: /token/pending/xyz
WWW-Authenticate: Payment id="x7Tg", method="stripe", request="..."
```

---

## Trust Establishment

Trust between PS and AS can be:

1. **Pre-established**: Business relationship configured
2. **Dynamic via interaction**: User binds their PS at the AS
3. **Dynamic via payment**: PS establishes billing relationship
4. **Claims-only**: AS trusts any PS providing sufficient claims

These can compose — payment first, then interaction, then claims.

---

## PS-AS Collapse

When PS and AS are the **same server**:

```
Agent          Resource          PS/AS
  │               │                │
  │  resource_token                │
  │  (aud = PS/AS URL)             │
  │◀──────────────│                │
  │               │                │
  │  POST /token                   │
  │───────────────────────────────▶│
  │               │                │ [single internal
  │               │                │  evaluation]
  │  auth_token                    │
  │◀───────────────────────────────│
```

This is common for organizations running their own infrastructure.

---

## What We'll Build

```
06_federation/
├── README.md
├── access_server.py       # AS implementation
├── federation.py          # PS-to-AS flow
├── claims_flow.py         # Claims requirement handling
└── demo.py                # Full 4-party flow
```

By the end:
1. AS that receives federated requests
2. Claims requirement flow
3. Complete 4-party authorization
4. PS-AS collapse scenario

---

## Auth Token Differences

| Issuer | `dwk` claim | Trust model |
|--------|-------------|-------------|
| PS (3-party) | `aauth-person.json` | Resource trusts PS identity |
| AS (4-party) | `aauth-access.json` | Resource trusts AS policy |

---

## Let's Start Building

We'll implement a basic AS and the federation flow.

**Say "ready" to begin.**

