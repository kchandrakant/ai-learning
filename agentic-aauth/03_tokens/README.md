# Module 03: Resource Tokens & Auth Tokens

**Time**: ~45 minutes  
**Goal**: Understand AAuth's token flow — what each token represents and how they're verified

---

## The Big Idea

AAuth has **four token types**, each with a specific purpose:

| Token | Issued By | Purpose |
|-------|-----------|---------|
| Agent Token | Agent Provider | Establishes agent identity |
| Resource Token | Resource | Describes what access is needed |
| Auth Token | PS or AS | Grants access to a resource |
| (AAuth-Access) | Resource | Opaque session token (Mode 2 only) |

The key insight: **every token is proof-of-possession**. A stolen token is useless without the private key.

---

## Token Flow

```
                                    Agent Provider
                                          │
                                          │ 1. Agent Token
                                          ▼
┌──────────┐                        ┌──────────┐
│          │   2. Request access    │          │
│  Agent   │ ──────────────────────▶│ Resource │
│          │                        │          │
│          │   3. Resource Token    │          │
│          │ ◀──────────────────────│          │
└────┬─────┘                        └──────────┘
     │
     │ 4. Exchange resource token
     ▼
┌──────────┐                        ┌──────────┐
│  Person  │   5. Federation        │  Access  │
│  Server  │ ◀─────────────────────▶│  Server  │
│          │                        │          │
└────┬─────┘                        └──────────┘
     │
     │ 6. Auth Token
     ▼
┌──────────┐
│  Agent   │   7. Access resource with auth token
│          │ ─────────────────────────────────────▶
└──────────┘
```

---

## Resource Token

Issued by the **resource** to describe what access is being requested.

```json
{
  "typ": "aa-resource+jwt",
  "alg": "EdDSA"
}
{
  "iss": "https://resource.example",
  "aud": "https://ps.example",          // or AS URL
  "agent": "aauth:assistant@agent.example",
  "agent_jkt": "NzbLsXh8uDCcd...",       // JWK thumbprint
  "scope": "data.read data.write",
  "iat": 1730217600,
  "exp": 1730217900,                     // Short-lived (5 min)
  "jti": "unique-token-id"
}
```

Key points:
- **Signed by the resource** — proves the resource is asking for this
- **Contains agent identity** — binds the request to a specific agent
- **Short-lived** — typically 5 minutes
- **`aud` determines the flow** — PS URL = 3-party, AS URL = 4-party

### Optional: Mission Reference

When operating under a mission:

```json
{
  "mission": {
    "approver": "https://ps.example",
    "s256": "dBjftJeZ4CVP-mB92K27uhbUJU1p1r..."  // SHA-256 of mission
  }
}
```

---

## Auth Token

Issued by **PS or AS** to grant access to a specific resource.

```json
{
  "typ": "aa-auth+jwt",
  "alg": "EdDSA"
}
{
  "iss": "https://ps.example",           // or AS URL
  "aud": "https://resource.example",
  "agent": "aauth:assistant@agent.example",
  "sub": "user-directed-identifier",     // Pairwise per resource
  "scope": "data.read data.write",
  "cnf": {
    "jwk": { "kty": "OKP", "crv": "Ed25519", "x": "..." }
  },
  "iat": 1730217600,
  "exp": 1730221200,                     // Max 1 hour
  "jti": "unique-token-id"
}
```

Key points:
- **`cnf.jwk`** — proof-of-possession; agent must sign with this key
- **`sub`** — directed identifier (different per resource for privacy)
- **`scope`** — what the agent can do (never broader than resource token)
- **Max 1 hour lifetime** — short-lived, re-authorize frequently

### Optional: Delegation Chain

When call chaining (resource acts as agent):

```json
{
  "act": {
    "agent": "aauth:upstream@other.example"
  }
}
```

---

## Proof-of-Possession

The critical difference from OAuth bearer tokens:

```
OAuth Bearer Token:
┌─────────────────────────────────────────────┐
│ Anyone with the token can use it            │
│ Stolen token = full access                  │
└─────────────────────────────────────────────┘

AAuth PoP Token:
┌─────────────────────────────────────────────┐
│ Token contains: cnf.jwk (public key)        │
│ Every request must be signed with the       │
│ corresponding private key                   │
│ Stolen token = useless without private key  │
└─────────────────────────────────────────────┘
```

---

## What We'll Build

```
03_tokens/
├── README.md
├── resource_token.py      # Create and verify resource tokens
├── auth_token.py          # Create and verify auth tokens
├── token_exchange.py      # The exchange flow
└── demo.py                # End-to-end token flow
```

By the end:
1. Create valid resource tokens
2. Create valid auth tokens
3. Verify both token types
4. Implement a complete token exchange

---

## Verification Checklist

### Resource Token Verification (at PS/AS)

1. ✓ Verify JWT signature against resource's JWKS
2. ✓ Check `typ` is `aa-resource+jwt`
3. ✓ Verify `exp` is in the future
4. ✓ Verify `aud` matches recipient (PS or AS)
5. ✓ Verify `agent` matches requesting agent
6. ✓ Verify `agent_jkt` matches request signature key

### Auth Token Verification (at Resource)

1. ✓ Verify JWT signature against issuer's JWKS
2. ✓ Check `typ` is `aa-auth+jwt`
3. ✓ Verify `exp` is in the future
4. ✓ Verify `aud` matches this resource
5. ✓ Verify `cnf.jwk` matches request signature key
6. ✓ Check `scope` grants required permissions

---

## Let's Start Building

We'll implement both token types and the full verification flow.

**Say "ready" to begin.**

