# Module 01: Agent Identity & HTTP Signatures

**Time**: ~45 minutes  
**Goal**: Give an agent its own cryptographic identity that works everywhere

---

## The Big Idea

In AAuth, every agent has its own **cryptographic identity**:

- **Agent Identifier**: `aauth:local@domain` (like `aauth:assistant-v2@agent.example`)
- **Signing Key**: Ed25519 key pair the agent owns
- **Agent Token**: JWT binding the identifier to the signing key

No pre-registration. No shared secrets. The agent proves who it is by signing every request.

---

## What We'll Learn

### 1. Agent Identifiers

```
aauth:assistant-v2@agent.example
      └─────┬─────┘ └────┬─────┘
         local       domain
         part     (agent provider)
```

- Format: `aauth:{local}@{domain}`
- Local part: lowercase letters, digits, hyphen, underscore, plus, period
- Domain: the agent provider's domain (issues agent tokens)

### 2. Key Pairs

Agents generate their own Ed25519 key pairs:

```python
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

private_key = Ed25519PrivateKey.generate()
public_key = private_key.public_key()
```

### 3. HTTP Message Signatures (RFC 9421)

Every request the agent makes is **signed**:

```http
POST /api/data HTTP/1.1
Host: resource.example
Signature-Input: sig=("@method" "@authority" "@path" "signature-key");created=1730217600
Signature: sig=:BASE64_SIGNATURE:
Signature-Key: sig=jwt;jwt="eyJhbGc..."
```

The signature proves:
- The request came from this specific agent
- The request hasn't been tampered with
- The agent holds the private key

### 4. Agent Tokens

A JWT issued by the agent provider:

```json
{
  "typ": "aa-agent+jwt",
  "alg": "EdDSA"
}
{
  "iss": "https://agent.example",
  "sub": "aauth:assistant-v2@agent.example",
  "cnf": {
    "jwk": { "kty": "OKP", "crv": "Ed25519", "x": "..." }
  },
  "ps": "https://ps.example",
  "iat": 1730217600,
  "exp": 1730304000
}
```

Key claims:
- `iss`: Agent provider URL
- `sub`: Agent identifier  
- `cnf.jwk`: Public key (for verification)
- `ps`: Person Server URL (optional, enables 3/4-party modes)

---

## What We'll Build

```
01_agent_identity/
├── README.md
├── agent_identity.py      # Agent class with identity
├── http_signatures.py     # Sign and verify requests
├── agent_token.py         # Create and verify agent tokens
└── demo.py                # End-to-end demo
```

By the end:
1. Generate an agent's key pair
2. Create a valid agent token
3. Sign HTTP requests
4. Verify signatures and tokens

---

## Key Concepts

| Concept | OAuth | AAuth |
|---------|-------|-------|
| Identity | `client_id` per server | `aauth:local@domain` universal |
| Credential | Shared secret or bearer token | Signing key (never shared) |
| Proof | Present the token | Sign the request |
| Theft protection | None (bearer = access) | Useless without private key |

---

## Let's Start Building

When ready, we'll implement:

1. **Key generation** — Create an Ed25519 key pair
2. **Agent token** — Build a valid `aa-agent+jwt`
3. **Request signing** — Implement HTTP Message Signatures
4. **Verification** — Verify signatures against the agent token

**Say "ready" to begin.**

