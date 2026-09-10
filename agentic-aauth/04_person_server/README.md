# Module 04: Person Server (PS)

**Time**: ~60 minutes  
**Goal**: Build the user's representative in the AAuth protocol

---

## The Big Idea

The Person Server represents **the user** (or organization) to the rest of the protocol:

- Manages consent for resource access
- Asserts user identity to resources
- Handles missions and governance
- Brokers authorization with Access Servers

**Key insight**: The user chooses their PS. It's not imposed by any other party.

---

## What a PS Does

```
┌────────────────────────────────────────────────────────┐
│                    Person Server                        │
├────────────────────────────────────────────────────────┤
│                                                         │
│  Token Endpoint (/token)                               │
│  ├── Receives resource tokens from agents              │
│  ├── Handles user consent                              │
│  ├── Issues auth tokens (3-party)                      │
│  └── Federates with AS (4-party)                       │
│                                                         │
│  Mission Endpoint (/mission)                           │
│  ├── Receives mission proposals                        │
│  ├── Manages approval flow                             │
│  └── Returns approved missions                         │
│                                                         │
│  Permission Endpoint (/permission)                     │
│  └── Grants permission for tool calls                  │
│                                                         │
│  Audit Endpoint (/audit)                               │
│  └── Logs agent actions                                │
│                                                         │
│  Interaction Endpoint (/interaction)                   │
│  └── Relays interactions to user                       │
│                                                         │
└────────────────────────────────────────────────────────┘
```

---

## PS Metadata

Published at `/.well-known/aauth-person.json`:

```json
{
  "issuer": "https://ps.example",
  "jwks_uri": "https://ps.example/.well-known/jwks.json",
  "token_endpoint": "https://ps.example/token",
  "mission_endpoint": "https://ps.example/mission",
  "permission_endpoint": "https://ps.example/permission",
  "audit_endpoint": "https://ps.example/audit",
  "interaction_endpoint": "https://ps.example/interaction",
  "scopes_supported": ["openid", "profile", "email"]
}
```

---

## The Token Endpoint Flow

### Request

Agent sends a signed POST with the resource token:

```http
POST /token HTTP/1.1
Host: ps.example
Content-Type: application/json
Signature-Input: sig=("@method" "@authority" "@path" "signature-key");created=...
Signature: sig=:...:
Signature-Key: sig=jwt;jwt="<agent_token>"

{
  "resource_token": "eyJhbGc...",
  "justification": "Need calendar access to schedule the meeting"
}
```

### Response Options

**Immediate grant** (200):
```json
{
  "auth_token": "eyJhbGc...",
  "expires_in": 3600
}
```

**Consent required** (202 + deferred response):
```http
HTTP/1.1 202 Accepted
Location: /pending/abc123
Retry-After: 0
AAuth-Requirement: requirement=interaction; url="https://ps.example/consent"; code="A1B2-C3D4"

{
  "status": "pending"
}
```

---

## Deferred Responses & Polling

AAuth treats "pending" as a **first-class state**, not an error.

```
Agent                                    PS                    User
  │                                       │                      │
  │  POST /token                          │                      │
  │──────────────────────────────────────▶│                      │
  │                                       │                      │
  │  202 Accepted                         │                      │
  │  Location: /pending/abc               │                      │
  │  requirement=interaction              │                      │
  │◀──────────────────────────────────────│                      │
  │                                       │                      │
  │  [direct user to consent URL]         │                      │
  │───────────────────────────────────────│─────────────────────▶│
  │                                       │                      │
  │  GET /pending/abc                     │      [user decides]  │
  │──────────────────────────────────────▶│◀─────────────────────│
  │                                       │                      │
  │  202 (still pending)                  │                      │
  │◀──────────────────────────────────────│                      │
  │                                       │                      │
  │  GET /pending/abc                     │                      │
  │──────────────────────────────────────▶│                      │
  │                                       │                      │
  │  200 OK + auth_token                  │                      │
  │◀──────────────────────────────────────│                      │
```

---

## Interaction Codes

The `code` in the interaction requirement:

- **Crockford base32**: `0-9 A-Z` minus ambiguous `I L O U`
- **At least 40 bits entropy**: 8+ characters
- **Single use**: consumed when user arrives
- **Rate limited**: prevents brute force

```
code="A1B2-C3D4"  // Hyphens are for display only
```

User visits: `https://ps.example/consent?code=A1B2-C3D4`

---

## What We'll Build

```
04_person_server/
├── README.md
├── ps_server.py           # Flask/FastAPI PS implementation
├── token_endpoint.py      # Token issuance logic
├── consent_flow.py        # User consent handling
├── deferred.py            # Pending requests & polling
└── demo.py                # End-to-end with mock user
```

By the end:
1. PS that accepts resource tokens
2. Consent flow with deferred responses
3. Auth token issuance
4. Full polling loop

---

## PS Trust Model

The PS is a **high-value target** — it sees every authorization:

| The PS knows... | Why it matters |
|-----------------|----------------|
| Every resource accessed | Full visibility into agent activity |
| Every scope requested | What the agent wants to do |
| Mission context | What the agent is trying to accomplish |
| User identity | Who the agent acts for |

**Mitigation**: User chooses their PS. PS can delegate auth to external IdPs.

---

## Let's Start Building

We'll create a minimal PS that can:
1. Accept token requests
2. Handle consent (simulated)
3. Issue auth tokens
4. Support the polling pattern

**Say "ready" to begin.**

