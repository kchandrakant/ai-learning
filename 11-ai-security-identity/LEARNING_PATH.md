# AI Security & Identity: A Step-by-Step Learning Journey

This course covers security, identity, and authorization patterns for AI agents and LLM-powered applications. From API key management to advanced agent identity protocols, learn to build secure AI systems.

---

## 🎯 Why AI Security Matters

AI agents introduce new security challenges:
- **Identity:** How do we identify agents vs humans?
- **Authorization:** What can an agent do on behalf of a user?
- **Trust:** How do agents establish trust across systems?
- **Audit:** How do we track what agents did?

Traditional OAuth was designed for pre-registered apps with human consent. Agents operate differently—they discover resources at runtime, execute long-running tasks, and need dynamic authorization.

---

## 🎯 Prerequisites

- Basic understanding of HTTP and REST APIs
- Familiarity with OAuth 2.0 concepts (we'll review)
- Python 3.10+
- Comfort with cryptographic concepts (signing, hashing)

---

## 📚 Part 1: Security Foundations

Understanding the baseline before agent-specific challenges.

### Step 1: API Key Management
**Why it matters:** Most LLM integrations start with API keys. Managing them securely is foundational.

**What we'll cover:**
- API key security best practices
- Secret management (environment variables, vaults)
- Key rotation strategies
- Rate limiting and abuse prevention
- Monitoring and alerting

**Key insight:**
```
API keys are shared secrets — if leaked, anyone can use them.
This is the fundamental problem agent identity solves.
```

---

### Step 2: OAuth 2.0 for AI Applications
**Why it matters:** OAuth is the foundation of modern authorization. Understanding its strengths and limits is essential.

**What we'll cover:**
- OAuth 2.0 flows (auth code, client credentials, device)
- Token types (access, refresh, ID tokens)
- Scopes and permissions
- OAuth for LLM applications
- When OAuth works and when it doesn't

**Key limitation:**
```
OAuth was designed for:
- Pre-registered clients (known client_id)
- Synchronous consent (user clicks "Allow")
- Standing permissions (scopes don't change per-call)

Agents need dynamic, per-task authorization.
```

---

### Step 3: OAuth Limitations for Agents
**Why it matters:** Understanding OAuth's gaps motivates the need for agent-specific protocols.

**What we'll cover:**
- Client IDs don't travel between domains
- Bearer tokens can be stolen
- Consent is synchronous
- Scopes are coarse-grained
- No cross-domain identity

**Build:** Demo showing OAuth limitations with agents

---

## 📚 Part 2: Agent Identity

How agents prove who they are.

### Step 4: Cryptographic Identity
**Why it matters:** Agents need portable, verifiable identity that works across systems.

**What we'll cover:**
- Public key cryptography recap
- Ed25519 key pairs
- Agent identifiers
- HTTP Message Signatures (RFC 9421)
- Self-sovereign identity concepts

**Key pattern:**
```
Agent generates key pair → Public key = identity
Agent signs requests → Recipient verifies signature
Identity travels with the agent across systems
```

---

### Step 5: Agent Tokens & Claims
**Why it matters:** Tokens carry identity and permissions in a verifiable package.

**What we'll cover:**
- JWT structure for agents
- Agent-specific claims
- Token lifetimes and rotation
- Proof-of-possession tokens
- Token verification

**Build:** Create and verify agent identity tokens

---

### Step 6: Identity Providers for Agents
**Why it matters:** Centralized identity management scales agent deployments.

**What we'll cover:**
- Agent registration
- Key management
- Identity federation
- SPIFFE/SPIRE for workload identity
- Cloud provider identity (AWS IAM roles, GCP service accounts)

---

## 📚 Part 3: Authorization Models

What agents are allowed to do.

### Step 7: Access Control Patterns
**Why it matters:** Different patterns suit different use cases.

**What we'll cover:**
- Role-based access control (RBAC)
- Attribute-based access control (ABAC)
- Capability-based security
- Policy engines (OPA, Cedar)
- Least privilege for agents

**Comparison:**
```
RBAC:  "Admin role can do X" — simple, coarse
ABAC:  "If user.dept=sales AND time<5pm" — flexible
Capability: "This token lets you read file X" — fine-grained
```

---

### Step 8: Per-Task Authorization
**Why it matters:** Agents need authorization that matches their task, not standing permissions.

**What we'll cover:**
- Task-scoped tokens
- Just-in-time authorization
- Consent at action time
- Mission-based authorization
- Deferred consent (async approval)

**Pattern:**
```
Agent: "I need to send an email for user"
System: "Hold — get user approval first"
User: "Yes, send to john@example.com only"
Agent: proceeds with scoped permission
```

---

### Step 9: The AAuth Protocol
**Why it matters:** AAuth (by Dick Hardt, creator of OAuth) addresses agent-specific authorization needs.

**What we'll cover:**
- AAuth architecture
- Agent identity in AAuth
- Resource tokens vs auth tokens
- Person Server concept
- The four access modes

**Build:** Implement AAuth token flow

---

### Step 10: Person Server & Consent
**Why it matters:** The Person Server represents user interests in agent interactions.

**What we'll cover:**
- Person Server role
- Consent management
- Deferred responses (202 Accepted)
- Mission governance
- Audit logging

**Build:** Minimal Person Server implementation

---

## 📚 Part 4: Trust & Federation

Trust across organizational boundaries.

### Step 11: Trust Models
**Why it matters:** Agents operate across trust boundaries (organizations, clouds, services).

**What we'll cover:**
- Direct trust vs federated trust
- Trust anchors
- Certificate chains
- Web of trust
- Zero trust architecture

---

### Step 12: Federation Patterns
**Why it matters:** Agents need to work across multiple organizations and systems.

**What we'll cover:**
- Identity federation
- Cross-domain authorization
- Trust establishment
- Claims mapping
- Federation protocols

---

### Step 13: Call Chaining & Delegation
**Why it matters:** Agent A calls Agent B calls Agent C — how does trust flow?

**What we'll cover:**
- Delegation tokens
- The `act` claim
- Upstream/downstream authorization
- Audit trails for delegation
- Limiting delegation depth

**Build:** Multi-hop agent authorization

---

## 📚 Part 5: Production Security

Real-world security implementation.

### Step 14: Secure Agent Deployment
**Why it matters:** Secure code means nothing if deployment is weak.

**What we'll cover:**
- Secret injection patterns
- Sandboxing agents
- Network isolation
- Supply chain security
- Container security

---

### Step 15: Monitoring & Audit
**Why it matters:** You can't secure what you can't see.

**What we'll cover:**
- Security logging for agents
- Audit trail requirements
- Anomaly detection
- Incident response
- Compliance considerations

---

### Step 16: Threat Modeling for AI
**Why it matters:** AI systems have unique attack surfaces.

**What we'll cover:**
- Prompt injection attacks
- Agent impersonation
- Token theft and replay
- Excessive permissions
- Data exfiltration via agents

---

## 🗂️ Project Structure

```
ai-security-identity/
├── LEARNING_PATH.md
├── requirements.txt
├── verify_setup.py
│
├── 01_api_key_management/
├── 02_oauth_for_ai/
├── 03_oauth_limitations/
├── 04_cryptographic_identity/
├── 05_agent_tokens/
├── 06_identity_providers/
├── 07_access_control/
├── 08_per_task_auth/
├── 09_aauth_protocol/
├── 10_person_server/
├── 11_trust_models/
├── 12_federation/
├── 13_call_chaining/
├── 14_secure_deployment/
├── 15_monitoring_audit/
├── 16_threat_modeling/
│
├── demo/
└── beyond/
```

---

## 🚀 Let's Begin!

Start with **Step 1: API Key Management** — the foundation of secure AI integration.

---

## 📖 References

### Specifications
- [AAuth Protocol](https://www.aauth.dev/)
- [OAuth 2.0 RFC 6749](https://tools.ietf.org/html/rfc6749)
- [HTTP Message Signatures RFC 9421](https://www.rfc-editor.org/rfc/rfc9421.html)
- [GNAP RFC 9635](https://www.rfc-editor.org/rfc/rfc9635.html)
- [SPIFFE](https://spiffe.io/)

### Books & Articles
- "OAuth 2.0 Simplified" by Aaron Parecki
- "API Security in Action" by Neil Madden
- OWASP API Security Top 10
