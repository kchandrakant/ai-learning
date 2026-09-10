# AI Security & Identity: Key References

A curated collection of papers and standards on agent authentication, authorization, and security.

---

## 📚 How to Use This Document

- **📖 Essential**: Core standards and papers everyone should read
- **🔧 Hands-on**: Practical implementation resources
- **🔬 Frontier**: Emerging standards and research

---

## Identity Standards

### SPIFFE: Secure Production Identity Framework for Everyone
**CNCF**

Standard for workload identity in cloud-native environments.

- **Key innovations**: SPIFFE ID format, SVID certificates, trust domains
- **Impact**: Foundation for agent identity
- **Link**: [spiffe.io](https://spiffe.io/)
- **Status**: 📖 Essential

---

### OAuth 2.0 Authorization Framework
**RFC 6749**

The fundamental authorization standard.

- **Key concepts**: Access tokens, refresh tokens, scopes, grants
- **Impact**: Foundation for all web authorization
- **Link**: [RFC 6749](https://tools.ietf.org/html/rfc6749)
- **Status**: 📖 Essential

---

### OAuth 2.0 Token Exchange
**RFC 8693**

Standard for exchanging tokens between services.

- **Key use case**: Agent delegation chains
- **Link**: [RFC 8693](https://tools.ietf.org/html/rfc8693)
- **Status**: 🔧 Hands-on

---

## AI Agent Authorization (IETF Working Drafts)

### AI Agent Authorization Integration Framework
**draft-liu-ai-agent-authorization-integration, 2026**

Framework combining OAuth extensions for AI agents.

- **Key innovations**: Cross-domain identity, policy-based auth, consent evidence
- **Link**: [IETF](https://www.ietf.org/archive/id/draft-liu-ai-agent-authorization-integration-00.html)
- **Status**: 🔬 Frontier

---

### The Kindred Agent Identity Framework (KAIF)
**draft-lundholm-kaif, 2026**

OAuth 2.0 token exchange for agent-to-service authorization.

- **Key innovations**: SPIFFE + RFC 8693, operator-assigned tiers
- **Link**: [IETF](https://www.ietf.org/archive/id/draft-lundholm-kaif-00.html)
- **Status**: 🔬 Frontier

---

### Agent Authorization Envelope (AAE)
**draft-kroehl-agentic-trust-aae, 2026**

Structured authorization container for AI agents.

- **Key innovations**: MANDATE, CONSTRAINTS, VALIDITY blocks
- **Link**: [IETF](https://www.ietf.org/archive/id/draft-kroehl-agentic-trust-aae-01.html)
- **Status**: 🔬 Frontier

---

### Agent Authorization Use Cases and Gap Analysis
**draft-chen-oauth-agent-authz-use-cases, 2026**

Analysis of OAuth gaps for AI agents.

- **Key value**: Understanding what's missing in current standards
- **Link**: [IETF](https://www.ietf.org/archive/id/draft-chen-oauth-agent-authz-use-cases-02.html)
- **Status**: 🔬 Frontier

---

## Workload Identity

### WIMSE: Workload Identity in Multi-Service Environments
**IETF Working Group**

Standards for workload authentication.

- **Key topics**: Strong workload auth, token formats
- **Link**: [IETF WIMSE](https://datatracker.ietf.org/wg/wimse/about/)
- **Status**: 📖 Essential

---

### Service Identity and Token Exchange
**IETF**

Framework for service-to-service authentication.

- **Key value**: Agent-to-agent authentication patterns
- **Link**: [IETF](https://datatracker.ietf.org/doc/draft-ietf-wimse-s2s-protocol/)
- **Status**: 🔧 Hands-on

---

## LLM Security

### Prompt Injection: A Systematic Study
**Greshake et al., 2023**

Comprehensive analysis of prompt injection attacks.

- **Key topics**: Direct/indirect injection, attack taxonomy
- **Link**: [arXiv:2302.12173](https://arxiv.org/abs/2302.12173)
- **Status**: 📖 Essential

---

### Universal and Transferable Adversarial Attacks on Aligned Language Models
**Zou et al., 2023**

Jailbreaking attacks on aligned LLMs.

- **Key findings**: Transferable adversarial suffixes
- **Link**: [arXiv:2307.15043](https://arxiv.org/abs/2307.15043)
- **Status**: 📖 Essential

---

### Red Teaming Language Models to Reduce Harms
**Perez et al., Anthropic 2022**

Methodology for discovering LLM vulnerabilities.

- **Key innovations**: Automated red teaming
- **Link**: [arXiv:2209.07858](https://arxiv.org/abs/2209.07858)
- **Status**: 🔧 Hands-on

---

### Practices for Governing Agentic AI Systems
**OpenAI, 2024**

Guidelines for safe agent deployment.

- **Key topics**: Human oversight, monitoring, access control
- **Link**: [OpenAI](https://openai.com/research/practices-for-governing-agentic-ai-systems)
- **Status**: 📖 Essential

---

## Agent Security Research

### Task-Scoped Authorization for AI Agents via Natural Language Slices
**2024**

Authorization for agent operations, not operators.

- **Key innovations**: Operation-level (not operator-level) permissions
- **Link**: [arXiv:2603.17170](https://arxiv.org/abs/2603.17170)
- **Status**: 🔬 Frontier

---

### AI Agent Authentication and Authorization: Challenges and Emerging Standards
**2025**

Survey of agent security challenges.

- **Key value**: Comprehensive problem overview
- **Link**: [arXiv:2510.25819](https://arxiv.org/abs/2510.25819)
- **Status**: 📖 Essential

---

## Security Frameworks

### OWASP Top 10 for LLM Applications
**OWASP, 2023**

Security risks specific to LLM applications.

- **Key topics**: Prompt injection, data poisoning, supply chain
- **Link**: [OWASP](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- **Status**: 📖 Essential

---

### NIST AI Risk Management Framework
**NIST, 2023**

Guidelines for managing AI risks.

- **Key value**: Governance and compliance framework
- **Link**: [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework)
- **Status**: 📖 Essential

---

## Online Resources

| Resource | Description | Link |
|----------|-------------|------|
| SPIFFE | Workload identity standard | [spiffe.io](https://spiffe.io/) |
| SPIRE | SPIFFE runtime environment | [spiffe.io/spire](https://spiffe.io/spire/) |
| OAuth.net | OAuth resources | [oauth.net](https://oauth.net/) |
| OWASP LLM Top 10 | LLM security risks | [owasp.org](https://owasp.org/www-project-top-10-for-large-language-model-applications/) |

---

## Implementation Resources

| Resource | Description | Link |
|----------|-------------|------|
| Auth0 | Identity platform | [auth0.com](https://auth0.com/) |
| Okta | Enterprise identity | [okta.com](https://www.okta.com/) |
| Keycloak | Open-source IAM | [keycloak.org](https://www.keycloak.org/) |
| HashiCorp Vault | Secrets management | [vaultproject.io](https://www.vaultproject.io/) |

---

*Last updated: September 2026*
