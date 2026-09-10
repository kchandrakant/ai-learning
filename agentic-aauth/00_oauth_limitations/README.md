# Module 00: Why OAuth Falls Short for Agents

**Time**: ~30 minutes  
**Goal**: Understand the specific limitations of OAuth 2.0 that motivated the creation of AAuth

---

## Quick OAuth 2.0 Refresher

OAuth 2.0 is the dominant authorization protocol on the web. Here's the core flow:

```
┌──────────┐                              ┌───────────────┐
│  User    │                              │ Authorization │
│ (Resource│     1. Redirect to login     │    Server     │
│  Owner)  │ ◀─────────────────────────── │               │
└────┬─────┘                              └───────┬───────┘
     │                                            │
     │ 2. User logs in & consents                 │
     │                                            │
     ▼                                            │
┌──────────┐     3. Authorization code            │
│  Client  │ ◀────────────────────────────────────┘
│  (App)   │
└────┬─────┘     4. Exchange code for tokens
     │           ─────────────────────────────────▶
     │
     │           5. Access token
     │           ◀─────────────────────────────────
     │
     ▼
┌──────────┐     6. API call with token
│ Resource │ ◀─────────────────────────────────────
│  Server  │
└──────────┘
```

**Key concepts**:
- **Client**: The app wanting access (pre-registered with client_id/secret)
- **Resource Owner**: The user who owns the data
- **Authorization Server**: Issues tokens after user consent
- **Resource Server**: The API being accessed
- **Access Token**: Bearer credential to access resources
- **Scopes**: Permissions the client requests (e.g., `read:email`)

---

## The Five Problems for Agents

### Problem 1: Client IDs Don't Travel

In OAuth, a `client_id` is issued by each authorization server. It's **meaningless** outside that relationship.

```python
# Your agent registered at Google
google_client_id = "abc123.apps.googleusercontent.com"

# Same agent at GitHub? You need a DIFFERENT client_id
github_client_id = "Iv1.def456"

# At Slack? Another one.
slack_client_id = "789012345678.apps"
```

**The problem**: An agent interacting with 100 services needs 100 separate registrations. Agents have no **portable identity**.

---

### Problem 2: Bearer Tokens Leak

OAuth access tokens are **bearer credentials** — anyone who has the token can use it.

```python
# Stolen token = full access
headers = {"Authorization": f"Bearer {stolen_token}"}
requests.get("https://api.example.com/sensitive-data", headers=headers)
# Works! No proof you're the legitimate holder.
```

**The problem**: Tokens get logged, cached, leaked. A stolen token grants full access with no way to verify the presenter is the legitimate holder.

---

### Problem 3: Consent is Synchronous

OAuth expects the user to be present during authorization. The flow **blocks** until consent is given.

```python
# Agent wants to access a new service mid-task
# OAuth says: "Redirect user to consent page NOW"
# But the user started this task 2 hours ago and walked away!

# OAuth's response to "consent pending"? 
# Error: access_denied
```

**The problem**: Agents run autonomously. They discover they need access to new resources **during** task execution, long after the user set them in motion.

---

### Problem 4: Scopes Are Standing Permission

OAuth scopes grant broad, static permissions. They don't capture **why** the agent needs access right now.

```python
# Both of these use the same scope
scope = "mail.read"

# Use case 1: Summarize unread emails (legitimate)
agent.summarize_inbox()

# Use case 2: Scrape all emails for data mining (concerning)
agent.export_all_emails()

# Policy can't distinguish them — same scope!
```

**The problem**: There's no way to express per-call intent. `mail.read` for summarizing looks identical to `mail.read` for bulk extraction.

---

### Problem 5: Calls Cross Trust Domains

A single agent task can span multiple organizations, clouds, and identity systems.

```
Agent Task: "Book my trip to Tokyo"

1. Search flights → TravelAPI (your company's subscription)
2. Check calendar → Google Workspace (personal account)  
3. Book hotel → Booking.com (no account yet!)
4. Process payment → Stripe (company account)
5. Add to expenses → SAP (corporate ERP)
```

Each of these has different:
- Identity providers
- Authorization servers
- Trust relationships

**The problem**: There's no standard way for the agent to carry its identity and authorization across these boundaries.

---

## What We'll Build

Let's make these problems concrete. We'll create a simple scenario that shows each limitation in action.

**Scenario**: An AI agent that helps users manage tasks across multiple services.

```
our_agent/
├── demo_problems.py      # We'll build this together
└── mock_services.py      # Simple mock OAuth services
```

---

## Let's Start Building

When you're ready, we'll implement:

1. A mock OAuth flow showing client_id fragmentation
2. A bearer token that gets "stolen" and reused
3. An async task that fails because consent isn't available
4. Two requests with identical scopes but very different intents
5. A cross-domain task that OAuth can't handle

**Say "ready" and we'll write the first demo together.**

---

## Key Takeaways (Preview)

After this module, you'll understand:

- OAuth was designed for **web apps with pre-registered integrations**
- Agents need **portable identity** that works everywhere
- Agents need **proof-of-possession** (can't use stolen tokens)
- Agents need **async consent** (pending is a valid state, not an error)
- Agents need **per-call intent** (why am I accessing this now?)
- Agents need **cross-domain identity** (one identity, many services)

**AAuth addresses all five problems.** We'll see how in Module 01.

