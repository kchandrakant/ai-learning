# Module 05: Missions & Governance

**Time**: ~60 minutes  
**Goal**: Add governance over agent actions with missions, permissions, and audit

---

## The Big Idea

Missions provide **scoped authorization contexts** for agent work:

- Agent proposes what it intends to do
- PS/user reviews, clarifies, approves
- Every subsequent action is evaluated against the mission
- Full audit trail of what happened

**Key insight**: Missions are natural language, not policy rules. The PS (human or AI) interprets intent.

---

## Why Missions?

Without missions:
```
Agent: "I need mail.read"
Resource: "OK, here's access"
User: "Wait, why is it reading ALL my emails?!"
```

With missions:
```
Agent: "I want to summarize your unread emails from today"
User: "That sounds reasonable, approved"
Agent: "I need mail.read"
PS: "This is within the approved mission scope, granted"
```

---

## Mission Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                     Mission Lifecycle                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. PROPOSAL                                                │
│     Agent → PS: "Here's what I want to do"                  │
│                                                              │
│  2. CLARIFICATION (optional)                                │
│     PS ↔ Agent: "Why do you need X?"                        │
│     User ↔ Agent: "Can you do Y instead?"                   │
│                                                              │
│  3. APPROVAL                                                │
│     User → PS: "Approved" (with any modifications)          │
│     PS → Agent: Mission blob + s256 hash                    │
│                                                              │
│  4. EXECUTION                                               │
│     Agent works, includes mission reference in requests     │
│     PS evaluates each request against mission context       │
│                                                              │
│  5. COMPLETION                                              │
│     Agent → PS: "Here's what I accomplished"                │
│     User: Accept or request follow-up                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Mission Structure

### Proposal

```json
{
  "description": "# Plan Japan Vacation\n\nPlan and book a trip to Japan...",
  "tools": [
    {"name": "WebSearch", "description": "Search the web"},
    {"name": "BookFlight", "description": "Book flights"}
  ]
}
```

### Approved Mission (the "blob")

```json
{
  "approver": "https://ps.example",
  "agent": "aauth:assistant@agent.example",
  "approved_at": "2026-04-07T14:30:00Z",
  "description": "# Plan Japan Vacation\n\n...",
  "approved_tools": [
    {"name": "WebSearch", "description": "Search the web"}
  ],
  "capabilities": ["interaction", "payment"]
}
```

### Mission Reference

Used in headers and tokens:

```
AAuth-Mission: approver="https://ps.example"; s256="dBjftJeZ4CVP..."
```

The `s256` is SHA-256 of the approved mission JSON — immutable binding.

---

## The Three Governance Endpoints

### 1. Permission Endpoint

For actions **not** governed by a remote resource (tool calls, file writes):

```http
POST /permission HTTP/1.1
Host: ps.example

{
  "action": "SendEmail",
  "description": "Send the proposed itinerary to the user",
  "parameters": {"to": "user@example.com", "subject": "Trip itinerary"},
  "mission": {"approver": "...", "s256": "..."}
}
```

Response:
```json
{"permission": "granted"}
// or
{"permission": "denied", "reason": "Outside mission scope"}
```

### 2. Audit Endpoint

Log what the agent has done:

```http
POST /audit HTTP/1.1
Host: ps.example

{
  "mission": {"approver": "...", "s256": "..."},
  "action": "WebSearch",
  "description": "Searched for flights to Tokyo",
  "parameters": {"query": "flights to Tokyo May 2026"},
  "result": {"status": "completed", "summary": "Found 12 options"}
}
```

### 3. Interaction Endpoint

Reach the user through the PS:

```http
POST /interaction HTTP/1.1
Host: ps.example

{
  "type": "question",
  "question": "Do you prefer window or aisle seats?",
  "mission": {"approver": "...", "s256": "..."}
}
```

Response:
```json
{"answer": "Window please"}
```

---

## Clarification Chat

During consent, user can ask questions:

```
User: "Why do you need write access to my calendar?"
  ↓
PS returns: 202 + requirement=clarification
  ↓
Agent responds: "I need to create a meeting invite for the participants"
  ↓
User: "OK, approved"
```

---

## Mission Log

The PS maintains an ordered record of everything:

```
Mission: "Plan Japan Vacation" (s256: dBjftJeZ...)
─────────────────────────────────────────────────
[14:30:00] Mission approved
[14:30:15] Token request: flights.example (scope: search) → granted
[14:30:45] Permission: WebSearch("flights Tokyo") → granted
[14:31:00] Audit: WebSearch completed, 12 results
[14:32:00] Token request: hotels.example (scope: search) → granted
[14:35:00] Interaction: "Budget question" → user: "$5k total"
...
```

---

## What We'll Build

```
05_missions/
├── README.md
├── mission.py             # Mission creation and approval
├── permission.py          # Permission endpoint
├── audit.py               # Audit logging
├── clarification.py       # Clarification chat
└── demo.py                # Full mission lifecycle
```

By the end:
1. Create and approve missions
2. Request permissions for tool calls
3. Log actions to audit
4. Handle clarification chat

---

## Missions vs Scopes

| Scopes | Missions |
|--------|----------|
| Machine-evaluable | Natural language |
| Standing permission | Per-task context |
| "What can you access?" | "What are you trying to do?" |
| Policy engine decides | Human/AI interprets intent |

**They work together**: Scopes for deterministic policy, missions for contextual governance.

---

## Let's Start Building

We'll implement the full mission lifecycle with all three governance endpoints.

**Say "ready" to begin.**

