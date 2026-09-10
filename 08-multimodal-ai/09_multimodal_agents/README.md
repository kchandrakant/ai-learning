# Module 9: Multimodal Agents

Agents that see and hear.

## Overview

Multimodal agents can perceive the world visually, enabling new applications like UI automation and robotics.

## Key Topics

### Vision-Enabled Agents
```
Traditional agent: Text in, text/action out
Multimodal agent:  Text + Image in, text/action out

New capabilities:
- See screenshots, webpages, documents
- Understand visual context
- Interact with visual interfaces
```

### Screen Understanding
```
UI Agents (Computer Use):
- Take screenshot
- VLM identifies elements
- Plan clicks/typing
- Execute and observe result

Challenges: Precise localization, state tracking
```

### Document Understanding
```
Process visual documents:
- Invoices, forms, receipts
- Charts and graphs
- Scientific papers with figures

Pipeline: Image → OCR + VLM → Structured output
```

### Robotic Perception
```
Physical world interaction:
- Camera input → understand scene
- Plan manipulation
- Execute action → observe result

RT-2, PaLM-E: Language models for robotics
```

### Multimodal Memory
```
Agent needs to remember what it saw:
- Store image embeddings
- Retrieve relevant past observations
- Build visual context over time
```

## Exercises

1. Build a simple screenshot-based agent
2. Extract data from visual documents
3. Implement visual memory retrieval

## Key Insight

Vision gives agents eyes. This enables interaction with the visual world humans inhabit.
