# Step 1: Prompting Basics

## The Foundation

A prompt is your primary interface with a language model. Understanding prompt structure is essential for everything that follows.

## Anatomy of a Prompt

```
┌─────────────────────────────────────────┐
│           SYSTEM MESSAGE                │
│  - Role definition                      │
│  - Behavioral constraints               │
│  - Output format requirements           │
├─────────────────────────────────────────┤
│           USER MESSAGE                  │
│  - Context (background information)     │
│  - Instruction (what to do)             │
│  - Input (the actual data/question)     │
│  - Output format (optional)             │
└─────────────────────────────────────────┘
```

## Zero-Shot Prompting

Zero-shot = no examples, just instructions.

```python
# Simple zero-shot
prompt = "Translate the following English text to French: 'Hello, how are you?'"

# Structured zero-shot
prompt = """
Task: Sentiment Analysis
Input: "This product exceeded my expectations!"
Output format: Return only "positive", "negative", or "neutral"
"""
```

**When to use:** Simple tasks, well-defined outputs, when examples aren't necessary.

## System vs User Messages

```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that responds in JSON format."
    },
    {
        "role": "user", 
        "content": "List three programming languages and their main use cases."
    }
]
```

**System message:** Sets behavior, persona, constraints (persists across conversation)
**User message:** The actual request or input

## Temperature and Sampling

```
Temperature = 0.0  → Deterministic, focused (best for factual tasks)
Temperature = 0.7  → Balanced creativity
Temperature = 1.0+ → More random, creative (best for brainstorming)
```

**Rule of thumb:**
- Factual Q&A, code generation: temp = 0.0 - 0.3
- Creative writing, brainstorming: temp = 0.7 - 1.0

## The CRAFT Framework

A mental model for effective prompts:

| Letter | Meaning | Example |
|--------|---------|---------|
| **C** | Context | "You are reviewing customer support tickets..." |
| **R** | Role | "Act as a senior software engineer..." |
| **A** | Action | "Analyze the following code for bugs..." |
| **F** | Format | "Return your answer as a JSON object with..." |
| **T** | Tone | "Be concise and technical..." |

## Common Pitfalls

1. **Ambiguous instructions** → Be specific
2. **Missing context** → Provide background
3. **No output format** → Specify expected structure
4. **Too many tasks** → One task per prompt

## Files

- `prompting_basics.py` - Implementation with examples

## Key Takeaways

1. Structure matters: Context + Instruction + Format
2. System messages set persistent behavior
3. Temperature controls randomness
4. Be specific, not vague
5. One task per prompt (usually)

## What's Next?

Step 2: **Few-Shot Learning** — using examples to "program" model behavior.
