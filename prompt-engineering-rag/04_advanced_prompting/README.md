# Step 4: Advanced Prompting Techniques

## Beyond Basic Patterns

Advanced techniques unlock sophisticated behaviors for complex applications.

## Tree-of-Thought (ToT)

CoT follows one reasoning path. ToT explores multiple paths like a tree:

```
                    Problem
                       │
         ┌─────────────┼─────────────┐
         │             │             │
      Approach A    Approach B    Approach C
         │             │             │
    ┌────┴────┐   ┌────┴────┐   ┌────┴────┐
    │         │   │         │   │         │
  Dead    Success Dead    Eval  Eval     Dead
  End        ✓   End       │     │       End
                           │     │
                        Success ...
                           ✓
```

**Implementation:**
```python
def tree_of_thought(problem, breadth=3, depth=3):
    """Explore multiple solution paths."""
    
    # Generate initial approaches
    approaches = generate_approaches(problem, n=breadth)
    
    for approach in approaches:
        # Evaluate viability
        score = evaluate_approach(approach)
        if score < threshold:
            continue  # Prune dead ends
        
        # Recursively explore
        result = explore_path(approach, depth-1)
        if result.is_solution:
            return result
    
    return best_partial_solution
```

**When to use:** Open-ended problems, planning tasks, creative problem solving.

## ReAct: Reasoning + Acting

Interleave thinking with tool use:

```
Question: What is the population of the capital of France?

Thought 1: I need to find the capital of France first.
Action 1: search("capital of France")
Observation 1: Paris is the capital of France.

Thought 2: Now I need to find Paris's population.
Action 2: search("population of Paris")
Observation 2: Paris has a population of about 2.1 million.

Thought 3: I have the answer.
Answer: The population of the capital of France (Paris) is approximately 2.1 million.
```

**Pattern:**
```
Thought → Action → Observation → Thought → Action → ... → Answer
```

## Self-Reflection

Have the model critique and improve its own output:

```python
# Step 1: Generate initial response
initial = llm(f"Answer this question: {question}")

# Step 2: Self-critique
critique = llm(f"""
Review this answer for errors or improvements:
Question: {question}
Answer: {initial}

What's wrong or could be improved?
""")

# Step 3: Refine based on critique
final = llm(f"""
Original answer: {initial}
Critique: {critique}

Provide an improved answer addressing the critique:
""")
```

## Personas and Role-Playing

Personas shape response style, expertise level, and perspective:

```python
personas = {
    "expert": "You are a senior software architect with 20 years of experience.",
    "beginner": "You are a helpful tutor explaining to a beginner.",
    "critic": "You are a skeptical code reviewer looking for issues.",
    "creative": "You are an innovative engineer who thinks outside the box.",
}

def ask_with_persona(question, persona_key):
    system = personas[persona_key]
    return llm(messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": question}
    ])
```

**Multi-persona debate:**
```python
def debate(topic):
    pro_argument = ask_with_persona(f"Argue FOR: {topic}", "advocate")
    con_argument = ask_with_persona(f"Argue AGAINST: {topic}", "critic")
    synthesis = llm(f"Synthesize these arguments:\nPro: {pro_argument}\nCon: {con_argument}")
    return synthesis
```

## Structured Output

### JSON Mode
```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    response_format={"type": "json_object"}
)
```

### Function Calling
```python
tools = [{
    "type": "function",
    "function": {
        "name": "extract_entities",
        "description": "Extract named entities from text",
        "parameters": {
            "type": "object",
            "properties": {
                "people": {"type": "array", "items": {"type": "string"}},
                "organizations": {"type": "array", "items": {"type": "string"}},
                "locations": {"type": "array", "items": {"type": "string"}}
            }
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": text}],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "extract_entities"}}
)
```

### Pydantic Validation (with Instructor)
```python
from pydantic import BaseModel
import instructor

class Person(BaseModel):
    name: str
    age: int
    occupation: str

client = instructor.patch(openai.OpenAI())
person = client.chat.completions.create(
    model="gpt-4",
    response_model=Person,
    messages=[{"role": "user", "content": "John is a 30 year old engineer."}]
)
# Returns: Person(name="John", age=30, occupation="engineer")
```

## Technique Selection Guide

| Task | Recommended Technique |
|------|----------------------|
| Complex reasoning | Chain-of-Thought |
| Open-ended exploration | Tree-of-Thought |
| Tool use / Actions | ReAct |
| Quality improvement | Self-Reflection |
| Specific expertise | Personas |
| API responses | Structured Output |

## Files

- `advanced_prompting.py` - Implementation of all techniques

## Key Takeaways

1. Tree-of-Thought explores multiple solution paths
2. ReAct interleaves reasoning with actions
3. Self-reflection improves output quality
4. Personas shape response style and expertise
5. Structured output guarantees parseable responses

## What's Next?

Step 5: **Prompt Optimization** — systematic evaluation and improvement.
