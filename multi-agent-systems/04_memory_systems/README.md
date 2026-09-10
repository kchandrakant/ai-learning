# Step 4: Memory Systems

## Why Memory Matters

Without memory, agents forget everything between interactions. Memory enables:
- Learning from past interactions
- Maintaining context across sessions
- Building up knowledge over time

## Types of Memory

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Memory Types                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │  Short-term     │  │  Working        │                   │
│  │  (Conversation) │  │  (Scratchpad)   │                   │
│  │                 │  │                 │                   │
│  │  Recent msgs    │  │  Current task   │                   │
│  │  Context window │  │  Intermediate   │                   │
│  │  ~4-128K tokens │  │  results        │                   │
│  └─────────────────┘  └─────────────────┘                   │
│                                                              │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │  Long-term      │  │  Episodic       │                   │
│  │  (Knowledge)    │  │  (Experiences)  │                   │
│  │                 │  │                 │                   │
│  │  Facts, prefs   │  │  Past sessions  │                   │
│  │  Vector store   │  │  What worked    │                   │
│  │  Persistent     │  │  Learnings      │                   │
│  └─────────────────┘  └─────────────────┘                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Short-term Memory

The conversation buffer — what's in the current context window.

```python
class ConversationMemory:
    def __init__(self, max_messages: int = 50):
        self.messages: list[dict] = []
        self.max_messages = max_messages
    
    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        
        # Trim if too long
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_context(self) -> list[dict]:
        return self.messages.copy()
    
    def clear(self):
        self.messages = []
```

### Smarter Truncation

Don't just drop old messages — summarize them:

```python
class SummarizingMemory:
    def __init__(self, llm, max_tokens: int = 4000):
        self.llm = llm
        self.max_tokens = max_tokens
        self.messages = []
        self.summary = ""
    
    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        
        if self._count_tokens() > self.max_tokens:
            self._compress()
    
    def _compress(self):
        # Summarize older messages
        old_messages = self.messages[:-5]  # Keep recent 5
        recent = self.messages[-5:]
        
        summary_prompt = f"""
        Summarize this conversation, preserving key facts and decisions:
        {self._format_messages(old_messages)}
        """
        
        new_summary = self.llm(summary_prompt)
        self.summary = f"{self.summary}\n{new_summary}"
        self.messages = recent
    
    def get_context(self) -> list[dict]:
        context = []
        if self.summary:
            context.append({
                "role": "system",
                "content": f"Previous conversation summary:\n{self.summary}"
            })
        context.extend(self.messages)
        return context
```

## Working Memory (Scratchpad)

Store intermediate results during a task:

```python
class Scratchpad:
    def __init__(self):
        self.notes: dict[str, str] = {}
        self.task_state: dict = {}
    
    def note(self, key: str, value: str):
        """Store a note for later reference."""
        self.notes[key] = value
    
    def recall(self, key: str) -> str:
        """Retrieve a stored note."""
        return self.notes.get(key, "Not found")
    
    def set_state(self, key: str, value):
        """Track task progress."""
        self.task_state[key] = value
    
    def get_state(self) -> dict:
        return self.task_state.copy()
    
    def to_context(self) -> str:
        """Format for inclusion in prompt."""
        if not self.notes:
            return ""
        
        lines = ["Working notes:"]
        for k, v in self.notes.items():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)
```

## Long-term Memory

Persistent storage for facts and preferences:

```python
import chromadb
from datetime import datetime

class LongTermMemory:
    def __init__(self, collection_name: str = "agent_memory"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)
    
    def remember(self, content: str, metadata: dict = None):
        """Store a memory."""
        memory_id = f"mem_{datetime.now().timestamp()}"
        
        self.collection.add(
            documents=[content],
            metadatas=[metadata or {}],
            ids=[memory_id]
        )
    
    def recall(self, query: str, k: int = 5) -> list[str]:
        """Retrieve relevant memories."""
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        return results["documents"][0]
    
    def forget(self, memory_id: str):
        """Remove a memory."""
        self.collection.delete(ids=[memory_id])
```

## Episodic Memory

Learn from past experiences:

```python
class EpisodicMemory:
    def __init__(self, vector_store):
        self.store = vector_store
    
    def record_episode(self, task: str, actions: list, outcome: str, success: bool):
        """Record a completed task episode."""
        episode = {
            "task": task,
            "actions": actions,
            "outcome": outcome,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store with task as searchable text
        self.store.add(
            documents=[f"Task: {task}\nOutcome: {outcome}"],
            metadatas=[episode]
        )
    
    def recall_similar(self, current_task: str, k: int = 3) -> list[dict]:
        """Find similar past tasks."""
        results = self.store.query(
            query_texts=[current_task],
            n_results=k
        )
        
        return [
            meta for meta in results["metadatas"][0]
            if meta.get("success")  # Prefer successful episodes
        ]
```

## Unified Memory System

```python
class AgentMemory:
    def __init__(self, llm):
        self.short_term = SummarizingMemory(llm)
        self.working = Scratchpad()
        self.long_term = LongTermMemory()
        self.episodic = EpisodicMemory(vector_store)
    
    def build_context(self, query: str) -> str:
        """Build full context from all memory types."""
        context_parts = []
        
        # Relevant long-term memories
        memories = self.long_term.recall(query)
        if memories:
            context_parts.append("Relevant knowledge:\n" + "\n".join(memories))
        
        # Similar past experiences
        episodes = self.episodic.recall_similar(query)
        if episodes:
            context_parts.append("Similar past tasks:\n" + self._format_episodes(episodes))
        
        # Working notes
        working_context = self.working.to_context()
        if working_context:
            context_parts.append(working_context)
        
        return "\n\n".join(context_parts)
    
    def _format_episodes(self, episodes: list[dict]) -> str:
        lines = []
        for ep in episodes:
            lines.append(f"- Task: {ep['task']} → {ep['outcome']}")
        return "\n".join(lines)
```

## Memory in Prompts

```python
def build_prompt(query: str, memory: AgentMemory) -> str:
    context = memory.build_context(query)
    
    return f"""You are a helpful assistant with access to your memories.

{context}

Current conversation:
{memory.short_term.get_context()}

User: {query}
Assistant:"""
```

## Files

- `memory_systems.py` - All memory implementations

## Key Takeaways

1. Four types: short-term, working, long-term, episodic
2. Short-term = conversation context (summarize when full)
3. Working = scratchpad for current task
4. Long-term = persistent facts and preferences
5. Episodic = learning from past experiences

## What's Next?

Step 5: **LangGraph Deep Dive** — building stateful agents with explicit control flow.
