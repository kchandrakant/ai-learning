# Step 8: Agent Communication

## The Challenge

Agents need to share information without:
- Losing critical context
- Overwhelming each other
- Creating circular dependencies

## Communication Patterns

### 1. Message Passing

Direct messages between agents:

```python
from dataclasses import dataclass
from typing import Any
from datetime import datetime

@dataclass
class AgentMessage:
    sender: str
    recipient: str
    content: str
    message_type: str  # "request", "response", "notification"
    metadata: dict
    timestamp: datetime = None
    
    def __post_init__(self):
        self.timestamp = self.timestamp or datetime.now()

class MessageBus:
    def __init__(self):
        self.queues: dict[str, list[AgentMessage]] = {}
        self.handlers: dict[str, Callable] = {}
    
    def register(self, agent_id: str, handler: Callable):
        self.queues[agent_id] = []
        self.handlers[agent_id] = handler
    
    def send(self, message: AgentMessage):
        if message.recipient in self.queues:
            self.queues[message.recipient].append(message)
            # Optionally trigger handler
            self.handlers[message.recipient](message)
    
    def receive(self, agent_id: str) -> list[AgentMessage]:
        messages = self.queues.get(agent_id, [])
        self.queues[agent_id] = []
        return messages
```

### 2. Shared State

Agents read/write to shared memory:

```python
from threading import Lock
from typing import Any

class SharedState:
    def __init__(self):
        self._state: dict[str, Any] = {}
        self._lock = Lock()
        self._history: list[dict] = []
    
    def read(self, key: str) -> Any:
        with self._lock:
            return self._state.get(key)
    
    def write(self, key: str, value: Any, agent_id: str):
        with self._lock:
            old_value = self._state.get(key)
            self._state[key] = value
            self._history.append({
                "agent": agent_id,
                "key": key,
                "old": old_value,
                "new": value,
                "timestamp": datetime.now()
            })
    
    def get_updates_since(self, timestamp: datetime) -> list[dict]:
        return [h for h in self._history if h["timestamp"] > timestamp]
```

### 3. Blackboard Pattern

Shared workspace where agents post and read:

```python
class Blackboard:
    def __init__(self):
        self.sections: dict[str, list[dict]] = {
            "goals": [],
            "facts": [],
            "hypotheses": [],
            "actions": [],
            "results": []
        }
    
    def post(self, section: str, content: dict, agent_id: str):
        entry = {
            "content": content,
            "author": agent_id,
            "timestamp": datetime.now()
        }
        self.sections[section].append(entry)
    
    def read_section(self, section: str) -> list[dict]:
        return self.sections.get(section, [])
    
    def find(self, section: str, criteria: Callable) -> list[dict]:
        return [e for e in self.sections[section] if criteria(e)]
```

## Context Compression

Agents can't pass entire histories. Summarize:

```python
class ContextCompressor:
    def __init__(self, llm, max_tokens: int = 1000):
        self.llm = llm
        self.max_tokens = max_tokens
    
    def compress_for_handoff(self, 
                              task: str, 
                              history: list[dict],
                              next_agent_role: str) -> str:
        """Compress context for another agent."""
        
        prompt = f"""
        Summarize this work for handoff to a {next_agent_role}.
        
        Original task: {task}
        
        Work done:
        {self._format_history(history)}
        
        Create a concise summary including:
        1. What was accomplished
        2. Key findings/decisions
        3. What remains to be done
        4. Any important context
        
        Keep under {self.max_tokens} tokens.
        """
        
        return self.llm(prompt)
    
    def _format_history(self, history: list[dict]) -> str:
        lines = []
        for entry in history:
            lines.append(f"- {entry['action']}: {entry['result'][:200]}...")
        return "\n".join(lines)
```

## Handoff Protocol

Formal handoff between agents:

```python
@dataclass
class Handoff:
    from_agent: str
    to_agent: str
    task_summary: str
    completed_work: str
    remaining_work: str
    key_context: dict
    artifacts: list[str]  # File paths, data references

class HandoffProtocol:
    def __init__(self, compressor: ContextCompressor):
        self.compressor = compressor
    
    def create_handoff(self,
                       from_agent: Agent,
                       to_agent: Agent,
                       task: str,
                       history: list) -> Handoff:
        
        summary = self.compressor.compress_for_handoff(
            task, history, to_agent.role
        )
        
        return Handoff(
            from_agent=from_agent.name,
            to_agent=to_agent.name,
            task_summary=task,
            completed_work=self._extract_completed(history),
            remaining_work=self._extract_remaining(task, history),
            key_context=self._extract_key_context(history),
            artifacts=self._collect_artifacts(history)
        )
    
    def accept_handoff(self, agent: Agent, handoff: Handoff) -> str:
        """Generate agent's starting context from handoff."""
        return f"""
        You are continuing work on: {handoff.task_summary}
        
        Previous agent ({handoff.from_agent}) completed:
        {handoff.completed_work}
        
        Your task:
        {handoff.remaining_work}
        
        Key context:
        {json.dumps(handoff.key_context, indent=2)}
        
        Available artifacts: {handoff.artifacts}
        """
```

## Communication Best Practices

### 1. Structured Messages

```python
# Good: Structured
message = AgentMessage(
    sender="researcher",
    recipient="writer",
    content=json.dumps({
        "findings": [...],
        "sources": [...],
        "confidence": 0.85
    }),
    message_type="research_complete"
)

# Bad: Unstructured
message = "I found some stuff about AI agents. Here it is: ..."
```

### 2. Explicit Dependencies

```python
class Task:
    def __init__(self, id: str, description: str):
        self.id = id
        self.description = description
        self.dependencies: list[str] = []  # Task IDs
        self.outputs: list[str] = []       # What this produces
        self.inputs: list[str] = []        # What this needs
```

### 3. Acknowledgment Protocol

```python
class ReliableMessaging:
    def send_with_ack(self, message: AgentMessage, timeout: int = 30):
        self.bus.send(message)
        
        # Wait for acknowledgment
        start = time.time()
        while time.time() - start < timeout:
            acks = self.bus.receive(message.sender)
            for ack in acks:
                if ack.metadata.get("ack_for") == message.id:
                    return True
            time.sleep(0.1)
        
        raise TimeoutError(f"No ack from {message.recipient}")
```

## Files

- `agent_communication.py` - All communication patterns

## Key Takeaways

1. Message passing for direct communication
2. Shared state for collaborative work
3. Blackboard for posting discoveries
4. Always compress context for handoffs
5. Use structured messages, not free text

## What's Next?

Step 9: **Workflow Orchestration** — building reliable multi-step workflows.
