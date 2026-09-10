# Module 2: Tool Selection

## Overview

The tool layer is where practitioners make their biggest mistakes—and where the research data is most counterintuitive. **More tools does not mean more capability.** This module teaches you how to design a tool architecture that maximizes agent performance.

---

## Learning Objectives

By the end of this module, you will be able to:

1. **Recognize the tool cliff effect** and why it occurs
2. **Apply tool rationalization** following the Vercel model
3. **Build embedding-based tool routers** for large toolboxes
4. **Implement phase-specific tool gating** without breaking KV-cache
5. **Audit your toolbox** using data-driven methods

---

## Key Concepts

### 1. The Tool Cliff Effect

**Intuition says**: More tools = more capability = better performance.

**Reality**: There's a cliff.

| Tool Count | Performance |
|------------|-------------|
| 10 tools | Perfect task performance |
| 30 tools | Noticeable degradation |
| 107 tools | **Complete failure** |

This is not a gradual decline. It's a cliff. At 107 tools, the agent functionally fails at tasks it would ace with 10 tools. **Same model.**

**Why it happens**: Attention fragmentation. When tool selection requires choosing from 107 options, the model spends so much cognitive budget on selection that it has less budget for reasoning about the task itself.

**Implication**: Your first tool audit is about **focus**, not capability.

### 2. The Vercel Case Study: 15 → 1

Vercel's engineering team had built an agent with 15+ specialized tools:
- `read_file`
- `write_file`
- `list_directory`
- `run_command`
- `check_syntax`
- `format_code`
- `run_tests`
- `install_package`
- ... and more

The hypothesis: dedicated tools for each operation would give clearer affordances.

The reality: **the opposite**.

They collapsed all 15+ tools into a **single bash tool**:

| Metric | Before (15+ tools) | After (1 bash tool) |
|--------|-------------------|---------------------|
| Task completion time | 274 seconds | **77 seconds** |
| Success rate | 80% | **100%** |
| Token usage | baseline | **37% fewer** |

**3.5x faster. 20 percentage points better. 37% cheaper. Same model.**

**Why it works**: The bash tool is a universal primitive. The model already knows bash. It knows how to compose bash commands to achieve any file or process operation. By wrapping that knowledge in 15 specialized tools, Vercel was actually **constraining** the model's ability to express what it already knew how to do.

```python
# Before: Over-specialized
tools = [
    read_file_tool,
    write_file_tool,
    list_directory_tool,
    run_command_tool,
    check_syntax_tool,
    format_code_tool,
    run_tests_tool,
    install_package_tool,
    # ... 7 more
]

# After: Rationalized
tools = [
    bash_tool  # Handles all of the above and more
]
```

### 3. When You Need More Than 15 Tools: Embedding Routers

GitHub Copilot faced a different problem. With 40 tools across diverse domains, they couldn't collapse to bash—the tools had meaningful semantic differences. But 40 tools was killing performance.

**Their solution**: Reduce static tools to 13 core + build an embedding router that dynamically adds relevant tools.

| Approach | Task Coverage | SWE-bench |
|----------|--------------|-----------|
| Static list of 13 | 69% | baseline |
| Embedding router (13 + dynamic) | **94.5%** | **+2-5 points** |

Plus: **400ms latency reduction per call**.

**The Architecture**:

```python
class EmbeddingToolRouter:
    def __init__(self, core_tools, extended_tools):
        self.core_tools = core_tools  # Always included (13 tools)
        self.extended_tools = extended_tools  # Dynamically included
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.tool_embeddings = self._embed_tools(extended_tools)

    def get_tools_for_task(self, task_description: str, top_k: int = 5):
        task_embedding = self.embedder.encode(task_description)
        similarities = cosine_similarity(
            [task_embedding],
            self.tool_embeddings
        )[0]
        
        top_indices = similarities.argsort()[-top_k:][::-1]
        dynamic_tools = [self.extended_tools[i] for i in top_indices]
        
        return self.core_tools + dynamic_tools  # Never exceeds 18 total
```

**The Pattern**:
1. Define a stable core of 10-15 universal tools
2. Build an embedding index of your extended library
3. Add at most 5 relevant tools dynamically per task
4. Stay far left of the tool cliff while maintaining full capability

### 4. Tool Naming and Prefixes

Consistent naming enables powerful routing without context changes:

```python
# Consistent prefixes enable routing
tools = [
    "browser_navigate",
    "browser_click",
    "browser_screenshot",
    "shell_execute",
    "shell_read_output",
    "file_read",
    "file_write",
    "file_delete",
    "search_web",
    "search_codebase",
]

# Now you can route by prefix using response prefilling:
# "For this task, use shell_" → constrains to shell tools
```

### 5. Logits Masking for Phase-Specific Gating

**The Problem**: You want different tools available in different phases (planning vs. execution). But changing the tool list mid-session **breaks KV-cache**.

**Wrong Approach**:
```python
# ❌ This breaks cache every time
def set_phase_tools(phase: str):
    if phase == "planning":
        tools = [planning_tools]  # Changed prefix!
    elif phase == "execution":
        tools = [execution_tools]  # Changed prefix!
```

**Right Approach**: Keep all tool definitions static. Apply **logits masking** during decoding to suppress tools you don't want.

```python
def generate_with_masked_tools(
    context,
    allowed_tool_names: list[str],
    all_tools: list[Tool]
):
    # Tool definitions stay in context — KV-cache preserved
    # Masking happens during token generation
    tool_token_ids = get_tool_call_token_ids(
        tool_names=[t.name for t in all_tools if t.name not in allowed_tool_names]
    )
    
    return model.generate(
        context,
        logits_processors=[SuppressTokensProcessor(token_ids=tool_token_ids)]
    )
```

**Alternative**: Response prefilling with tool prefixes.

```python
# Constrain to file tools by prefilling response
response = model.generate(
    context,
    prefix="I'll use file_"  # Model completes with file_read, file_write, etc.
)
```

### 6. The Tool Audit Framework

Run this audit before every significant architecture decision:

```python
def tool_audit(tools: list, agent_logs: list) -> dict:
    """Analyze tool usage patterns from agent run logs."""
    usage_counts = {t.name: 0 for t in tools}
    error_rates = {t.name: [] for t in tools}
    last_used = {t.name: None for t in tools}

    for run in agent_logs:
        for call in run.tool_calls:
            name = call.tool_name
            usage_counts[name] += 1
            error_rates[name].append(1 if call.error else 0)
            last_used[name] = run.timestamp

    return {
        "never_used": [
            t.name for t in tools 
            if usage_counts[t.name] == 0
        ],
        "high_error_rate": [
            t.name for t in tools
            if error_rates[t.name] and mean(error_rates[t.name]) > 0.3
        ],
        "stale": [
            t.name for t in tools
            if last_used[t.name] and (now - last_used[t.name]).days > 14
        ],
    }
```

**Rules**:
- **Never used**: Remove it
- **>30% error rate**: Redesign or remove
- **Not called in 14 days**: Candidate for removal

---

## Tool Selection Checklist

- [ ] Tool count audited (target under 15 active tools)
- [ ] Consolidation considered ("could this be a bash command?")
- [ ] Consistent tool naming prefix convention
- [ ] Embedding router built for large toolboxes (if >15 needed)
- [ ] Logits masking over tool removal for phase gating
- [ ] Tool error rate tracked (<30% target)
- [ ] Stale tools removed (unused >14 days)

---

## Tool Design Principles

### 1. Narrow and Explicit
```python
# ❌ Too broad
def execute_action(action: str, params: dict) -> str:
    """Execute any action with any params."""
    pass

# ✅ Narrow and explicit
def refund_customer(
    customer_id: str,
    order_id: str,
    amount: float,
    policy_reason: str,
    approval_status: str
) -> RefundResult:
    """Process a customer refund with required approval."""
    pass
```

### 2. Clear Error Messages
```python
# ❌ Unhelpful
return "Error: operation failed"

# ✅ Actionable
return {
    "error": "InvalidCustomerId",
    "message": "Customer ID 'cust_xyz' not found in database",
    "suggestion": "Verify the customer_id parameter matches an existing customer",
    "similar_ids": ["cust_xyza", "cust_xyzb"]
}
```

### 3. Schema Validation
```python
from pydantic import BaseModel, validator

class RefundRequest(BaseModel):
    customer_id: str
    amount: float
    
    @validator('customer_id')
    def validate_customer_id(cls, v):
        if not v.startswith('cust_'):
            raise ValueError('customer_id must start with "cust_"')
        return v
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('amount must be positive')
        if v > 10000:
            raise ValueError('amount exceeds maximum (requires manual approval)')
        return v
```

---

## Permission Matrices

Don't rely on telling the AI "please don't delete things"—**physically prevent** operations:

| Category | Operations | Permission |
|----------|-----------|------------|
| **Free** | Read files, run tests, query read-only APIs | Auto-execute |
| **Approval Required** | Modify configs, send external requests, delete files | Human approval |
| **Banned** | Modify `.env`, direct database operations, push to main | Blocked |

```python
class PermissionMatrix:
    FREE = {"file_read", "run_tests", "search_codebase", "list_directory"}
    APPROVAL = {"file_write", "file_delete", "api_call", "modify_config"}
    BANNED = {"db_drop", "git_push_main", "modify_env", "rm_rf"}
    
    def check(self, tool_name: str, params: dict) -> PermissionResult:
        if tool_name in self.BANNED:
            return PermissionResult.BLOCKED
        if tool_name in self.APPROVAL:
            return PermissionResult.NEEDS_APPROVAL
        return PermissionResult.ALLOWED
```

---

## Exercises

### Exercise 2.1: Tool Audit
Take an existing agent (or design one) and:
1. List all tools
2. For each tool, ask: "Could this be a bash command?"
3. Identify candidates for consolidation
4. Reduce to under 15 tools

### Exercise 2.2: Build an Embedding Router
Implement a simple embedding-based tool router:
1. Define 10 core tools
2. Define 20 extended tools
3. Build the embedding index
4. Test retrieval for different task descriptions

### Exercise 2.3: Design a Permission Matrix
For your agent task:
1. List all operations the agent might perform
2. Classify each as Free / Approval / Banned
3. Implement the permission check

### Exercise 2.4: Tool Naming Convention
Redesign these tool names with consistent prefixes:
- `readFile`, `writeFile`, `deleteFile`
- `runCommand`, `executeShell`
- `searchGoogle`, `fetchUrl`
- `createTicket`, `updateTicket`

---

## Key Takeaways

1. **Tool cliff is real**: Performance degrades non-linearly above ~15-30 tools

2. **Consolidation often wins**: Vercel's 15→1 produced 100% success rate

3. **Embedding routers extend capability**: 13 core + dynamic selection → 94.5% coverage

4. **Never change tool definitions mid-session**: Use logits masking instead

5. **Audit by data**: Remove never-used, high-error, and stale tools

---

## Next Module

Continue to **[Module 3: Constraint Management](../03_constraints/README.md)** to learn how to make agents reliable through boundaries.

---

## References

- Vercel. "We Removed 80% of Our Agent's Tools"
- GitHub. "How We're Making GitHub Copilot Smarter with Fewer Tools"
- Jenova.ai. "The MCP Tool Scalability Problem" - Tool cliff research
- Manus. "Context Engineering for AI Agents" - Logits masking
