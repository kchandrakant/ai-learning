# Agent System Design: Key References

A curated collection of papers on agent runtime architecture, evaluation, and production systems.

---

## 📚 How to Use This Document

- **📖 Essential**: Core papers everyone should read
- **🔧 Hands-on**: Resources with practical implementations
- **🔬 Frontier**: Cutting-edge research

---

## Agent Architecture Foundations

### ReAct: Synergizing Reasoning and Acting in Language Models
**Yao et al., ICLR 2023**

The foundational pattern for tool-using agents.

- **Key innovations**: Thought-action-observation loops
- **Impact**: Standard agent architecture
- **Link**: [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
- **Status**: 📖 Essential

---

### Generative Agents: Interactive Simulacra of Human Behavior
**Park et al., UIST 2023**

Comprehensive agent architecture with memory and planning.

- **Key innovations**: Memory stream, reflection, planning modules
- **Impact**: Reference architecture for complex agents
- **Link**: [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)
- **Status**: 📖 Essential

---

### Cognitive Architectures for Language Agents (CoALA)
**Sumers et al., 2023**

Framework for understanding agent architectures.

- **Key value**: Taxonomy and design space for agents
- **Link**: [arXiv:2309.02427](https://arxiv.org/abs/2309.02427)
- **Status**: 📖 Essential

---

## Tool Use & Function Calling

### Toolformer: Language Models Can Teach Themselves to Use Tools
**Schick et al., NeurIPS 2023**

Self-supervised tool learning.

- **Key innovations**: Learning when and how to use tools
- **Link**: [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)
- **Status**: 📖 Essential

---

### Gorilla: Large Language Model Connected with Massive APIs
**Patil et al., 2023**

LLM trained for API calling.

- **Key innovations**: API documentation retrieval, hallucination reduction
- **Link**: [arXiv:2305.15334](https://arxiv.org/abs/2305.15334)
- **Status**: 🔧 Hands-on

---

### ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs
**Qin et al., 2023**

Large-scale tool use benchmark and training.

- **Key innovations**: ToolBench dataset, DFSDT algorithm
- **Link**: [arXiv:2307.16789](https://arxiv.org/abs/2307.16789)
- **Status**: 📝 Reference

---

## Agent Memory Systems

### MemGPT: Towards LLMs as Operating Systems
**Packer et al., 2023**

Virtual memory management for agents.

- **Key innovations**: Hierarchical memory, memory paging
- **Impact**: Long-term agent memory
- **Link**: [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)
- **Status**: 🔧 Hands-on

---

### Reflexion: Language Agents with Verbal Reinforcement Learning
**Shinn et al., NeurIPS 2023**

Learning from verbal self-reflection.

- **Key innovations**: Episodic memory, verbal reinforcement
- **Link**: [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)
- **Status**: 🔧 Hands-on

---

## Agent Evaluation

### AgentBench: Evaluating LLMs as Agents
**Liu et al., ICLR 2024**

Comprehensive benchmark for LLM agents.

- **Key value**: Standardized evaluation across 8 environments
- **Link**: [arXiv:2308.03688](https://arxiv.org/abs/2308.03688)
- **Status**: 📖 Essential

---

### WebArena: A Realistic Web Environment for Building Autonomous Agents
**Zhou et al., ICLR 2024**

Benchmark for web-based agents.

- **Key innovations**: Realistic web tasks, functional evaluation
- **Link**: [arXiv:2307.13854](https://arxiv.org/abs/2307.13854)
- **Status**: 🔧 Hands-on

---

### SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
**Jimenez et al., 2024**

Benchmark for software engineering agents.

- **Key innovations**: Real GitHub issues, executable tests
- **Impact**: Standard for coding agents
- **Link**: [arXiv:2310.06770](https://arxiv.org/abs/2310.06770)
- **Status**: 📖 Essential

---

### GAIA: A Benchmark for General AI Assistants
**Mialon et al., 2023**

Benchmark for general-purpose AI assistants.

- **Key innovations**: Multi-step reasoning, tool use required
- **Link**: [arXiv:2311.12983](https://arxiv.org/abs/2311.12983)
- **Status**: 📖 Essential

---

## Production Agent Systems

### Devin: AI Software Engineer
**Cognition AI, 2024**

Production autonomous coding agent.

- **Key value**: Real-world agent deployment lessons
- **Link**: [cognition-labs.com](https://www.cognition-labs.com/)
- **Status**: 📝 Reference

---

### The Landscape of Emerging AI Agent Architectures
**LangChain, 2024**

Survey of production agent patterns.

- **Key value**: Practical architecture patterns
- **Link**: [LangChain Blog](https://blog.langchain.dev/the-landscape-of-emerging-ai-agent-architectures/)
- **Status**: 🔧 Hands-on

---

## Safety & Reliability

### Practices for Governing Agentic AI Systems
**OpenAI, 2024**

Guidelines for safe agent deployment.

- **Key topics**: Human oversight, monitoring, access control
- **Link**: [OpenAI](https://openai.com/research/practices-for-governing-agentic-ai-systems)
- **Status**: 📖 Essential

---

### Constitutional AI: Harmlessness from AI Feedback
**Bai et al., Anthropic 2022**

Training safe AI agents.

- **Key innovations**: Self-critique, constitutional principles
- **Link**: [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
- **Status**: 📝 Reference

---

## Online Resources

| Resource | Description | Link |
|----------|-------------|------|
| LangGraph | Agent orchestration | [langchain.com/langgraph](https://www.langchain.com/langgraph) |
| OpenAI Assistants | Production agent API | [OpenAI Docs](https://platform.openai.com/docs/assistants) |
| Anthropic Claude Tools | Tool use patterns | [Anthropic Docs](https://docs.anthropic.com/claude/docs/tool-use) |
| Agent Protocol | Agent API standard | [agentprotocol.ai](https://agentprotocol.ai/) |

---

## Implementation Resources

| Resource | Description | Link |
|----------|-------------|------|
| LangSmith | Agent observability | [smith.langchain.com](https://smith.langchain.com/) |
| Weights & Biases Prompts | Trace debugging | [wandb.ai](https://wandb.ai/) |
| Phoenix | LLM observability | [GitHub](https://github.com/Arize-ai/phoenix) |

---

*Last updated: September 2026*
