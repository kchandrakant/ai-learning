# Research Agent Template

An autonomous research agent that searches, synthesizes, and reports on complex topics. Uses iterative planning, multi-source retrieval, and structured output generation.

## Overview

This template implements a research agent that can:

- **Plan Research**: Break down complex questions into sub-questions
- **Gather Sources**: Search multiple sources (web, papers, local docs)
- **Synthesize Information**: Combine findings into coherent answers
- **Generate Reports**: Create structured research reports
- **Cite Sources**: Provide proper attribution and references

## Architecture

```
research_agent/
├── src/
│   ├── main.py              # Application entry point
│   ├── config.py            # Configuration
│   ├── api/
│   │   ├── routes.py        # API endpoints
│   │   └── models.py        # Request/response schemas
│   ├── agent/
│   │   ├── planner.py       # Research planning
│   │   ├── searcher.py      # Multi-source search
│   │   ├── synthesizer.py   # Information synthesis
│   │   └── reporter.py      # Report generation
│   ├── sources/
│   │   ├── web.py           # Web search integration
│   │   ├── arxiv.py         # ArXiv paper search
│   │   └── local.py         # Local document search
│   └── core/
│       ├── prompts.py       # Agent prompts
│       └── memory.py        # Agent memory/context
├── data/
│   ├── cache/               # Search result cache
│   └── reports/             # Generated reports
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Key Features

### 1. Iterative Research Planning

The agent uses a ReAct-style loop:

```
1. Plan: Break question into sub-questions
2. Search: Find relevant sources for each sub-question
3. Reflect: Evaluate if enough information gathered
4. Synthesize: Combine findings
5. Repeat if needed
```

### 2. Multi-Source Integration

```python
SOURCES = [
    "web",      # Web search via DuckDuckGo/SearXNG
    "arxiv",    # Academic papers
    "wikipedia",# Encyclopedic content
    "local",    # Your uploaded documents
]
```

### 3. Structured Report Generation

Output formats:
- **Summary**: Brief answer with key points
- **Report**: Full research report with sections
- **Outline**: Hierarchical breakdown
- **Citations**: Source bibliography

## Getting Started

### 1. Prerequisites

- Docker and Docker Compose
- Ollama with a capable model (Llama 3, Mixtral)
- API keys for search services (optional)

### 2. Quick Start

```bash
# Navigate to template
cd templates/research_agent

# Copy environment file
cp .env.example .env

# Start services
docker-compose up -d

# Test the agent
curl http://localhost:8002/health
```

### 3. Run a Research Query

```bash
# Simple query
curl -X POST http://localhost:8002/research \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main approaches to reducing LLM hallucinations?",
    "depth": "detailed",
    "sources": ["web", "arxiv"]
  }'

# With report generation
curl -X POST http://localhost:8002/research \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Compare transformer architectures for vision tasks",
    "output_format": "report",
    "max_iterations": 3
  }'
```

## Agent Loop

```python
class ResearchAgent:
    async def research(self, question: str) -> ResearchResult:
        # Step 1: Plan
        plan = await self.planner.create_plan(question)
        
        # Step 2: Iterative search and synthesis
        for iteration in range(self.max_iterations):
            # Search for each sub-question
            for sub_q in plan.sub_questions:
                results = await self.searcher.search(sub_q)
                self.memory.add_findings(sub_q, results)
            
            # Synthesize current findings
            synthesis = await self.synthesizer.synthesize(
                question, 
                self.memory.get_all_findings()
            )
            
            # Check if we have enough
            if await self.evaluator.is_sufficient(synthesis, question):
                break
            
            # Refine plan for next iteration
            plan = await self.planner.refine(plan, synthesis.gaps)
        
        # Step 3: Generate final output
        return await self.reporter.generate(question, synthesis)
```

## Configuration

Key settings in `.env`:

```bash
# Agent behavior
MAX_ITERATIONS=3
MAX_SOURCES_PER_QUERY=10
SEARCH_TIMEOUT_SECONDS=30

# Sources
ENABLE_WEB_SEARCH=true
ENABLE_ARXIV_SEARCH=true
WEB_SEARCH_API=duckduckgo  # or searxng, bing

# Output
DEFAULT_OUTPUT_FORMAT=report
MAX_REPORT_LENGTH_WORDS=2000
```

## Customization Points

### 1. Add New Sources

Implement the source interface in `sources/`:

```python
class CustomSource:
    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        # Implement your search
        pass
    
    async def fetch_content(self, url: str) -> str:
        # Fetch full content if needed
        pass
```

### 2. Customize Planning Prompts

Modify prompts in `core/prompts.py`:

```python
PLANNING_PROMPT = """
Given the research question: {question}

Create a research plan with:
1. Key sub-questions to answer
2. Best sources for each sub-question
3. Information priorities

Consider the user's expertise level: {expertise_level}
"""
```

### 3. Report Templates

Add templates in `templates/`:

```python
REPORT_TEMPLATE = """
# {title}

## Executive Summary
{summary}

## Key Findings
{findings}

## Methodology
{methodology}

## References
{references}
"""
```

## Implementation Tasks

When using this template, implement:

- [ ] Planning logic (decompose questions)
- [ ] Web search integration
- [ ] ArXiv API integration
- [ ] Synthesis prompts
- [ ] Report generation
- [ ] Source caching
- [ ] Citation formatting

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/research` | POST | Start research on a question |
| `/research/{id}` | GET | Get research status/results |
| `/research/{id}/report` | GET | Get formatted report |
| `/sources` | GET | List available sources |
| `/cache/clear` | POST | Clear search cache |

## Testing

```bash
# Unit tests
pytest tests/ -v

# Test with sample questions
python scripts/test_questions.py

# Benchmark research quality
python scripts/benchmark_research.py
```

## Example Use Cases

1. **Literature Review**: "What are recent advances in protein structure prediction?"
2. **Comparison**: "Compare RAG vs fine-tuning for domain adaptation"
3. **Trend Analysis**: "What are emerging trends in AI safety research?"
4. **Technical Deep-Dive**: "How does flash attention work?"

## Evaluation Criteria

Your research agent will be evaluated on:

1. **Coverage**: Does it find relevant information?
2. **Accuracy**: Is the synthesized information correct?
3. **Coherence**: Is the output well-organized?
4. **Citation Quality**: Are sources properly attributed?
5. **Efficiency**: Reasonable number of API calls?

---

**Next Steps**:
1. Review the starter code in `src/`
2. Implement the search integrations
3. Build the planning and synthesis logic
4. Test with diverse research questions
5. Iterate based on evaluation
