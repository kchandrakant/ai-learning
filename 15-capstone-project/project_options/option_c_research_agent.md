# Option C: Research Agent

Build an autonomous agent that researches complex topics using multiple sources and generates comprehensive reports.

## Project Description

Create an agent that can take a research question, break it into sub-questions, search multiple sources (web, academic papers, local documents), synthesize findings, and produce a structured report with citations.

## Core Capabilities

1. **Research Planning**: Decompose questions into sub-questions
2. **Multi-Source Search**: Web, ArXiv, Wikipedia, local docs
3. **Information Synthesis**: Combine findings coherently
4. **Report Generation**: Structured output with citations

## Core Requirements

### Must Have (80% of grade)

1. **Research Planning**
   - Question decomposition
   - Priority assignment
   - Iterative refinement

2. **Multi-Source Search**
   - At least 2 external sources (web + ArXiv)
   - Local document search
   - Result ranking

3. **Agent Loop**
   - Plan → Search → Synthesize → Evaluate
   - Iterative refinement
   - Termination conditions

4. **Information Synthesis**
   - Combine multiple sources
   - Conflict resolution
   - Gap identification

5. **Output Generation**
   - Structured reports
   - Source citations
   - Summary/key findings

### Should Have (Additional 15%)

6. **Research Memory**
   - Track explored topics
   - Avoid redundant searches
   - Build knowledge graph

7. **Quality Control**
   - Source credibility assessment
   - Fact verification
   - Confidence scoring

### Nice to Have (Additional 5%)

8. **Advanced Features**
   - Follow-up question handling
   - Report customization
   - Export formats (PDF, HTML)

## Technical Stack

```
Required:
- Python 3.10+
- FastAPI
- LangChain (for agent framework)
- Ollama with capable model (Llama 3, Mixtral)
- DuckDuckGo Search or SearXNG
- ArXiv API

Optional:
- Redis (for caching)
- Wikipedia API
- PDF generation
```

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Research Quality | 25% | Thorough and relevant findings |
| Source Integration | 20% | Multiple sources used well |
| Agent Logic | 20% | Proper planning and iteration |
| Output Quality | 15% | Clear, well-structured reports |
| Citation | 10% | Proper source attribution |
| Code Quality | 10% | Clean, documented code |

## Milestones

### Week 1-2: Search Integration
- [ ] Web search setup
- [ ] ArXiv API integration
- [ ] Result parsing

### Week 3-4: Agent Framework
- [ ] Planning logic
- [ ] Agent loop implementation
- [ ] State management

### Week 5-6: Synthesis
- [ ] Information combination
- [ ] Prompt engineering
- [ ] Conflict handling

### Week 7-8: Output
- [ ] Report generation
- [ ] Citation formatting
- [ ] API endpoints

### Week 9-10: Polish
- [ ] Caching
- [ ] Error handling
- [ ] Testing

## Getting Started

```bash
# Use the provided template
cp -r templates/research_agent my-research-agent
cd my-research-agent

# Set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Pull capable model
ollama pull llama3.2

# Run
python -m src.main
```

## Agent Architecture

```
┌─────────────────────────────────────────┐
│              Research Agent             │
├─────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │ Planner │→│ Searcher│→│Synthesizer│ │
│  └─────────┘  └─────────┘  └─────────┘ │
│       ↑                          │      │
│       └──────── Evaluator ←──────┘      │
├─────────────────────────────────────────┤
│  Sources: Web | ArXiv | Wikipedia | RAG │
└─────────────────────────────────────────┘
```

## Deliverables

1. **Source Code**: Complete agent implementation
2. **Design Document**: Agent architecture and prompts
3. **Demo**: Research complex question end-to-end
4. **Self-Evaluation**: Using the evaluation template

## Challenges to Consider

- **Search Quality**: How to get good search results?
- **Synthesis**: How to combine conflicting info?
- **Termination**: When is research "done"?
- **Cost**: API calls add up, how to optimize?

## Resources

- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- Course 08 (Agentic Patterns)
- Course 07 (Tool Integration)
