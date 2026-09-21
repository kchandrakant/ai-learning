# Code Assistant Template

A coding assistant that helps with code generation, review, debugging, and explanation. Uses RAG over code repositories and documentation to provide context-aware coding help.

## Overview

This template provides a starting point for building a code-focused AI assistant that can:

- **Generate Code**: Create functions, classes, and modules based on descriptions
- **Review Code**: Analyze code for bugs, style issues, and improvements
- **Debug Code**: Help identify and fix errors with explanations
- **Explain Code**: Provide clear explanations of complex code
- **Repository Q&A**: Answer questions about a codebase using RAG

## Architecture

```
code_assistant/
├── src/
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration management
│   ├── api/
│   │   ├── routes.py        # API endpoints
│   │   └── models.py        # Request/response schemas
│   ├── core/
│   │   ├── code_rag.py      # Code-aware RAG pipeline
│   │   ├── code_parser.py   # AST parsing and code analysis
│   │   ├── prompts.py       # Coding-specific prompts
│   │   └── embeddings.py    # Code embeddings strategy
│   └── tools/
│       ├── linter.py        # Code linting integration
│       └── formatter.py     # Code formatting
├── data/
│   └── repos/               # Indexed code repositories
├── tests/
│   └── test_code_rag.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Key Features

### 1. Code-Aware Chunking

Unlike generic text chunking, code requires special handling:

```python
# Traditional chunking breaks code randomly
# Code-aware chunking respects:
# - Function boundaries
# - Class definitions
# - Import blocks
# - Logical code blocks
```

### 2. Semantic Code Search

Uses code-specific embeddings that understand:
- Variable naming patterns
- Function signatures
- Code structure and flow

### 3. Multi-Language Support

Configure for your target languages:
- Python, JavaScript/TypeScript, Java, Go, Rust
- Language-specific parsing and analysis

## Getting Started

### 1. Prerequisites

- Docker and Docker Compose
- Ollama with CodeLlama or similar code model
- 8GB+ RAM recommended

### 2. Quick Start

```bash
# Clone and navigate to template
cd templates/code_assistant

# Copy environment file
cp .env.example .env

# Start services
docker-compose up -d

# Pull code model
docker exec -it code-assistant-ollama ollama pull codellama

# Test the API
curl http://localhost:8001/health
```

### 3. Index a Repository

```bash
# Index a local repository
curl -X POST http://localhost:8001/repos/index \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/repo", "name": "my-project"}'

# Or index from GitHub
curl -X POST http://localhost:8001/repos/index-github \
  -H "Content-Type: application/json" \
  -d '{"url": "https://github.com/user/repo", "branch": "main"}'
```

### 4. Ask Questions

```bash
# Generate code
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a function to validate email addresses",
    "language": "python",
    "context_repo": "my-project"
  }'

# Review code
curl -X POST http://localhost:8001/review \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def add(a, b): return a+b",
    "language": "python"
  }'

# Debug code
curl -X POST http://localhost:8001/debug \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def factorial(n): return n * factorial(n-1)",
    "error": "RecursionError: maximum recursion depth exceeded"
  }'
```

## Configuration

Key settings in `.env`:

```bash
# LLM Model (code-optimized)
OLLAMA_MODEL=codellama:13b

# Code parsing
SUPPORTED_LANGUAGES=python,javascript,typescript
MAX_FILE_SIZE_KB=500

# Chunking
CODE_CHUNK_STRATEGY=ast  # ast, function, or fixed
MAX_CHUNK_TOKENS=1000
```

## Customization Points

### 1. Language Support

Add new languages in `core/code_parser.py`:

```python
LANGUAGE_CONFIGS = {
    "python": {
        "extensions": [".py"],
        "comment_prefix": "#",
        "parser": "tree_sitter_python"
    },
    # Add your language here
}
```

### 2. Prompt Templates

Customize prompts in `core/prompts.py`:

```python
CODE_REVIEW_PROMPT = """
Review the following {language} code.
Focus on: {review_focus}

Code:
{code}

Provide specific, actionable feedback.
"""
```

### 3. Embedding Strategy

Choose how code is embedded in `core/embeddings.py`:

```python
# Options:
# 1. Generic text embeddings (default)
# 2. Code-specific embeddings (CodeBERT, StarEncoder)
# 3. Hybrid approach
```

## Implementation Tasks

When using this template, implement:

- [ ] Code-aware chunking logic
- [ ] AST parsing for target languages
- [ ] Repository indexing pipeline
- [ ] Code generation with proper formatting
- [ ] Review feedback formatting
- [ ] Multi-file context handling

## Testing Your Implementation

```bash
# Run unit tests
pytest tests/ -v

# Test with sample repository
python scripts/test_with_sample_repo.py

# Benchmark retrieval quality
python scripts/benchmark_retrieval.py
```

## Resources

- [Tree-sitter](https://tree-sitter.github.io/) - Code parsing
- [CodeBERT](https://github.com/microsoft/CodeBERT) - Code embeddings
- [CodeLlama](https://github.com/facebookresearch/codellama) - Code LLM
- [StarCoder](https://huggingface.co/bigcode/starcoder) - Open code model

## Evaluation Criteria

Your code assistant will be evaluated on:

1. **Retrieval Quality**: Does it find relevant code context?
2. **Generation Accuracy**: Is generated code correct and runnable?
3. **Review Helpfulness**: Are code reviews actionable?
4. **Response Speed**: Acceptable latency for interactive use?
5. **Multi-Language Support**: Works across configured languages?

---

**Next Steps**: 
1. Review the starter code in `src/`
2. Implement the TODOs
3. Test with a sample repository
4. Iterate based on evaluation
