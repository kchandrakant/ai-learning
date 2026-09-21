# Domain-Specific Assistant - Starter Template

This template provides a foundation for building a domain-specific AI assistant with RAG capabilities.

## Quick Start

```bash
# 1. Copy this template to your project
cp -r templates/domain_assistant my-assistant
cd my-assistant

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env
# Edit .env with your settings

# 5. Start Ollama (if using local models)
ollama serve
ollama pull llama3.2

# 6. Add your documents
# Place documents in data/documents/

# 7. Index documents
python scripts/index_documents.py

# 8. Run the API
python -m uvicorn src.main:app --reload

# 9. Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

## Project Structure

```
domain_assistant/
├── README.md
├── requirements.txt
├── .env.example
├── pyproject.toml
│
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py        # API endpoints
│   │   └── models.py        # Pydantic models
│   ├── core/
│   │   ├── __init__.py
│   │   ├── llm.py           # LLM client
│   │   ├── rag.py           # RAG pipeline
│   │   └── prompts.py       # Prompt templates
│   └── utils/
│       ├── __init__.py
│       └── logging.py
│
├── data/
│   ├── documents/           # Source documents
│   └── vectorstore/         # Vector database
│
├── scripts/
│   ├── index_documents.py   # Document indexing
│   └── evaluate.py          # Evaluation script
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_rag.py
│
├── docs/
│   ├── requirements.md
│   └── design.md
│
└── deployment/
    ├── Dockerfile
    └── docker-compose.yml
```

## Configuration

Edit `.env` or `src/config.py`:

```python
# LLM Settings
LLM_PROVIDER = "ollama"  # or "openai"
LLM_MODEL = "llama3.2"
LLM_BASE_URL = "http://localhost:11434"

# RAG Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# API Settings
API_HOST = "0.0.0.0"
API_PORT = 8000
```

## Customization

### 1. Add Your Domain Documents

Place documents in `data/documents/`:
- PDF files
- Text files
- Markdown files
- Word documents

### 2. Customize Prompts

Edit `src/core/prompts.py`:

```python
SYSTEM_PROMPT = """You are a helpful assistant specializing in [YOUR DOMAIN].

Your role is to:
- Answer questions accurately based on the provided context
- Cite sources when possible
- Admit when you don't know something

Always be [professional/friendly/formal] in tone.
"""
```

### 3. Add Domain-Specific Tools (Optional)

Edit `src/core/tools.py` to add domain-specific capabilities.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/chat` | POST | Send a chat message |
| `/documents` | GET | List indexed documents |
| `/documents` | POST | Add a new document |
| `/documents/{id}` | DELETE | Remove a document |

## Next Steps

1. [ ] Define your requirements (docs/requirements.md)
2. [ ] Complete the design document (docs/design.md)
3. [ ] Add your domain documents
4. [ ] Customize prompts for your domain
5. [ ] Implement additional features
6. [ ] Add tests
7. [ ] Evaluate quality
8. [ ] Deploy
