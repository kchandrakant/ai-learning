# Option A: Domain-Specific Assistant

Build a RAG-powered assistant that answers questions about a specific domain using uploaded documents.

## Project Description

Create an intelligent assistant that can ingest documents (PDFs, markdown, text files) and answer questions using RAG. The assistant should provide accurate, sourced answers from the knowledge base.

## Example Domains

Choose a domain that interests you:

- **Technical Documentation**: Product manuals, API docs, guides
- **Legal/Compliance**: Contracts, policies, regulations
- **Medical/Healthcare**: Clinical guidelines, drug information
- **Educational**: Course materials, textbooks, research
- **Customer Support**: FAQs, troubleshooting guides
- **Internal Knowledge Base**: Company wiki, processes

## Core Requirements

### Must Have (80% of grade)

1. **Document Ingestion**
   - Support at least 2 file formats (e.g., PDF, Markdown)
   - Proper text extraction and cleaning
   - Chunking with configurable parameters

2. **Vector Storage**
   - Store embeddings in Chroma or FAISS
   - Efficient similarity search
   - Metadata filtering

3. **RAG Pipeline**
   - Context-aware retrieval
   - Source attribution in responses
   - Handle "I don't know" cases

4. **API Interface**
   - REST API with FastAPI
   - Chat endpoint with session support
   - Document management endpoints

5. **Local LLM Integration**
   - Ollama with configurable model
   - Proper prompt engineering
   - Temperature and token control

### Should Have (Additional 15%)

6. **Conversation Memory**
   - Multi-turn conversations
   - Context carryover
   - Session management

7. **Source Quality**
   - Relevance scoring display
   - Multiple source citation
   - Confidence indicators

### Nice to Have (Additional 5%)

8. **Advanced Features**
   - Hybrid search (keyword + semantic)
   - Query rewriting
   - Response streaming

## Technical Stack

```
Required:
- Python 3.10+
- FastAPI
- LangChain or LlamaIndex
- Chroma or FAISS
- Ollama
- Sentence Transformers

Optional:
- Redis (for production sessions)
- Docker
- React/Streamlit (for UI)
```

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Retrieval Quality | 25% | Finds relevant documents |
| Response Accuracy | 25% | Answers are correct and grounded |
| Source Attribution | 15% | Proper citation of sources |
| Code Quality | 15% | Clean, documented, tested |
| UX/API Design | 10% | Intuitive and well-documented |
| Edge Cases | 10% | Handles out-of-scope queries |

## Milestones

### Week 1-2: Foundation
- [ ] Project setup and environment
- [ ] Document loading and chunking
- [ ] Basic embedding generation

### Week 3-4: RAG Pipeline
- [ ] Vector store integration
- [ ] Retrieval implementation
- [ ] Basic prompt engineering

### Week 5-6: API & Integration
- [ ] FastAPI endpoints
- [ ] Ollama integration
- [ ] Session management

### Week 7-8: Polish & Deploy
- [ ] Error handling
- [ ] Testing
- [ ] Docker deployment
- [ ] Documentation

## Getting Started

```bash
# Use the provided template
cp -r templates/domain_assistant my-assistant
cd my-assistant

# Set up environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Start Ollama
ollama serve
ollama pull llama3.2

# Run the application
python -m src.main
```

## Deliverables

1. **Source Code**: Complete, runnable application
2. **Design Document**: Architecture decisions and tradeoffs
3. **Demo**: 5-minute video demonstration
4. **Self-Evaluation**: Using the evaluation template

## Resources

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [Chroma Documentation](https://docs.trychroma.com/)
- [Ollama Documentation](https://ollama.ai/docs)
- Course 03 (RAG Fundamentals)
- Course 06 (Prompt Engineering)
