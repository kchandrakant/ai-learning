# Sample Design Document: Medical FAQ Assistant

**Author**: Sample Student  
**Date**: September 2026  
**Project**: Domain-Specific Assistant (Option A)

---

## 1. Executive Summary

This project builds a RAG-powered assistant that answers questions about common medical conditions using authoritative health sources. The assistant is designed for educational purposes and general health information only.

**Disclaimer**: This assistant is NOT for medical diagnosis or treatment advice.

## 2. Problem Statement

### 2.1 Background
People frequently search online for health information but often encounter:
- Unreliable sources
- Overwhelming amounts of information
- Difficulty understanding medical terminology

### 2.2 Objective
Create a conversational assistant that provides accurate, sourced health information from authoritative sources with clear explanations.

### 2.3 Success Criteria
- Retrieval accuracy > 80% (relevant sources found)
- Response accuracy > 90% (factually correct)
- User satisfaction > 4/5 in usability testing

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌────────────────────────────────────────────────────┐
│                    User Interface                   │
│              (API / Streamlit Demo)                │
└─────────────────────┬──────────────────────────────┘
                      │ HTTP/REST
┌─────────────────────▼──────────────────────────────┐
│                   FastAPI Server                    │
├─────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │   Session   │  │    Chat     │  │  Document  │ │
│  │   Manager   │  │  Handler    │  │  Manager   │ │
│  └─────────────┘  └──────┬──────┘  └────────────┘ │
└──────────────────────────┼──────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────┐
│                    RAG Pipeline                      │
├──────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │
│  │  Retriever  │  │   Context   │  │  Response  │  │
│  │  (ChromaDB) │  │   Builder   │  │ Generator  │  │
│  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘  │
└─────────┼────────────────┼───────────────┼──────────┘
          │                │               │
┌─────────▼────────────────▼───────────────▼──────────┐
│                 External Services                    │
├──────────────────────────────────────────────────────┤
│  ┌────────────────┐        ┌────────────────┐       │
│  │    ChromaDB    │        │     Ollama     │       │
│  │ (Vector Store) │        │     (LLM)      │       │
│  └────────────────┘        └────────────────┘       │
└──────────────────────────────────────────────────────┘
```

### 3.2 Data Flow

1. **Document Ingestion**: Health documents → chunking → embedding → ChromaDB
2. **Query Processing**: User question → embedding → vector search → retrieve top-k
3. **Response Generation**: Context + question → prompt → Llama 3 → response

## 4. Key Components

### 4.1 Document Processing

**Chunking Strategy**: 
- Chunk size: 500 tokens
- Overlap: 50 tokens
- Preserve paragraph boundaries when possible

**Rationale**: Medical information often has important context. Smaller chunks with overlap ensure we don't split critical information.

### 4.2 Embedding Model

**Choice**: `sentence-transformers/all-MiniLM-L6-v2`

**Rationale**:
- Fast and lightweight
- Good balance of quality and speed
- Runs locally without API costs

**Alternatives Considered**:
- `all-mpnet-base-v2`: Better quality but slower
- OpenAI `text-embedding-3-small`: Requires API, costs money

### 4.3 Vector Store

**Choice**: ChromaDB

**Rationale**:
- Simple setup
- Persistent storage
- Good enough for this scale (< 10K documents)

### 4.4 LLM

**Choice**: Llama 3.2 8B via Ollama

**Rationale**:
- Local deployment (privacy for health queries)
- Good instruction following
- No API costs

## 5. Prompt Engineering

### 5.1 System Prompt

```
You are a helpful health information assistant. Your role is to provide 
accurate, educational information about health topics using ONLY the 
provided context documents.

CRITICAL RULES:
1. Only answer based on the provided context
2. If the context doesn't contain the answer, say so clearly
3. Always recommend consulting a healthcare provider for personal advice
4. Never provide diagnosis or treatment recommendations
5. Cite which source document you're using

Format your response with:
- A clear, direct answer
- Supporting details from the sources
- A reminder to consult professionals when appropriate
```

### 5.2 Query Prompt

```
Context Documents:
{context}

Conversation History:
{history}

User Question: {question}

Provide a helpful, accurate response based solely on the context above.
```

## 6. API Design

### 6.1 Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Send a message, get response |
| POST | `/documents` | Upload new document |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{id}` | Remove a document |
| GET | `/health` | Service health check |

### 6.2 Chat Request/Response

```json
// Request
{
  "message": "What are common symptoms of the flu?",
  "session_id": "user-123",
  "include_sources": true
}

// Response
{
  "response": "According to the CDC guidelines, common flu symptoms include...",
  "session_id": "user-123",
  "sources": [
    {
      "content": "Flu symptoms often come on suddenly...",
      "source": "cdc_flu_guide.pdf",
      "score": 0.89
    }
  ],
  "processing_time_ms": 1250
}
```

## 7. Testing Strategy

### 7.1 Unit Tests
- Document chunking
- Embedding generation
- Prompt construction

### 7.2 Integration Tests
- End-to-end RAG pipeline
- API endpoint responses
- Error handling

### 7.3 Evaluation Tests
- Retrieval accuracy (MRR, Hit@5)
- Response factuality
- Source attribution correctness

### 7.4 Test Questions

| Question | Expected Source | Key Points |
|----------|-----------------|------------|
| "What is diabetes?" | diabetes_overview.md | Types, causes |
| "How is the flu different from a cold?" | flu_vs_cold.md | Comparison |
| "What causes migraines?" | headache_guide.md | Triggers |

## 8. Deployment

### 8.1 Docker Compose

```yaml
services:
  assistant:
    build: .
    ports: ["8000:8000"]
    depends_on: [ollama]
    
  ollama:
    image: ollama/ollama
    ports: ["11434:11434"]
    volumes: [ollama_data:/root/.ollama]
```

### 8.2 Resource Requirements
- Memory: 8GB minimum (Llama 3.2 8B)
- Storage: 10GB for model + data
- CPU: 4 cores recommended

## 9. Limitations & Future Work

### 9.1 Known Limitations
- Only handles English text
- No real-time updates (static document corpus)
- Single-turn context only (no deep conversation memory)

### 9.2 Future Improvements
- Add multi-language support
- Implement conversation summarization for longer sessions
- Add user feedback collection for improvement

## 10. Ethical Considerations

- **Privacy**: All processing is local, no data sent to external APIs
- **Medical Disclaimer**: Prominently displayed, included in responses
- **Source Transparency**: All responses cite sources
- **Bias Awareness**: Limited to provided documents, may not cover all perspectives

---

**Appendix A**: Document Sources
- CDC Health Guides (public domain)
- WHO Fact Sheets (public domain)
- NIH Health Information (public domain)
