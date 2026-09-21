"""
Pydantic Models for API Request/Response

These models define the data structures for API communication.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# =============================================================================
# Chat Models
# =============================================================================

class Message(BaseModel):
    """A single message in a conversation."""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    include_sources: bool = Field(True, description="Include source documents in response")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "What are the key features of this product?",
                    "session_id": "user-123-session-1",
                    "include_sources": True
                }
            ]
        }
    }


class SourceDocument(BaseModel):
    """A source document used for RAG."""
    content: str = Field(..., description="Relevant text chunk")
    source: str = Field(..., description="Document source/filename")
    score: float = Field(..., ge=0, le=1, description="Relevance score")
    metadata: dict = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="Assistant's response")
    session_id: str = Field(..., description="Session ID")
    sources: list[SourceDocument] = Field(default_factory=list)
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_used: str = Field(..., description="LLM model used")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "response": "Based on the documentation, the key features are...",
                    "session_id": "user-123-session-1",
                    "sources": [
                        {
                            "content": "Feature 1: Advanced analytics...",
                            "source": "product_guide.pdf",
                            "score": 0.92,
                            "metadata": {"page": 5}
                        }
                    ],
                    "processing_time_ms": 1250.5,
                    "model_used": "llama3.2"
                }
            ]
        }
    }


# =============================================================================
# Document Models
# =============================================================================

class DocumentUpload(BaseModel):
    """Request model for document upload."""
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type")
    
    
class DocumentMetadata(BaseModel):
    """Metadata for an indexed document."""
    id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    chunk_count: int = Field(..., description="Number of chunks created")
    indexed_at: datetime = Field(default_factory=datetime.utcnow)
    file_size_bytes: int = Field(..., ge=0)
    

class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    documents: list[DocumentMetadata] = Field(default_factory=list)
    total_count: int = Field(..., ge=0)


class DocumentDeleteResponse(BaseModel):
    """Response model for document deletion."""
    success: bool
    message: str
    deleted_chunks: int = Field(default=0)


# =============================================================================
# System Models
# =============================================================================

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    ollama_status: str = Field(..., description="Ollama connection status")
    vectorstore_status: str = Field(..., description="Vector store status")
    document_count: int = Field(..., ge=0)
    

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict] = Field(None, description="Additional error details")
