"""
API Routes for Domain Assistant

FastAPI router defining all API endpoints.
"""

import time
import uuid
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends

from .models import (
    ChatRequest, 
    ChatResponse, 
    SourceDocument,
    DocumentMetadata,
    DocumentListResponse,
    DocumentDeleteResponse,
    HealthResponse,
    ErrorResponse
)
from ..config import settings
from ..core.rag import RAGPipeline
from ..core.session import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency injection for RAG pipeline and session manager
# In production, these would be properly initialized singletons
_rag_pipeline: Optional[RAGPipeline] = None
_session_manager: Optional[SessionManager] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create RAG pipeline instance."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


def get_session_manager() -> SessionManager:
    """Get or create session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


# =============================================================================
# Chat Endpoints
# =============================================================================

@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Chat with the domain assistant",
    description="Send a message and receive a response based on domain knowledge."
)
async def chat(
    request: ChatRequest,
    rag: RAGPipeline = Depends(get_rag_pipeline),
    sessions: SessionManager = Depends(get_session_manager)
) -> ChatResponse:
    """
    Process a chat message using RAG.
    
    1. Retrieve relevant documents from vector store
    2. Build context-aware prompt
    3. Generate response using LLM
    4. Return response with sources
    """
    start_time = time.time()
    
    try:
        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())
        conversation_history = sessions.get_history(session_id)
        
        # Run RAG pipeline
        result = await rag.query(
            question=request.message,
            conversation_history=conversation_history,
            include_sources=request.include_sources
        )
        
        # Update conversation history
        sessions.add_message(session_id, "user", request.message)
        sessions.add_message(session_id, "assistant", result["response"])
        
        # Build source documents
        sources = []
        if request.include_sources and result.get("sources"):
            sources = [
                SourceDocument(
                    content=doc["content"],
                    source=doc["source"],
                    score=doc["score"],
                    metadata=doc.get("metadata", {})
                )
                for doc in result["sources"]
            ]
        
        processing_time = (time.time() - start_time) * 1000
        
        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            sources=sources,
            processing_time_ms=processing_time,
            model_used=settings.ollama_model
        )
        
    except Exception as e:
        logger.exception("Error processing chat request")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/chat/session/{session_id}",
    summary="Clear conversation history",
    description="Delete all messages for a specific session."
)
async def clear_session(
    session_id: str,
    sessions: SessionManager = Depends(get_session_manager)
) -> dict:
    """Clear conversation history for a session."""
    sessions.clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}


# =============================================================================
# Document Management Endpoints
# =============================================================================

@router.post(
    "/documents/upload",
    response_model=DocumentMetadata,
    summary="Upload a document",
    description="Upload and index a document for RAG."
)
async def upload_document(
    file: UploadFile = File(...),
    rag: RAGPipeline = Depends(get_rag_pipeline)
) -> DocumentMetadata:
    """
    Upload and index a document.
    
    Supported formats: .txt, .md, .pdf, .docx
    """
    # Validate file extension
    filename = file.filename or "unknown"
    extension = "." + filename.split(".")[-1].lower() if "." in filename else ""
    
    if extension not in settings.supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {settings.supported_extensions}"
        )
    
    try:
        content = await file.read()
        
        # Index document
        result = await rag.index_document(
            content=content,
            filename=filename,
            content_type=file.content_type or "application/octet-stream"
        )
        
        return DocumentMetadata(
            id=result["id"],
            filename=filename,
            chunk_count=result["chunk_count"],
            file_size_bytes=len(content)
        )
        
    except Exception as e:
        logger.exception(f"Error indexing document: {filename}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all documents",
    description="Get a list of all indexed documents."
)
async def list_documents(
    rag: RAGPipeline = Depends(get_rag_pipeline)
) -> DocumentListResponse:
    """List all indexed documents."""
    documents = await rag.list_documents()
    return DocumentListResponse(
        documents=documents,
        total_count=len(documents)
    )


@router.delete(
    "/documents/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document",
    description="Remove a document and its chunks from the index."
)
async def delete_document(
    document_id: str,
    rag: RAGPipeline = Depends(get_rag_pipeline)
) -> DocumentDeleteResponse:
    """Delete a document from the index."""
    try:
        result = await rag.delete_document(document_id)
        return DocumentDeleteResponse(
            success=True,
            message=f"Document {document_id} deleted",
            deleted_chunks=result.get("deleted_chunks", 0)
        )
    except Exception as e:
        logger.exception(f"Error deleting document: {document_id}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# System Endpoints
# =============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of all components."
)
async def health_check(
    rag: RAGPipeline = Depends(get_rag_pipeline)
) -> HealthResponse:
    """
    Comprehensive health check.
    
    Checks:
    - Ollama connection
    - Vector store status
    - Document count
    """
    # Check Ollama
    ollama_status = "healthy"
    try:
        await rag.check_ollama_health()
    except Exception:
        ollama_status = "unhealthy"
    
    # Check vector store
    vectorstore_status = "healthy"
    document_count = 0
    try:
        document_count = await rag.get_document_count()
    except Exception:
        vectorstore_status = "unhealthy"
    
    return HealthResponse(
        status="healthy" if ollama_status == "healthy" else "degraded",
        version=settings.app_version,
        ollama_status=ollama_status,
        vectorstore_status=vectorstore_status,
        document_count=document_count
    )


@router.get(
    "/",
    summary="API root",
    description="Welcome message and API info."
)
async def root() -> dict:
    """API root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }
