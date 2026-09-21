"""
API Routes for Research Agent

Endpoints for research queries, reports, and source management.
"""

import uuid
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks

from .models import (
    ResearchRequest, ResearchResponse, ResearchStatus,
    ReportRequest, ReportResponse,
    SourcesResponse
)
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for research tasks (use Redis in production)
_research_tasks: dict = {}


# =============================================================================
# Research Endpoints
# =============================================================================

@router.post("/research", response_model=ResearchResponse)
async def start_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks
) -> ResearchResponse:
    """
    Start a research task on a question.
    
    The research runs asynchronously. Use the returned task_id
    to check status and retrieve results.
    """
    task_id = str(uuid.uuid4())
    
    # Initialize task
    _research_tasks[task_id] = {
        "status": "planning",
        "question": request.question,
        "progress": 0,
        "result": None,
        "error": None
    }
    
    # TODO: Implement actual research in background
    # background_tasks.add_task(run_research, task_id, request)
    
    # For now, return stub
    return ResearchResponse(
        task_id=task_id,
        status="planning",
        message="Research started. Implementation pending - see TODOs in routes.py"
    )


@router.get("/research/{task_id}", response_model=ResearchStatus)
async def get_research_status(task_id: str) -> ResearchStatus:
    """Get the status and results of a research task."""
    if task_id not in _research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = _research_tasks[task_id]
    return ResearchStatus(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        question=task["question"],
        result=task["result"],
        error=task["error"]
    )


@router.post("/research/{task_id}/report", response_model=ReportResponse)
async def generate_report(
    task_id: str,
    request: ReportRequest
) -> ReportResponse:
    """Generate a formatted report from research results."""
    if task_id not in _research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = _research_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Research not complete. Status: {task['status']}"
        )
    
    # TODO: Generate report in requested format
    raise HTTPException(
        status_code=501,
        detail="Report generation not yet implemented"
    )


@router.delete("/research/{task_id}")
async def cancel_research(task_id: str) -> dict:
    """Cancel an ongoing research task."""
    if task_id not in _research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # TODO: Actually cancel the background task
    _research_tasks[task_id]["status"] = "cancelled"
    
    return {"message": f"Task {task_id} cancelled"}


# =============================================================================
# Source Management
# =============================================================================

@router.get("/sources", response_model=SourcesResponse)
async def list_sources() -> SourcesResponse:
    """List available research sources and their status."""
    sources = []
    
    for source_name in settings.enabled_sources:
        sources.append({
            "name": source_name,
            "enabled": True,
            "status": "available",  # TODO: Actually check status
            "description": _get_source_description(source_name)
        })
    
    return SourcesResponse(sources=sources)


def _get_source_description(name: str) -> str:
    """Get description for a source."""
    descriptions = {
        "web": "General web search via DuckDuckGo",
        "arxiv": "Academic papers from arXiv",
        "wikipedia": "Wikipedia articles",
        "local": "Locally uploaded documents"
    }
    return descriptions.get(name, "Unknown source")


@router.post("/sources/test/{source_name}")
async def test_source(source_name: str) -> dict:
    """Test connectivity to a specific source."""
    if source_name not in settings.enabled_sources:
        raise HTTPException(status_code=404, detail="Source not enabled")
    
    # TODO: Actually test the source
    return {
        "source": source_name,
        "status": "available",
        "message": "Source connectivity test not implemented"
    }


# =============================================================================
# Cache Management
# =============================================================================

@router.post("/cache/clear")
async def clear_cache() -> dict:
    """Clear the search result cache."""
    # TODO: Implement cache clearing
    return {"message": "Cache clearing not implemented"}


@router.get("/cache/stats")
async def cache_stats() -> dict:
    """Get cache statistics."""
    # TODO: Return actual cache stats
    return {
        "entries": 0,
        "size_mb": 0,
        "hit_rate": 0
    }


# =============================================================================
# System
# =============================================================================

@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.app_version,
        "model": settings.ollama_model,
        "enabled_sources": settings.enabled_sources,
        "active_tasks": len([t for t in _research_tasks.values() if t["status"] == "running"])
    }


@router.get("/")
async def root() -> dict:
    """API root."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "endpoints": {
            "research": "/research",
            "sources": "/sources",
            "health": "/health"
        }
    }
