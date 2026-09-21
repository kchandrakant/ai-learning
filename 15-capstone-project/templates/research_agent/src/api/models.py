"""
Pydantic Models for Research Agent API
"""

from typing import Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field


# =============================================================================
# Research Request/Response
# =============================================================================

class ResearchRequest(BaseModel):
    """Request to start a research task."""
    question: str = Field(..., min_length=10, description="Research question")
    depth: Literal["quick", "standard", "detailed"] = Field(
        "standard",
        description="Research depth"
    )
    sources: list[str] = Field(
        default=["web", "arxiv"],
        description="Sources to search"
    )
    max_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max research iterations"
    )
    output_format: Literal["summary", "report", "outline"] = Field(
        "summary",
        description="Desired output format"
    )
    expertise_level: Literal["beginner", "intermediate", "expert"] = Field(
        "intermediate",
        description="Target expertise level"
    )


class ResearchResponse(BaseModel):
    """Initial response when starting research."""
    task_id: str
    status: str
    message: str
    estimated_time_seconds: Optional[int] = None


class Finding(BaseModel):
    """A single research finding."""
    content: str
    source_url: str
    source_name: str
    relevance_score: float
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)


class SubQuestion(BaseModel):
    """A sub-question in the research plan."""
    question: str
    priority: int
    findings: list[Finding] = Field(default_factory=list)
    answered: bool = False


class ResearchResult(BaseModel):
    """Complete research result."""
    summary: str
    key_findings: list[str]
    sub_questions: list[SubQuestion]
    sources_used: int
    iterations_taken: int
    confidence: float = Field(..., ge=0, le=1)
    limitations: list[str] = Field(default_factory=list)


class ResearchStatus(BaseModel):
    """Status of a research task."""
    task_id: str
    status: Literal["planning", "searching", "synthesizing", "completed", "failed", "cancelled"]
    progress: float = Field(..., ge=0, le=100)
    question: str
    current_step: Optional[str] = None
    result: Optional[ResearchResult] = None
    error: Optional[str] = None


# =============================================================================
# Report Generation
# =============================================================================

class ReportRequest(BaseModel):
    """Request to generate a report from research results."""
    format: Literal["markdown", "html", "pdf"] = "markdown"
    include_sources: bool = True
    include_methodology: bool = False
    max_length_words: Optional[int] = Field(None, description="Max report length")


class ReportResponse(BaseModel):
    """Generated report response."""
    format: str
    content: str
    word_count: int
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    download_url: Optional[str] = None


# =============================================================================
# Sources
# =============================================================================

class SourceInfo(BaseModel):
    """Information about a research source."""
    name: str
    enabled: bool
    status: Literal["available", "unavailable", "rate_limited"]
    description: str
    last_used: Optional[datetime] = None


class SourcesResponse(BaseModel):
    """List of available sources."""
    sources: list[SourceInfo]
