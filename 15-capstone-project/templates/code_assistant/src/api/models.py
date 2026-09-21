"""
Pydantic Models for Code Assistant API
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


# =============================================================================
# Code Generation
# =============================================================================

class GenerateRequest(BaseModel):
    """Request for code generation."""
    prompt: str = Field(..., description="Natural language description of desired code")
    language: str = Field("python", description="Target programming language")
    context_repo: Optional[str] = Field(None, description="Repository for context")
    style_guide: Optional[str] = Field(None, description="Coding style preferences")
    include_tests: bool = Field(False, description="Generate tests alongside code")
    include_docs: bool = Field(True, description="Include docstrings/comments")


class GenerateResponse(BaseModel):
    """Response from code generation."""
    code: str = Field(..., description="Generated code")
    language: str
    explanation: str = Field(..., description="Brief explanation of the code")
    tests: Optional[str] = Field(None, description="Generated tests if requested")
    context_used: list[str] = Field(default_factory=list, description="Files used for context")


# =============================================================================
# Code Review
# =============================================================================

class ReviewRequest(BaseModel):
    """Request for code review."""
    code: str = Field(..., description="Code to review")
    language: str = Field("python", description="Programming language")
    context_repo: Optional[str] = Field(None, description="Repository for project context")
    review_focus: list[str] = Field(
        default=["bugs", "style", "performance", "security"],
        description="Areas to focus review on"
    )


class ReviewFeedback(BaseModel):
    """Individual review feedback item."""
    severity: Literal["error", "warning", "info", "suggestion"]
    category: str
    line: Optional[int] = None
    message: str
    suggestion: Optional[str] = None


class ReviewResponse(BaseModel):
    """Response from code review."""
    overall_assessment: str
    score: int = Field(..., ge=0, le=100, description="Code quality score")
    feedback: list[ReviewFeedback]
    summary: str


# =============================================================================
# Debugging
# =============================================================================

class DebugRequest(BaseModel):
    """Request for debugging help."""
    code: str = Field(..., description="Code with the bug")
    error: str = Field(..., description="Error message or description")
    language: str = Field("python", description="Programming language")
    stack_trace: Optional[str] = Field(None, description="Full stack trace if available")


class DebugSuggestion(BaseModel):
    """A suggested fix."""
    description: str
    fixed_code: str
    confidence: float = Field(..., ge=0, le=1)


class DebugResponse(BaseModel):
    """Response from debugging."""
    root_cause: str = Field(..., description="Identified root cause")
    explanation: str = Field(..., description="Detailed explanation")
    suggestions: list[DebugSuggestion]
    prevention_tips: list[str] = Field(default_factory=list)


# =============================================================================
# Code Explanation
# =============================================================================

class ExplainRequest(BaseModel):
    """Request for code explanation."""
    code: str = Field(..., description="Code to explain")
    language: str = Field("python", description="Programming language")
    detail_level: Literal["brief", "detailed", "line-by-line"] = Field("detailed")
    audience: Literal["beginner", "intermediate", "expert"] = Field("intermediate")


class ExplainResponse(BaseModel):
    """Response from code explanation."""
    summary: str = Field(..., description="High-level summary")
    explanation: str = Field(..., description="Detailed explanation")
    concepts: list[str] = Field(default_factory=list, description="Key concepts used")
    complexity: str = Field(..., description="Complexity assessment")


# =============================================================================
# Repository Management
# =============================================================================

class RepoIndexRequest(BaseModel):
    """Request to index a repository."""
    path: Optional[str] = Field(None, description="Local path to repository")
    github_url: Optional[str] = Field(None, description="GitHub URL to clone")
    name: str = Field(..., description="Name for the indexed repository")
    branch: str = Field("main", description="Branch to index")
    include_patterns: list[str] = Field(default=["**/*.py", "**/*.js", "**/*.ts"])
    exclude_patterns: list[str] = Field(default=["**/node_modules/**", "**/__pycache__/**"])


class RepoIndexResponse(BaseModel):
    """Response from repository indexing."""
    name: str
    files_indexed: int
    chunks_created: int
    languages_detected: list[str]
    indexing_time_seconds: float
