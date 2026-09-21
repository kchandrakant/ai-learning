"""
Configuration for Research Agent

Settings for search sources, agent behavior, and output generation.
"""

from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    app_name: str = "Research Agent"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    
    # LLM
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"  # Need capable model for reasoning
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096
    
    # Agent behavior
    max_iterations: int = 3
    max_sources_per_query: int = 10
    search_timeout_seconds: int = 30
    
    # Sources
    enabled_sources: list[str] = ["web", "arxiv", "wikipedia"]
    web_search_api: str = "duckduckgo"  # or searxng
    
    # ArXiv settings
    arxiv_max_results: int = 10
    arxiv_sort_by: str = "relevance"
    
    # Output
    default_output_format: str = "report"
    max_report_length_words: int = 2000
    
    # Caching
    enable_cache: bool = True
    cache_ttl_hours: int = 24
    cache_path: str = "./data/cache"
    
    # Reports
    reports_path: str = "./data/reports"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8002
    cors_origins: list[str] = ["http://localhost:3000"]
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
