"""
Configuration for Code Assistant

Code-specific settings for parsing, embedding, and generation.
"""

from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    app_name: str = "Code Assistant"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    
    # LLM (Ollama with code model)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "codellama"  # or deepseek-coder, starcoder2
    llm_temperature: float = 0.2  # Lower for code generation
    llm_max_tokens: int = 4096  # Longer for code
    
    # Code-specific settings
    supported_languages: list[str] = ["python", "javascript", "typescript"]
    max_file_size_kb: int = 500
    
    # Chunking strategy
    chunk_strategy: Literal["ast", "function", "fixed"] = "function"
    max_chunk_tokens: int = 1000
    chunk_overlap_tokens: int = 100
    
    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # For code-specific: "microsoft/codebert-base" or "Salesforce/codet5-base"
    
    # Vector store
    vector_store_path: str = "./data/vectorstore"
    repos_path: str = "./data/repos"
    
    # Retrieval
    retrieval_top_k: int = 10
    context_window_files: int = 5
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    cors_origins: list[str] = ["http://localhost:3000"]
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
