"""
Configuration Management for Domain Assistant

Centralized configuration using environment variables and Pydantic settings.
"""

from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # ==========================================================================
    # Application Settings
    # ==========================================================================
    app_name: str = "Domain Assistant"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    
    # ==========================================================================
    # LLM Settings (Ollama)
    # ==========================================================================
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"  # or "mistral", "codellama", etc.
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    llm_timeout: int = 120  # seconds
    
    # ==========================================================================
    # RAG Settings
    # ==========================================================================
    # Embedding model (local via sentence-transformers)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Vector store
    vector_store_type: Literal["chroma", "faiss"] = "chroma"
    vector_store_path: str = "./data/vectorstore"
    collection_name: str = "domain_docs"
    
    # Retrieval settings
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.5
    
    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # ==========================================================================
    # Document Processing
    # ==========================================================================
    docs_path: str = "./data/documents"
    supported_extensions: list[str] = [".txt", ".md", ".pdf", ".docx"]
    
    # ==========================================================================
    # API Settings
    # ==========================================================================
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000"]
    
    # ==========================================================================
    # Memory & Conversation
    # ==========================================================================
    max_conversation_history: int = 10
    session_timeout_minutes: int = 30
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Using lru_cache ensures settings are only loaded once.
    """
    return Settings()


# Convenience access
settings = get_settings()


# ==========================================================================
# Configuration Validation
# ==========================================================================

def validate_config():
    """
    Validate configuration on startup.
    
    Raises:
        ValueError: If configuration is invalid.
    """
    s = get_settings()
    
    # Validate chunk settings
    if s.chunk_overlap >= s.chunk_size:
        raise ValueError(
            f"chunk_overlap ({s.chunk_overlap}) must be less than "
            f"chunk_size ({s.chunk_size})"
        )
    
    # Validate retrieval settings
    if s.retrieval_top_k < 1:
        raise ValueError("retrieval_top_k must be at least 1")
    
    if not 0 <= s.retrieval_score_threshold <= 1:
        raise ValueError("retrieval_score_threshold must be between 0 and 1")
    
    # Validate temperature
    if not 0 <= s.llm_temperature <= 2:
        raise ValueError("llm_temperature should be between 0 and 2")
    
    return True


if __name__ == "__main__":
    # Print current configuration (useful for debugging)
    import json
    s = get_settings()
    print("Current Configuration:")
    print(json.dumps(s.model_dump(), indent=2, default=str))
    
    print("\nValidating configuration...")
    validate_config()
    print("✓ Configuration is valid")
