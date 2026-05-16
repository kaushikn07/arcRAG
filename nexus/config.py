from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """NEXUS configuration settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Database - Neon (Postgres with pgvector)
    neon_database_url: str

    # OpenRouter API Configuration
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # LLM Models
    llm_model_classify: str = "meta-llama/llama-3.3-70b-instruct:free"
    llm_model_synthesis: str = "google/gemma-3-27b-it:free"
    llm_model_fast: str = "meta-llama/llama-3.1-8b-instruct:free"

    # Neo4j Graph Database
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str

    # Redis Cache
    redis_url: str = "redis://localhost:6379"

    # Embeddings Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Application Settings
    app_name: str = "NEXUS"
    debug: bool = False
    log_level: str = "INFO"

    # Chunking Configuration
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Retrieval Configuration
    top_k_dense: int = 5
    top_k_keyword: int = 5
    top_k_hybrid: int = 10
    top_k_graph: int = 5

    # Guardrails Configuration
    enable_input_guard: bool = True
    enable_output_guard: bool = True
    max_tokens_output: int = 2048

    # Evaluation Configuration
    eval_dataset_path: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None


settings = Settings()
