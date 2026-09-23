"""
Configuration management for Fintech Agentic RAG Demo (Docker Edition)

Migrated from Modal to Docker-based deployment.
Uses environment variables from docker-compose.yml and .env file.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Neo4jConfig(BaseModel):
    """Neo4j database configuration"""
    uri: str = Field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://neo4j:7687"))
    user: str = Field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    password: str = Field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    database: str = Field(default="neo4j")


class PostgresConfig(BaseModel):
    """PostgreSQL database configuration"""
    host: str = Field(default_factory=lambda: os.getenv("POSTGRES_HOST", "postgresql"))
    port: int = Field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    database: str = Field(default_factory=lambda: os.getenv("POSTGRES_DB", "fintech_rag"))
    user: str = Field(default_factory=lambda: os.getenv("POSTGRES_USER", "fintech_user"))
    password: str = Field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""))


class RedisConfig(BaseModel):
    """Redis cache configuration"""
    host: str = Field(default_factory=lambda: os.getenv("REDIS_HOST", "redis"))
    port: int = Field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    db: int = Field(default=0)
    password: Optional[str] = Field(default_factory=lambda: os.getenv("REDIS_PASSWORD"))
    ttl: int = Field(default_factory=lambda: int(os.getenv("REDIS_CACHE_TTL", "3600")))


class LLMConfig(BaseModel):
    """LLM and Embedding model configuration"""
    # LLM model for reasoning
    model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-70B-Instruct"))

    # Embedding model for semantic search
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"))

    # HuggingFace token (required)
    huggingface_token: str = Field(default_factory=lambda: os.getenv("HUGGINGFACE_TOKEN", ""))

    # OpenAI API key (optional - only if using OpenAI models)
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # LLM parameters
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=1024)

    # Base URL for HuggingFace Inference API
    base_url: str = Field(default="https://api-inference.huggingface.co/v1")


class LangSmithConfig(BaseModel):
    """LangSmith observability configuration (optional)"""
    enabled: bool = Field(
        default_factory=lambda: os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    )
    api_key: str = Field(default_factory=lambda: os.getenv("LANGCHAIN_API_KEY", ""))
    project: str = Field(default_factory=lambda: os.getenv("LANGCHAIN_PROJECT", "fintech-agentic-rag-docker"))
    endpoint: str = Field(default="https://api.smith.langchain.com")


class VectorStoreConfig(BaseModel):
    """Vector store (ChromaDB) configuration"""
    persist_directory: str = Field(default="/app/vectors")  # Docker volume mount
    collection_name: str = Field(default="fintech-rag-demo")
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)


class PrometheusConfig(BaseModel):
    """Prometheus monitoring configuration"""
    enabled: bool = Field(default=True)
    port: int = Field(default=8000)
    endpoint: str = Field(default="/metrics")


class AppConfig(BaseSettings):
    """Main application configuration"""

    # Environment
    environment: str = Field(default_factory=lambda: os.getenv("ENVIRONMENT", "production"))

    # Database configurations
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)

    # LLM configuration
    llm: LLMConfig = Field(default_factory=LLMConfig)

    # Observability
    langsmith: LangSmithConfig = Field(default_factory=LangSmithConfig)
    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)

    # Vector store
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)

    # Application metadata
    app_name: str = Field(default="Fintech Agentic RAG v2.0")
    version: str = Field(default="2.0.0")
    architecture: str = Field(default="5-Layer Agentic RAG")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global config instance
config = AppConfig()


def get_config() -> AppConfig:
    """
    Return the application configuration.

    Settings are read once at import time, so every caller sees the same
    instance. Prefer this over importing the module-level `config` binding:
    it is what the tests and scripts use, and it keeps the singleton behind
    an accessor should loading ever need to become lazy or per-environment.
    """
    return config


def validate_config() -> bool:
    """
    Validate that all required configuration is present

    Returns:
        True if valid, False otherwise
    """
    errors = []

    # Required: HuggingFace token
    if not config.llm.huggingface_token:
        errors.append("HUGGINGFACE_TOKEN not set (required for embeddings and LLM)")

    # Required: Neo4j password
    if not config.neo4j.password:
        errors.append("NEO4J_PASSWORD not set (required for graph database)")

    # Required: PostgreSQL password
    if not config.postgres.password:
        errors.append("POSTGRES_PASSWORD not set (required for logging database)")

    # Warning: OpenAI API key (optional but recommended)
    if not config.llm.openai_api_key:
        print("ℹOPENAI_API_KEY not set (optional - using HuggingFace Inference API)")

    # Warning: LangSmith (optional)
    if config.langsmith.enabled and not config.langsmith.api_key:
        errors.append("LANGCHAIN_API_KEY not set but LANGCHAIN_TRACING_V2=true")

    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"   - {error}")
        return False

    print("Configuration validated successfully")
    print(f"   Environment: {config.environment}")
    print(f"   LLM Model: {config.llm.model}")
    print(f"   Embedding Model: {config.llm.embedding_model}")
    print(f"   Neo4j: {config.neo4j.uri}")
    print(f"   PostgreSQL: {config.postgres.host}:{config.postgres.port}/{config.postgres.database}")
    print(f"   Redis: {config.redis.host}:{config.redis.port}")
    print(f"   LangSmith: {'Enabled' if config.langsmith.enabled else 'Disabled'}")
    print(f"   Prometheus: {'Enabled' if config.prometheus.enabled else 'Disabled'}")

    return True


def print_config_summary() -> None:
    """Print configuration summary for debugging"""
    print("\n" + "=" * 80)
    print("CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"App: {config.app_name} v{config.version}")
    print(f"Architecture: {config.architecture}")
    print(f"Environment: {config.environment}")
    print("\nDatabases:")
    print(f"  - Neo4j: {config.neo4j.uri}")
    print(f"  - PostgreSQL: {config.postgres.user}@{config.postgres.host}:{config.postgres.port}/{config.postgres.database}")
    print(f"  - Redis: {config.redis.host}:{config.redis.port} (TTL: {config.redis.ttl}s)")
    print("\nLLM:")
    print(f"  - Model: {config.llm.model}")
    print(f"  - Embeddings: {config.llm.embedding_model}")
    print(f"  - Temperature: {config.llm.temperature}")
    print(f"  - Max Tokens: {config.llm.max_tokens}")
    print("\nMonitoring:")
    print(f"  - LangSmith: {'Enabled' if config.langsmith.enabled else 'Disabled'}")
    print(f"  - Prometheus: {'Enabled' if config.prometheus.enabled else 'Disabled'}")
    print("\nVector Store:")
    print(f"  - Path: {config.vectorstore.persist_directory}")
    print(f"  - Collection: {config.vectorstore.collection_name}")
    print("=" * 80 + "\n")
