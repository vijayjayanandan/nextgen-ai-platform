from pydantic import PostgresDsn, Field, field_validator
from pydantic_core.core_schema import FieldValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Any, Dict, List, Optional, Union
import secrets
import json
from enum import Enum


class EnvironmentType(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # Application
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "IRCC AI Platform"
    ENVIRONMENT: EnvironmentType = EnvironmentType.DEVELOPMENT
    DEBUG: bool = False

    # Security
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 8
    ALGORITHM: str = "HS256"
    ALLOWED_HOSTS: List[str] = ["*"]

    @field_validator("API_V1_STR", mode="before")
    @classmethod
    def ensure_leading_slash(cls, v: str) -> str:
        return v if v.startswith("/") else f"/{v}"

    @field_validator("ALLOWED_HOSTS", mode="before")
    @classmethod
    def parse_allowed_hosts(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)  # Try parsing as JSON list
            except json.JSONDecodeError:
                return [host.strip() for host in v.split(",")]
        return v

    # Database
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_PORT: str = "5432"
    SQLALCHEMY_DATABASE_URI: Optional[PostgresDsn] = None

    @field_validator("SQLALCHEMY_DATABASE_URI", mode="after")
    def assemble_db_connection(cls, v: Optional[str], info: FieldValidationInfo) -> Any:
        if isinstance(v, str):
            return v
        values = info.data
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            username=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_SERVER"),
            port=int(values.get("POSTGRES_PORT", 5432)),
            path=f"{values.get('POSTGRES_DB') or ''}",
        )

    # Vector Database (Legacy - kept for backward compatibility)
    VECTOR_DB_TYPE: str = "qdrant"  # Changed default to qdrant
    VECTOR_DB_URI: str = ""  # Made optional
    VECTOR_DB_API_KEY: str = ""  # Made optional
    VECTOR_DB_NAMESPACE: str = "ircc-documents"
    
    # Qdrant Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "documents"
    QDRANT_MEMORY_COLLECTION: str = "conversation_memory"
    QDRANT_TIMEOUT: int = 30

    # LLM Services
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None

    # External API Base URLs
    OPENAI_API_BASE: str = "https://api.openai.com/v1"
    ANTHROPIC_API_BASE: str = "https://api.anthropic.com"

    # Local Embedding Configuration
    EMBEDDING_MODEL_TYPE: str = "local"  # "local", "openai", "anthropic"
    LOCAL_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_CACHE_SIZE: int = 1000
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DEVICE: str = "cpu"  # "cpu", "cuda", "auto"

    # On-Premises Model Config
    ON_PREM_MODEL_ENABLED: bool = False
    ON_PREM_MODEL_ENDPOINT: Optional[str] = None

    # Feature Flags
    ENABLE_RETRIEVAL_AUGMENTATION: bool = True
    ENABLE_CONTENT_FILTERING: bool = True
    ENABLE_EXPLANATION: bool = False
    ENABLE_FUNCTION_CALLING: bool = False

    # RAG Model Configuration (LLM-Agnostic) - Claude Setup
    RAG_QUERY_ANALYSIS_MODEL: str = "claude-3-haiku-20240307"
    RAG_GENERATION_MODEL: str = "claude-3-5-sonnet-20241022"
    RAG_RERANKING_MODEL: str = "claude-3-haiku-20240307"
    RAG_MEMORY_RETRIEVAL_MODEL: str = "claude-3-haiku-20240307"
    RAG_CITATION_MODEL: str = "claude-3-haiku-20240307"

    # Enhanced PII Filtering Configuration
    ENABLE_PII_FILTERING: bool = True
    PII_RISK_THRESHOLD: float = 0.7
    ENABLE_ML_PII_DETECTION: bool = False
    DEFAULT_ANONYMIZATION_METHOD: str = "tokenization"
    ENABLE_REVERSIBLE_ANONYMIZATION: bool = True
    PII_PROCESSING_TIMEOUT_SECONDS: int = 5
    ENABLE_PII_AUDIT_LOGGING: bool = True

    # Performance
    MAX_CONCURRENT_REQUESTS: int = 100
    REQUEST_TIMEOUT_SECONDS: int = 30

    # Logging
    LOG_LEVEL: str = "INFO"
    ENABLE_AUDIT_LOGGING: bool = True

    # IRCC System Integration
    GCDOCS_API_ENDPOINT: Optional[str] = None
    GCMS_API_ENDPOINT: Optional[str] = None
    DYNAMICS_API_ENDPOINT: Optional[str] = None
    


# Instantiate settings
settings = Settings()
