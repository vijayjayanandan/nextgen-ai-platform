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

    # Vector Database
    VECTOR_DB_TYPE: str = "pinecone"
    VECTOR_DB_URI: str
    VECTOR_DB_API_KEY: str
    VECTOR_DB_NAMESPACE: str = "ircc-documents"

    # LLM Services
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None

    # External API Base URLs
    OPENAI_API_BASE: str = "https://api.openai.com/v1"
    ANTHROPIC_API_BASE: str = "https://api.anthropic.com"

    # On-Premises Model Config
    ON_PREM_MODEL_ENABLED: bool = False
    ON_PREM_MODEL_ENDPOINT: Optional[str] = None

    # Feature Flags
    ENABLE_RETRIEVAL_AUGMENTATION: bool = True
    ENABLE_CONTENT_FILTERING: bool = True
    ENABLE_EXPLANATION: bool = True

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
