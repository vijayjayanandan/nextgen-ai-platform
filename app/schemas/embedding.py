from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field


class EmbeddingBase(BaseModel):
    """Base schema for embeddings."""
    model_name: str
    model_version: str
    dimensions: int


class EmbeddingCreate(EmbeddingBase):
    """Schema for creating embeddings."""
    chunk_id: UUID
    vector: List[float]
    vector_db_id: Optional[str] = None


class EmbeddingUpdate(BaseModel):
    """Schema for updating embeddings."""
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    vector: Optional[List[float]] = None
    vector_db_id: Optional[str] = None


class EmbeddingInDB(EmbeddingBase):
    """Schema for embeddings in the database."""
    id: UUID
    chunk_id: UUID
    created_at: datetime
    updated_at: datetime
    vector_db_id: Optional[str] = None

    class Config:
        from_attributes = True


class EmbeddingWithVector(EmbeddingInDB):
    """Schema for embeddings with vector data."""
    vector: List[float]


# Embedding generation schemas
class EmbeddingRequest(BaseModel):
    """Schema for embedding generation requests."""
    text: str
    model_name: Optional[str] = None


class EmbeddingResponse(BaseModel):
    """Schema for embedding generation responses."""
    embedding: List[float]
    model_name: str
    model_version: str
    dimensions: int
    usage: Dict[str, Any] = Field(default_factory=dict)


class BulkEmbeddingRequest(BaseModel):
    """Schema for bulk embedding generation requests."""
    texts: List[str]
    model_name: Optional[str] = None


class BulkEmbeddingResponse(BaseModel):
    """Schema for bulk embedding generation responses."""
    embeddings: List[List[float]]
    model_name: str
    model_version: str
    dimensions: int
    usage: Dict[str, Any] = Field(default_factory=dict)


# Vector search schemas
class VectorSearchQuery(BaseModel):
    """Schema for vector search queries."""
    query: List[float]  # Vector embedding for search
    filters: Optional[Dict[str, Any]] = None
    top_k: int = 5
    include_metadata: bool = True
    include_vectors: bool = False


class VectorSearchResult(BaseModel):
    """Schema for vector search results."""
    chunk_id: UUID
    document_id: UUID
    content: str
    similarity: float
    metadata: Optional[Dict[str, Any]] = None
    vector: Optional[List[float]] = None
