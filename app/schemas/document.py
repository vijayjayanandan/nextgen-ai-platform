from typing import List, Dict, Optional, Any, Union
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field

from app.models.document import DocumentSourceType, DocumentStatus


class DocumentChunkBase(BaseModel):
    """Base schema for document chunks."""
    content: str
    chunk_index: int
    metadata: Optional[Dict[str, Any]] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None


class DocumentChunkCreate(DocumentChunkBase):
    """Schema for creating document chunks."""
    document_id: UUID


class DocumentChunkUpdate(BaseModel):
    """Schema for updating document chunks."""
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None


class DocumentChunkInDB(DocumentChunkBase):
    """Schema for document chunks in the database."""
    id: UUID
    document_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentBase(BaseModel):
    """Base schema for documents."""
    title: str
    description: Optional[str] = None
    source_type: DocumentSourceType
    source_id: Optional[str] = None
    source_url: Optional[str] = None
    content_type: str
    language: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    is_public: bool = False
    security_classification: Optional[str] = None
    allowed_roles: List[str] = Field(default_factory=list)


class DocumentCreate(DocumentBase):
    """Schema for creating documents."""
    # Optional fields for document creation that aren't in the base
    status: DocumentStatus = DocumentStatus.PENDING
    content_hash: Optional[str] = None
    storage_path: Optional[str] = None


class DocumentUpdate(BaseModel):
    """Schema for updating documents."""
    title: Optional[str] = None
    description: Optional[str] = None
    source_url: Optional[str] = None
    language: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    status: Optional[DocumentStatus] = None
    error_message: Optional[str] = None
    is_public: Optional[bool] = None
    security_classification: Optional[str] = None
    allowed_roles: Optional[List[str]] = None
    storage_path: Optional[str] = None


class DocumentInDB(DocumentBase):
    """Schema for documents in the database."""
    id: UUID
    created_at: datetime
    updated_at: datetime
    status: DocumentStatus
    error_message: Optional[str] = None
    content_hash: Optional[str] = None
    storage_path: Optional[str] = None

    class Config:
        from_attributes = True


class DocumentWithChunks(DocumentInDB):
    """Schema for documents with their chunks."""
    chunks: List[DocumentChunkInDB] = []


# File upload and processing schemas
class DocumentUploadResponse(BaseModel):
    """Response schema for document uploads."""
    document_id: UUID
    status: DocumentStatus
    message: str


class DocumentProcessingStatus(BaseModel):
    """Schema for document processing status."""
    document_id: UUID
    status: DocumentStatus
    progress: Optional[float] = None
    message: Optional[str] = None
    chunks_processed: Optional[int] = None
    total_chunks: Optional[int] = None


# Search/retrieval schemas
class DocumentSearchQuery(BaseModel):
    """Schema for document search queries."""
    query: str
    filters: Optional[Dict[str, Any]] = None
    limit: int = 10
    include_chunks: bool = True


class DocumentSearchResult(BaseModel):
    """Schema for document search results."""
    document: DocumentInDB
    chunks: Optional[List[DocumentChunkInDB]] = None
    similarity_score: float