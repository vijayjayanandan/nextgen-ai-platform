from sqlalchemy import Column, String, ForeignKey, Text, Integer, Boolean, Enum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
import enum

from app.db.base import BaseModel


class DocumentSourceType(str, enum.Enum):
    """Enum for document source types."""
    GCDOCS = "gcdocs"
    GCMS = "gcms"
    DYNAMICS = "dynamics"
    UPLOADED = "uploaded"
    WEB = "web"
    OTHER = "other"


class DocumentStatus(str, enum.Enum):
    """Enum for document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class Document(BaseModel):
    """
    Document model representing original documents in the system.
    """
    title = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Source information
    source_type = Column(Enum(DocumentSourceType), nullable=False, index=True)
    source_id = Column(String, nullable=True, index=True)  # ID in the source system
    source_url = Column(String, nullable=True)
    
    # Content type and metadata
    content_type = Column(String, nullable=False)  # e.g., "application/pdf"
    language = Column(String, nullable=True, index=True)
    meta_data = Column(JSONB, nullable=True)
    
    # Processing status
    status = Column(Enum(DocumentStatus), default=DocumentStatus.PENDING, nullable=False, index=True)
    error_message = Column(Text, nullable=True)
    
    # Security and access control
    is_public = Column(Boolean, default=False, nullable=False)
    security_classification = Column(String, nullable=True, index=True)
    allowed_roles = Column(JSONB, default=[], nullable=False)
    
    # Original content hash for deduplication and versioning
    content_hash = Column(String, nullable=True, index=True)
    
    # Storage information
    storage_path = Column(String, nullable=True)
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document {self.id}: {self.title}>"


class DocumentChunk(BaseModel):
    """
    DocumentChunk model representing chunks of text extracted from documents.
    These chunks are used for embedding and retrieval.
    """
    document_id = Column(UUID(as_uuid=True), ForeignKey("document.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Chunk content and metadata
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    meta_data = Column(JSONB, nullable=True)
    
    # Page or section information
    page_number = Column(Integer, nullable=True)
    section_title = Column(String, nullable=True)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    embeddings = relationship("Embedding", back_populates="chunk", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<DocumentChunk {self.id}: Document {self.document_id}, Chunk {self.chunk_index}>"
