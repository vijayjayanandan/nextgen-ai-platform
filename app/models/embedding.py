from sqlalchemy import Column, Float, ForeignKey, String, Integer, LargeBinary
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship

from app.db.base import BaseModel


class Embedding(BaseModel):
    """
    Embedding model for storing vector representations of document chunks.
    """
    # Foreign key to the document chunk this embedding represents
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("document_chunk.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Model information
    model_name = Column(String, nullable=False, index=True)
    model_version = Column(String, nullable=False)
    
    # Vector dimensions
    dimensions = Column(Integer, nullable=False)
    
    # The embedding vector itself
    # For PostgreSQL, we can use the ARRAY type
    # For production with many vectors, consider using pgvector extension or a dedicated vector DB
    vector = Column(ARRAY(Float), nullable=False)
    
    # For binary storage option (e.g., for models that use binary representations)
    # vector_binary = Column(LargeBinary, nullable=True)
    
    # Vector database reference (if stored in external vector DB)
    vector_db_id = Column(String, nullable=True, index=True)
    
    # Relationships
    chunk = relationship("DocumentChunk", back_populates="embeddings")
    
    def __repr__(self):
        return f"<Embedding {self.id}: Chunk {self.chunk_id}, Model {self.model_name}>"