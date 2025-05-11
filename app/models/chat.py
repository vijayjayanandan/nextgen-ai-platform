from sqlalchemy import Column, String, ForeignKey, Text, Integer, Enum, Boolean
from sqlalchemy.dialects.postgresql import JSONB, UUID, ARRAY
from sqlalchemy.orm import relationship
import enum

from app.db.base import BaseModel


class MessageRole(str, enum.Enum):
    """Enum for message roles in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class Conversation(BaseModel):
    """
    Conversation model representing a chat session.
    """
    # The user who owns this conversation
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False, index=True)
    
    # Conversation metadata
    title = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    
    # Model information
    model_name = Column(String, nullable=False, index=True)
    model_params = Column(JSONB, default={}, nullable=False)
    
    # Conversation state and metadata
    is_active = Column(Boolean, default=True, nullable=False)
    meta_data = Column(JSONB, default={}, nullable=False)
    
    # Context information
    system_prompt = Column(Text, nullable=True)
    
    # Retrieved context tracking (for RAG conversations)
    retrieved_document_ids = Column(ARRAY(UUID(as_uuid=True)), default=[], nullable=False)
    
    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.sequence")
    
    def __repr__(self):
        return f"<Conversation {self.id}: User {self.user_id}>"


class Message(BaseModel):
    """
    Message model representing a single message in a conversation.
    """
    # Foreign key to the conversation this message belongs to
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversation.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Message content and metadata
    role = Column(Enum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    sequence = Column(Integer, nullable=False)  # Order in the conversation
    
    # For function calling
    function_name = Column(String, nullable=True)
    function_arguments = Column(JSONB, nullable=True)
    
    # Metadata
    meta_data = Column(JSONB, default={}, nullable=False)
    
    # Tokens count for billing/usage tracking
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    
    # LLM specific information
    model_name = Column(String, nullable=True)
    model_version = Column(String, nullable=True)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    # For tracking source documents used in retrieval augmentation
    source_documents = Column(ARRAY(UUID(as_uuid=True)), default=[], nullable=False)
    
    def __repr__(self):
        return f"<Message {self.id}: Conversation {self.conversation_id}, Role {self.role}, Sequence {self.sequence}>"
