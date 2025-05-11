from sqlalchemy import Column, String, Text, Enum, Boolean
from sqlalchemy.dialects.postgresql import JSONB, UUID
import enum

from app.db.base import BaseModel


class AuditActionType(str, enum.Enum):
    """Enum for audit action types."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    INFERENCE = "inference"
    EMBEDDING = "embedding"
    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_PROCESS = "document_process"
    MODEL_TRAINING = "model_training"
    MODEL_DEPLOYMENT = "model_deployment"
    ERROR = "error"
    OTHER = "other"


class AuditResourceType(str, enum.Enum):
    """Enum for audit resource types."""
    USER = "user"
    DOCUMENT = "document"
    CONVERSATION = "conversation"
    MESSAGE = "message"
    MODEL = "model"
    SYSTEM = "system"
    ENDPOINT = "endpoint"
    OTHER = "other"


class AuditLog(BaseModel):
    """
    AuditLog model for comprehensive logging of system actions.
    """
    # Who performed the action
    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)  # Nullable for system actions
    user_email = Column(String, nullable=True)
    user_roles = Column(JSONB, default=[], nullable=False)
    
    # What action was performed
    action_type = Column(Enum(AuditActionType), nullable=False, index=True)
    action_detail = Column(String, nullable=True)
    
    # On what resource
    resource_type = Column(Enum(AuditResourceType), nullable=False, index=True)
    resource_id = Column(String, nullable=True, index=True)
    
    # Result of the action
    status = Column(String, nullable=False)  # "success", "failure", etc.
    error_message = Column(Text, nullable=True)
    
    # Additional context
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    session_id = Column(String, nullable=True)
    
    # Detailed information
    request_data = Column(JSONB, nullable=True)
    response_data = Column(JSONB, nullable=True)
    
    # For security events
    security_relevant = Column(Boolean, default=False, nullable=False, index=True)
    
    def __repr__(self):
        return f"<AuditLog {self.id}: {self.action_type} on {self.resource_type} by {self.user_id}>"