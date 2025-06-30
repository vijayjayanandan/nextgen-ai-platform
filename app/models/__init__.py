"""
Database models for the NextGen AI Platform.

This module contains all SQLAlchemy models for the application.
"""

from .user import User
from .chat import Conversation, Message, MessageRole
from .document import Document, DocumentChunk, DocumentSourceType, DocumentStatus
from .embedding import Embedding
from .model import Model, ModelVersion, ModelType, ModelProvider, ModelDeploymentType, ModelStatus
from .audit import AuditLog, AuditActionType, AuditResourceType

__all__ = [
    # User models
    "User",
    
    # Chat models
    "Conversation",
    "Message", 
    "MessageRole",
    
    # Document models
    "Document",
    "DocumentChunk",
    "DocumentSourceType",
    "DocumentStatus",
    
    # Embedding models
    "Embedding",
    
    # Model management
    "Model",
    "ModelVersion",
    "ModelType",
    "ModelProvider", 
    "ModelDeploymentType",
    "ModelStatus",
    
    # Audit models
    "AuditLog",
    "AuditActionType",
    "AuditResourceType",
]
