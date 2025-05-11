from sqlalchemy import Boolean, Column, String, ARRAY
from sqlalchemy.dialects.postgresql import JSONB

from app.db.session import Base
from app.db.base import BaseModel


class User(BaseModel):
    """
    User model representing system users.
    """
    email = Column(String, index=True, unique=True, nullable=False)
    full_name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    
    # Store user roles as an array of strings
    roles = Column(ARRAY(String), default=[], nullable=False)
    
    # Azure AD integration fields
    azure_ad_id = Column(String, unique=True, nullable=True, index=True)
    
    # Department and other GC-specific fields
    department = Column(String, nullable=True)
    position = Column(String, nullable=True)
    
    # User preferences
    preferences = Column(JSONB, default={}, nullable=False)
    
    # User API keys for programmatic access
    api_keys = Column(JSONB, default=[], nullable=False)