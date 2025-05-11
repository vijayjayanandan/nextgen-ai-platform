from typing import List, Dict, Optional, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field


class UserBase(BaseModel):
    """Base user schema with common attributes."""
    email: EmailStr
    full_name: Optional[str] = None
    is_active: bool = True
    department: Optional[str] = None
    position: Optional[str] = None
    roles: List[str] = []


class UserCreate(UserBase):
    """Schema for user creation."""
    password: str = Field(..., min_length=8)
    is_superuser: bool = False


class UserUpdate(BaseModel):
    """Schema for user updates."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None
    department: Optional[str] = None
    position: Optional[str] = None
    roles: Optional[List[str]] = None
    preferences: Optional[Dict[str, Any]] = None


class UserInDBBase(UserBase):
    """Base schema for users in the database."""
    id: UUID
    created_at: datetime
    updated_at: datetime
    is_superuser: bool
    preferences: Dict[str, Any] = {}
    
    class Config:
        from_attributes = True


class UserInDB(UserInDBBase):
    """Schema for user with sensitive data (like hashed_password)."""
    hashed_password: str
    azure_ad_id: Optional[str] = None


class User(UserInDBBase):
    """Schema for user without sensitive data."""
    pass


class UserResponse(User):
    """Schema for user response."""
    pass


class UserWithAPIKeys(User):
    """Schema for user including API keys."""
    api_keys: List[Dict[str, Any]] = []


class Token(BaseModel):
    """Schema for authentication token."""
    access_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    """Schema for token payload."""
    sub: Optional[str] = None
    exp: Optional[int] = None
