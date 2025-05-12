from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union
import uuid

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import ValidationError

from app.core.config import settings
from app.db.session import get_db
from app.models.user import User
from app.core.logging import get_logger, audit_log
from app.schemas.user import UserInDB

logger = get_logger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 authentication scheme
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/token"  # Changed from auth/login to match your endpoint
)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password.
    
    Args:
        plain_password: The plain-text password
        hashed_password: The hashed password to compare against
        
    Returns:
        True if the password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Generate a hashed password.
    
    Args:
        password: The plain-text password to hash
        
    Returns:
        The hashed password
    """
    return pwd_context.hash(password)


def create_access_token(
    subject: Union[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        subject: The subject of the token (typically user ID)
        expires_delta: Optional expiration time. If not provided, uses the default from settings
        
    Returns:
        JWT token as string
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode = {"exp": expire, "sub": str(subject), "iat": datetime.utcnow()}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


async def authenticate_user(email: str, password: str, db: AsyncSession) -> Optional[User]:
    """
    Authenticate a user with email and password.
    
    Args:
        email: The user's email
        password: The user's password
        db: Database session
        
    Returns:
        User object if authentication is successful, None otherwise
    """
    try:
        # Find user by email
        result = await db.execute(select(User).filter(User.email == email))
        user = result.scalars().first()
        
        # Check if user exists and password is correct
        if not user or not verify_password(password, user.hashed_password):
            logger.warning(f"Failed login attempt for user: {email}")
            return None
        
        # Check if user is active
        if not user.is_active:
            logger.warning(f"Login attempt for inactive user: {email}")
            return None
        
        logger.info(f"User {email} authenticated successfully")
        return user
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        return None


async def get_current_user(
    db: AsyncSession = Depends(get_db), token: str = Depends(oauth2_scheme)
) -> UserInDB:
    """
    Validate the access token and get the current user.
    
    Args:
        db: Database session
        token: JWT token from the request
        
    Returns:
        Current user object
        
    Raises:
        HTTPException: If authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            logger.warning("Token without user ID")
            raise credentials_exception
    except JWTError:
        logger.warning("Invalid JWT token")
        raise credentials_exception
    
    try:
        # Fetch user from database
        user = await db.get(User, uuid.UUID(user_id))
        if user is None:
            logger.warning(f"User with ID {user_id} not found")
            raise credentials_exception
            
        # Check if user is active
        if not user.is_active:
            logger.warning(f"Inactive user {user_id} attempted to access the system")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Inactive user"
            )
            
        return UserInDB.model_validate(user)
    except (ValidationError, ValueError):
        logger.error(f"Invalid user data for ID {user_id}")
        raise credentials_exception


async def get_current_active_superuser(
    current_user: UserInDB = Depends(get_current_user),
) -> UserInDB:
    """
    Get the current user and verify they have superuser privileges.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Current user with superuser privileges
        
    Raises:
        HTTPException: If the user is not a superuser
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


def require_role(required_role: str):
    """
    Dependency factory that creates a dependency to check if a user has a specific role.
    
    Args:
        required_role: The role that's required to access a resource
        
    Returns:
        A dependency function that validates the user's role
    """
    async def role_checker(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
        if required_role not in current_user.roles:
            # Audit log unauthorized access attempts
            audit_log(
                user_id=str(current_user.id),
                action="unauthorized_access",
                resource_type="role_protected_endpoint",
                resource_id=required_role,
                details={"required_role": required_role, "user_roles": current_user.roles}
            )
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required"
            )
        return current_user
    
    return role_checker