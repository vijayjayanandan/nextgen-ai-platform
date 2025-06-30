"""
Core dependencies for the FastAPI application.
"""

from typing import Dict, Any, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """
    Get current user from authorization token.
    For now, returns a mock user for testing purposes.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        User information dictionary
    """
    # For testing purposes, return a mock user
    # In production, this would validate the JWT token and return real user info
    if credentials and credentials.credentials:
        return {
            "user_id": "test-user",
            "username": "test",
            "email": "test@example.com",
            "roles": ["user"]
        }
    else:
        # For monitoring endpoints, allow anonymous access during testing
        return {
            "user_id": "anonymous",
            "username": "anonymous",
            "email": "anonymous@example.com",
            "roles": ["anonymous"]
        }


async def get_current_active_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current active user.
    
    Args:
        current_user: Current user from get_current_user
        
    Returns:
        Active user information
    """
    # In production, check if user is active/enabled
    return current_user


def require_admin_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Require admin user for protected endpoints.
    
    Args:
        current_user: Current user from get_current_user
        
    Returns:
        Admin user information
        
    Raises:
        HTTPException: If user is not admin
    """
    if "admin" not in current_user.get("roles", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user
