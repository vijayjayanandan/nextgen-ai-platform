from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.core.config import settings
from app.core.security import get_current_user, require_role
from app.core.logging import get_logger, audit_log
from app.schemas.user import UserInDB
from app.services.moderation.content_filter import ContentFilter

router = APIRouter()
logger = get_logger(__name__)


class ContentCheckRequest(BaseModel):
    """Request model for content moderation."""
    content: str
    context: Optional[Dict[str, Any]] = None


class ContentFilterRequest(BaseModel):
    """Request model for content filtering."""
    content: str
    context: Optional[Dict[str, Any]] = None


@router.post("/check", response_model=Dict[str, Any])
async def check_content(
    request: ContentCheckRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Check if content complies with content policies.
    
    Returns flags for any policy violations detected.
    """
    try:
        # Initialize content filter
        content_filter = ContentFilter()
        
        # Check the content
        is_allowed, details = await content_filter.check_content(
            request.content,
            str(current_user.id),
            request.context
        )
        
        # Format response
        response = {
            "is_allowed": is_allowed,
            "flags": details.get("flags", []),
            "filtered": details.get("filtered", False)
        }
        
        # Log the moderation check
        if not is_allowed:
            audit_log(
                user_id=str(current_user.id),
                action="content_moderation",
                resource_type="moderation",
                resource_id="check",
                details={
                    "is_allowed": is_allowed,
                    "flags": details.get("flags", []),
                    "content_length": len(request.content),
                    "context": request.context or {}
                }
            )
        
        return response
    except Exception as e:
        logger.error(f"Error checking content: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking content: {str(e)}"
        )


@router.post("/filter", response_model=Dict[str, Any])
async def filter_content(
    request: ContentFilterRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Filter content to remove or redact sensitive information.
    """
    try:
        # Initialize content filter
        content_filter = ContentFilter()
        
        # Filter the content
        filtered_content, details = await content_filter.filter_prompt(
            request.content,
            str(current_user.id),
            request.context
        )
        
        # Format response
        response = {
            "original_content": request.content,
            "filtered_content": filtered_content,
            "was_filtered": filtered_content != request.content,
            "flags": details.get("flags", [])
        }
        
        # Log the filtering
        if filtered_content != request.content:
            audit_log(
                user_id=str(current_user.id),
                action="content_filtering",
                resource_type="moderation",
                resource_id="filter",
                details={
                    "was_filtered": filtered_content != request.content,
                    "flags": details.get("flags", []),
                    "original_length": len(request.content),
                    "filtered_length": len(filtered_content),
                    "context": request.context or {}
                }
            )
        
        return response
    except Exception as e:
        logger.error(f"Error filtering content: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error filtering content: {str(e)}"
        )


class ModerateResponseRequest(BaseModel):
    """Request model for moderating model responses."""
    content: str
    model: str
    context: Optional[Dict[str, Any]] = None


@router.post("/moderate-response", response_model=Dict[str, Any])
async def moderate_response(
    request: ModerateResponseRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Moderate an AI-generated response.
    
    This endpoint is used to check and filter model responses.
    """
    try:
        # Initialize content filter
        content_filter = ContentFilter()
        
        # Add model info to context
        context = request.context or {}
        context["model"] = request.model
        
        # Filter the response
        filtered_content, details = await content_filter.filter_response(
            request.content,
            str(current_user.id),
            context
        )
        
        # Format response
        response = {
            "original_content": request.content,
            "filtered_content": filtered_content,
            "was_filtered": filtered_content != request.content,
            "flags": details.get("flags", []),
            "fully_filtered": details.get("fully_filtered", False)
        }
        
        # Log the moderation
        if filtered_content != request.content:
            audit_log(
                user_id=str(current_user.id),
                action="response_moderation",
                resource_type="moderation",
                resource_id="moderate",
                details={
                    "was_filtered": filtered_content != request.content,
                    "flags": details.get("flags", []),
                    "model": request.model,
                    "original_length": len(request.content),
                    "filtered_length": len(filtered_content),
                    "fully_filtered": details.get("fully_filtered", False)
                }
            )
        
        return response
    except Exception as e:
        logger.error(f"Error moderating response: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error moderating response: {str(e)}"
        )