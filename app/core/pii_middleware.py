from typing import Dict, List, Optional, Any, Tuple
import json
import time
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.services.pii import safe_quick_pii_check, submit_for_analysis, ProcessingPriority
from app.core.logging import get_logger, audit_log
from app.core.config import settings

logger = get_logger(__name__)


class PIIFilteringMiddleware(BaseHTTPMiddleware):
    """
    Middleware for real-time PII filtering on all incoming requests
    and outgoing responses in the FastAPI application.
    """
    
    def __init__(self, app, enable_filtering: bool = True):
        """
        Initialize PII filtering middleware.
        
        Args:
            app: FastAPI application instance
            enable_filtering: Whether to enable PII filtering
        """
        super().__init__(app)
        self.enable_filtering = enable_filtering
        # Fast PII screening will be handled by the new fast screener
        self.content_filter = None
        
        # Endpoints that should be filtered
        self.filtered_endpoints = {
            "/api/v1/chat",
            "/api/v1/chat/",
            "/api/v1/documents/upload",
            "/api/v1/documents/",
            "/api/v1/moderation/",
            "/api/v1/retrieval/",
        }
        
        # Content fields to filter in request bodies
        self.content_fields = {
            "content", "message", "text", "query", "prompt", 
            "description", "notes", "comment", "body"
        }
        
        logger.info(f"PII filtering middleware initialized (enabled: {enable_filtering})")
    
    async def dispatch(self, request: Request, call_next):
        """
        Process request and response through fast PII screening.
        Uses industry-standard fast path + background processing approach.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/endpoint in chain
            
        Returns:
            HTTP response with fast PII screening applied
        """
        # Skip filtering if disabled
        if not self.enable_filtering:
            return await call_next(request)
        
        # Check if this request should be filtered
        if not self._should_filter_request(request):
            return await call_next(request)
        
        start_time = time.perf_counter()
        
        try:
            # Get user ID for logging
            user_id = getattr(request.state, 'user_id', 'anonymous')
            
            # Read request body for fast screening (non-blocking)
            body = await request.body()
            if body:
                try:
                    body_json = json.loads(body)
                    content_to_screen = self._extract_content_for_screening(body_json)
                    
                    if content_to_screen and len(content_to_screen.strip()) > 0:
                        # Fast PII screening (< 5ms target) - with timeout protection
                        try:
                            screening_result = await safe_quick_pii_check(content_to_screen)
                            
                            # Log fast screening result
                            logger.debug(
                                f"Fast PII screening: {screening_result.processing_time_ms:.2f}ms, "
                                f"critical_pii={screening_result.has_critical_pii}"
                            )
                            
                            # Block request if critical PII detected
                            if screening_result.should_block:
                                logger.warning(
                                    f"Blocking request due to critical PII detection: "
                                    f"types={[t.value for t in screening_result.detected_types]}"
                                )
                                
                                return JSONResponse(
                                    status_code=400,
                                    content={
                                        "error": "Content contains sensitive information that cannot be processed",
                                        "code": "PII_DETECTED",
                                        "details": "Please remove any personal information and try again"
                                    }
                                )
                            
                            # Submit for background comprehensive analysis (fire and forget)
                            try:
                                task_id = await submit_for_analysis(
                                    content=content_to_screen,
                                    user_id=user_id,
                                    endpoint=request.url.path,
                                    priority=ProcessingPriority.MEDIUM,
                                    context={
                                        "method": request.method,
                                        "fast_screening_result": {
                                            "has_critical_pii": screening_result.has_critical_pii,
                                            "detected_types": [t.value for t in screening_result.detected_types]
                                        }
                                    }
                                )
                                request.state.pii_analysis_task_id = task_id
                            except Exception as bg_error:
                                # Don't block request if background submission fails
                                logger.warning(f"Background PII analysis submission failed: {bg_error}")
                        
                        except Exception as screen_error:
                            # Don't block request if screening fails
                            logger.warning(f"Fast PII screening failed: {screen_error}")
                
                except json.JSONDecodeError:
                    # Not JSON content, skip PII screening
                    pass
                except Exception as parse_error:
                    # Don't block request if parsing fails
                    logger.warning(f"Request parsing failed: {parse_error}")
            
            # Process request normally (fast path completed)
            response = await call_next(request)
            
            # Log performance metrics
            total_time = (time.perf_counter() - start_time) * 1000
            logger.debug(f"PII middleware total time: {total_time:.2f}ms")
            
            return response
            
        except Exception as e:
            # Log error but don't block request (fail open for availability)
            logger.error(f"PII middleware error: {e}")
            return await call_next(request)
    
    def _should_filter_request(self, request: Request) -> bool:
        """
        Determine if request should be filtered based on path and method.
        
        Args:
            request: HTTP request
            
        Returns:
            True if request should be filtered
        """
        # Only filter specific HTTP methods
        if request.method not in ["POST", "PUT", "PATCH"]:
            return False
        
        # Check if path matches filtered endpoints
        path = request.url.path
        return any(endpoint in path for endpoint in self.filtered_endpoints)
    
    def _extract_content_for_screening(self, data: Dict[str, Any]) -> str:
        """
        Extract content from request body for fast PII screening.
        
        Args:
            data: Parsed JSON request body
            
        Returns:
            Combined content string for screening
        """
        content_parts = []
        
        # Extract content from known fields
        for key, value in data.items():
            if isinstance(value, str) and key.lower() in self.content_fields:
                content_parts.append(value)
            elif isinstance(value, dict):
                # Recursively extract from nested objects
                nested_content = self._extract_content_for_screening(value)
                if nested_content:
                    content_parts.append(nested_content)
            elif isinstance(value, list):
                # Extract from list items
                for item in value:
                    if isinstance(item, str):
                        content_parts.append(item)
                    elif isinstance(item, dict):
                        nested_content = self._extract_content_for_screening(item)
                        if nested_content:
                            content_parts.append(nested_content)
        
        # Combine all content with space separator
        combined_content = " ".join(content_parts)
        
        # Limit content length for fast screening (performance optimization)
        max_length = 5000
        if len(combined_content) > max_length:
            combined_content = combined_content[:max_length]
        
        return combined_content
    
    # Note: Old filtering methods removed - now using fast screening approach
    # The new system uses fast screening + background processing instead of
    # synchronous filtering which was causing performance issues


def create_pii_middleware(enable_filtering: bool = None) -> PIIFilteringMiddleware:
    """
    Factory function to create PII filtering middleware.
    
    Args:
        enable_filtering: Override for enabling filtering (uses settings if None)
        
    Returns:
        Configured PII filtering middleware
    """
    if enable_filtering is None:
        enable_filtering = getattr(settings, 'ENABLE_PII_FILTERING', True)
    
    return PIIFilteringMiddleware(None, enable_filtering)
