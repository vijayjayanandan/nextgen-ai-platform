from typing import Dict, List, Optional, Any
import json
import time
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.services.moderation.enhanced_content_filter import EnterpriseContentFilter, FilterResult
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
        self.content_filter = EnterpriseContentFilter() if enable_filtering else None
        
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
        Process request and response through PII filtering.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/endpoint in chain
            
        Returns:
            HTTP response with PII filtering applied
        """
        start_time = time.time()
        
        # Skip filtering if disabled or not applicable
        if not self.enable_filtering or not self._should_filter_request(request):
            return await call_next(request)
        
        try:
            # 1. Filter incoming request
            filtered_request = await self._filter_request(request)
            
            # 2. Process request through application
            response = await call_next(filtered_request)
            
            # 3. Filter outgoing response
            filtered_response = await self._filter_response(response, request)
            
            # 4. Log middleware performance
            processing_time = (time.time() - start_time) * 1000
            logger.debug(f"PII middleware processing time: {processing_time:.2f}ms")
            
            return filtered_response
            
        except Exception as e:
            logger.error(f"Error in PII filtering middleware: {str(e)}")
            # Continue without filtering on error
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
    
    async def _filter_request(self, request: Request) -> Request:
        """
        Filter PII from incoming request body.
        
        Args:
            request: Original HTTP request
            
        Returns:
            Request with PII filtered from body
        """
        # Read request body
        body = await request.body()
        
        if not body:
            return request
        
        try:
            # Parse JSON body
            body_json = json.loads(body)
            
            # Get user ID for audit logging
            user_id = getattr(request.state, 'user_id', 'anonymous')
            
            # Filter content fields
            filtered_body, pii_detected = await self._filter_json_content(
                body_json, user_id, "request"
            )
            
            # Update request body if PII was detected
            if pii_detected:
                # Store original body for audit
                request.state.original_body = body_json
                request.state.pii_filtered_request = True
                
                # Replace request body
                new_body = json.dumps(filtered_body).encode()
                request._body = new_body
                
                # Update content-length header
                request.headers.__dict__["_list"] = [
                    (name, value) for name, value in request.headers.items()
                    if name.lower() != "content-length"
                ]
                request.headers.__dict__["_list"].append(
                    (b"content-length", str(len(new_body)).encode())
                )
            
            return request
            
        except json.JSONDecodeError:
            # Not JSON content, skip filtering
            return request
        except Exception as e:
            logger.error(f"Error filtering request: {str(e)}")
            return request
    
    async def _filter_response(self, response: Response, request: Request) -> Response:
        """
        Filter PII from outgoing response.
        
        Args:
            response: Original HTTP response
            request: Original HTTP request
            
        Returns:
            Response with PII filtered
        """
        # Only filter JSON responses
        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            return response
        
        try:
            # Read response body
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            
            if not response_body:
                return response
            
            # Parse JSON response
            response_json = json.loads(response_body)
            
            # Get user ID for audit logging
            user_id = getattr(request.state, 'user_id', 'anonymous')
            
            # Filter content fields
            filtered_response, pii_detected = await self._filter_json_content(
                response_json, user_id, "response"
            )
            
            # Create new response if PII was detected
            if pii_detected:
                # Log PII detection in response
                logger.warning(
                    f"PII detected in response for user {user_id}",
                    extra={"endpoint": request.url.path, "method": request.method}
                )
                
                return JSONResponse(
                    content=filtered_response,
                    status_code=response.status_code,
                    headers=dict(response.headers)
                )
            
            # Return original response if no PII detected
            return JSONResponse(
                content=response_json,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
            
        except json.JSONDecodeError:
            # Not JSON content, return original
            return response
        except Exception as e:
            logger.error(f"Error filtering response: {str(e)}")
            return response
    
    async def _filter_json_content(
        self, 
        data: Dict[str, Any], 
        user_id: str, 
        direction: str
    ) -> tuple[Dict[str, Any], bool]:
        """
        Filter PII from JSON data structure.
        
        Args:
            data: JSON data to filter
            user_id: User ID for audit logging
            direction: "request" or "response"
            
        Returns:
            Tuple of (filtered_data, pii_detected)
        """
        filtered_data = data.copy()
        pii_detected = False
        
        # Recursively filter content fields
        for key, value in data.items():
            if isinstance(value, str) and key.lower() in self.content_fields:
                # Filter string content
                filter_result = await self.content_filter.enhanced_filter(
                    content=value,
                    user_id=user_id,
                    context={"direction": direction, "field": key}
                )
                
                if filter_result.pii_detected:
                    filtered_data[key] = filter_result.filtered_content
                    pii_detected = True
                    
                    # Log PII detection
                    await self._log_pii_detection(
                        user_id=user_id,
                        field=key,
                        direction=direction,
                        filter_result=filter_result
                    )
            
            elif isinstance(value, dict):
                # Recursively filter nested objects
                nested_filtered, nested_pii = await self._filter_json_content(
                    value, user_id, direction
                )
                if nested_pii:
                    filtered_data[key] = nested_filtered
                    pii_detected = True
            
            elif isinstance(value, list):
                # Filter list items
                filtered_list = []
                for item in value:
                    if isinstance(item, str):
                        filter_result = await self.content_filter.enhanced_filter(
                            content=item,
                            user_id=user_id,
                            context={"direction": direction, "field": f"{key}[]"}
                        )
                        
                        if filter_result.pii_detected:
                            filtered_list.append(filter_result.filtered_content)
                            pii_detected = True
                        else:
                            filtered_list.append(item)
                    
                    elif isinstance(item, dict):
                        nested_filtered, nested_pii = await self._filter_json_content(
                            item, user_id, direction
                        )
                        filtered_list.append(nested_filtered)
                        if nested_pii:
                            pii_detected = True
                    else:
                        filtered_list.append(item)
                
                if pii_detected:
                    filtered_data[key] = filtered_list
        
        return filtered_data, pii_detected
    
    async def _log_pii_detection(
        self,
        user_id: str,
        field: str,
        direction: str,
        filter_result: FilterResult
    ):
        """
        Log PII detection for audit and compliance.
        
        Args:
            user_id: User ID
            field: Field name where PII was detected
            direction: "request" or "response"
            filter_result: Filtering result details
        """
        # Log warning for PII detection
        logger.warning(
            f"PII detected in {direction} field '{field}'",
            extra={
                "user_id": user_id,
                "field": field,
                "direction": direction,
                "pii_types": [d.entity_type.value for d in filter_result.detections],
                "risk_score": filter_result.risk_score,
                "anonymization_applied": filter_result.anonymization_applied
            }
        )
        
        # Create audit log entry
        audit_log(
            user_id=user_id,
            action="pii_middleware_detection",
            resource_type="content",
            resource_id=f"{direction}_{field}",
            details={
                "direction": direction,
                "field": field,
                "pii_detections_count": len(filter_result.detections),
                "pii_types": [d.entity_type.value for d in filter_result.detections],
                "risk_levels": [d.risk_level.value for d in filter_result.detections],
                "overall_risk_score": filter_result.risk_score,
                "anonymization_applied": filter_result.anonymization_applied,
                "processing_time_ms": filter_result.processing_time_ms,
                "original_length": filter_result.original_length,
                "filtered_length": filter_result.filtered_length
            }
        )


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
