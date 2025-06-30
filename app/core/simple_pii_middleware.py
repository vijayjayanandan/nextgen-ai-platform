"""
Simplified PII middleware for testing without background processing dependencies.
"""

import json
import time
import re
from typing import Dict, Any
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_logger

logger = get_logger(__name__)


class SimplePIIMiddleware(BaseHTTPMiddleware):
    """
    Simplified PII filtering middleware for testing.
    Only does basic regex-based PII detection without background processing.
    """
    
    def __init__(self, app, enable_filtering: bool = True):
        super().__init__(app)
        self.enable_filtering = enable_filtering
        
        # Simple regex patterns for critical PII
        self.pii_patterns = {
            'sin': re.compile(r'\b\d{3}[-\s]?\d{3}[-\s]?\d{3}\b'),
            'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            'ircc': re.compile(r'\bIRCC[-\s]?\d{8,10}\b', re.IGNORECASE),
            'uci': re.compile(r'\bUCI[-\s]?\d{8,10}\b', re.IGNORECASE),
        }
        
        # Endpoints to filter
        self.filtered_endpoints = {"/api/v1/chat"}
        
        logger.info(f"Simple PII middleware initialized (enabled: {enable_filtering})")
    
    async def dispatch(self, request: Request, call_next):
        """Simple PII filtering without background processing."""
        
        if not self.enable_filtering:
            return await call_next(request)
        
        # Only filter POST requests to chat endpoints
        if request.method != "POST" or not any(endpoint in str(request.url.path) for endpoint in self.filtered_endpoints):
            return await call_next(request)
        
        start_time = time.perf_counter()
        
        try:
            # Read request body
            body = await request.body()
            if body:
                try:
                    body_json = json.loads(body)
                    content = self._extract_content(body_json)
                    
                    if content and self._contains_pii(content):
                        logger.warning(f"PII detected in request to {request.url.path}")
                        return JSONResponse(
                            status_code=400,
                            content={
                                "error": "Content contains sensitive information",
                                "code": "PII_DETECTED"
                            }
                        )
                
                except json.JSONDecodeError:
                    pass  # Not JSON, skip filtering
            
            # Process request normally
            response = await call_next(request)
            
            # Log performance
            total_time = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Simple PII middleware: {total_time:.2f}ms")
            
            return response
            
        except Exception as e:
            logger.error(f"Simple PII middleware error: {e}")
            return await call_next(request)
    
    def _extract_content(self, data: Dict[str, Any]) -> str:
        """Extract content from request body."""
        content_parts = []
        
        for key, value in data.items():
            if isinstance(value, str) and key.lower() in ['content', 'message', 'text', 'query', 'prompt']:
                content_parts.append(value)
            elif isinstance(value, dict):
                nested = self._extract_content(value)
                if nested:
                    content_parts.append(nested)
        
        return " ".join(content_parts)
    
    def _contains_pii(self, content: str) -> bool:
        """Check if content contains PII using simple regex patterns."""
        for pii_type, pattern in self.pii_patterns.items():
            if pattern.search(content):
                logger.debug(f"PII detected: {pii_type}")
                return True
        return False
