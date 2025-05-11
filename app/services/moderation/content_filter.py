from typing import Dict, List, Optional, Any, Set, Tuple
import re
import httpx
from fastapi import HTTPException

from app.core.config import settings
from app.core.logging import get_logger, audit_log

logger = get_logger(__name__)


class ContentFilter:
    """
    Service for filtering content to ensure it meets ethical guidelines
    and policy requirements.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None
    ):
        """
        Initialize the content filter.
        
        Args:
            api_key: API key for OpenAI moderation API
            api_base: Base URL for OpenAI API
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.api_base = api_base or settings.OPENAI_API_BASE
        
        # Keywords and patterns that should trigger moderation
        self.sensitive_keywords = [
            # PII
            r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",  # SSN/SIN pattern
            r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Credit card pattern
            r"\b(?:[A-Za-z]{2}\d{6}|\d{9})\b",  # Passport number pattern
            
            # Security classification keywords
            r"\b(?:protected|confidential|classified|secret|top secret)\s+[ab]\b",
            
            # Harmful content keywords
            "bomb-making", "terrorist", "assassination", "suicide",
            
            # Bias, discrimination, and offensive content
            "racial slur", "sexist", "homophobic", "transphobic",
            
            # Policy violations
            "illegal immigration", "bypass security"
        ]
    
    async def check_content(
        self,
        content: str,
        user_id: str,
        context: Dict[str, Any] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check content against moderation policies.
        
        Args:
            content: Text content to check
            user_id: ID of the user submitting the content
            context: Additional context for the moderation check
            
        Returns:
            Tuple of (is_allowed, details) where details include flags and reasons
        """
        # Skip if content filtering is disabled
        if not settings.ENABLE_CONTENT_FILTERING:
            return True, {"flags": [], "filtered": False}
        
        # Check for sensitive patterns
        pattern_flags = self._check_patterns(content)
        
        # Use OpenAI moderation API for more sophisticated checks
        api_flags = await self._check_with_moderation_api(content)
        
        # Combine flags
        all_flags = pattern_flags + api_flags
        
        # Determine if content should be allowed
        is_allowed = len(all_flags) == 0
        
        # If not allowed, log the violation
        if not is_allowed:
            details = {
                "flags": all_flags,
                "filtered": True,
                "content_length": len(content),
                "context": context or {}
            }
            
            # Log the moderation action
            logger.warning(
                f"Content filtered: {len(all_flags)} flags raised",
                extra={"flags": all_flags, "user_id": user_id}
            )
            
            # Create an audit log entry
            audit_log(
                user_id=user_id,
                action="content_filtered",
                resource_type="content",
                resource_id="moderation",
                details=details
            )
            
            return False, details
        
        return True, {"flags": [], "filtered": False}
    
    def _check_patterns(self, content: str) -> List[str]:
        """
        Check content against predefined patterns and keywords.
        
        Args:
            content: Text content to check
            
        Returns:
            List of flags raised by the content
        """
        flags = []
        
        # Check each pattern
        for pattern in self.sensitive_keywords:
            if pattern.startswith('r"\\b') or pattern.startswith("r'\\b"):
                # Regex pattern
                if re.search(pattern, content, re.IGNORECASE):
                    flags.append(f"pattern_match:{pattern}")
            else:
                # Simple keyword
                if pattern.lower() in content.lower():
                    flags.append(f"keyword_match:{pattern}")
        
        return flags
    
    async def _check_with_moderation_api(self, content: str) -> List[str]:
        """
        Check content using OpenAI's moderation API.
        
        Args:
            content: Text content to check
            
        Returns:
            List of flags raised by the moderation API
        """
        url = f"{self.api_base}/moderations"
        
        # Skip API check if no key
        if not self.api_key:
            logger.warning("Skipping moderation API check: No API key provided")
            return []
        
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    url,
                    json={"input": content},
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Moderation API error: {response.status_code} - {response.text}")
                    return []
                
                result = response.json()
                
                # Extract flags
                flags = []
                results = result.get("results", [])
                
                if results and len(results) > 0:
                    # Check if flagged
                    if results[0].get("flagged", False):
                        # Extract specific categories
                        categories = results[0].get("categories", {})
                        for category, flagged in categories.items():
                            if flagged:
                                flags.append(f"moderation_api:{category}")
                
                return flags
                
        except Exception as e:
            logger.error(f"Error calling moderation API: {str(e)}")
            return []
    
    async def filter_prompt(
        self,
        prompt: str,
        user_id: str,
        context: Dict[str, Any] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Filter a user prompt to remove or redact sensitive information.
        
        Args:
            prompt: User prompt to filter
            user_id: ID of the user submitting the prompt
            context: Additional context for the filtering
            
        Returns:
            Tuple of (filtered_prompt, details)
        """
        # Check if filtering is needed
        is_allowed, details = await self.check_content(prompt, user_id, context)
        
        if not is_allowed:
            # Apply filtering based on flags
            filtered_prompt = prompt
            
            # Redact PII
            pii_patterns = [
                # SSN/SIN
                (r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b", "[REDACTED_ID_NUMBER]"),
                # Credit card
                (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[REDACTED_CARD_NUMBER]"),
                # Passport number
                (r"\b(?:[A-Za-z]{2}\d{6}|\d{9})\b", "[REDACTED_PASSPORT_NUMBER]"),
                # Email addresses
                (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[REDACTED_EMAIL]"),
                # Phone numbers
                (r"\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b", "[REDACTED_PHONE]")
            ]
            
            for pattern, replacement in pii_patterns:
                filtered_prompt = re.sub(pattern, replacement, filtered_prompt)
            
            # Return filtered prompt
            details["original_length"] = len(prompt)
            details["filtered_length"] = len(filtered_prompt)
            
            return filtered_prompt, details
        
        return prompt, details
    
    async def filter_response(
        self,
        response: str,
        user_id: str,
        context: Dict[str, Any] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Filter a model response to remove content that violates policies.
        
        Args:
            response: Model response to filter
            user_id: ID of the user
            context: Additional context for the filtering
            
        Returns:
            Tuple of (filtered_response, details)
        """
        # Check if filtering is needed
        is_allowed, details = await self.check_content(response, user_id, context)
        
        if not is_allowed:
            # Simple filtering strategy - redact with explanation
            if any("moderation_api:" in flag for flag in details["flags"]):
                # More serious violation detected by moderation API
                filtered_response = (
                    "I apologize, but the generated response contained content that "
                    "violates our content policy. Please rephrase your request "
                    "or contact support if you believe this is an error."
                )
            else:
                # Less serious violation - try to redact only sensitive parts
                filtered_response = response
                
                # Redact PII (same as in filter_prompt)
                pii_patterns = [
                    # SSN/SIN
                    (r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b", "[REDACTED_ID_NUMBER]"),
                    # Credit card
                    (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[REDACTED_CARD_NUMBER]"),
                    # Passport number
                    (r"\b(?:[A-Za-z]{2}\d{6}|\d{9})\b", "[REDACTED_PASSPORT_NUMBER]"),
                    # Email addresses
                    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[REDACTED_EMAIL]"),
                    # Phone numbers
                    (r"\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b", "[REDACTED_PHONE]")
                ]
                
                for pattern, replacement in pii_patterns:
                    filtered_response = re.sub(pattern, replacement, filtered_response)
            
            # Return filtered response
            details["original_length"] = len(response)
            details["filtered_length"] = len(filtered_response)
            details["fully_filtered"] = filtered_response != response
            
            return filtered_response, details
        
        return response, details