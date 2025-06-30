# app/utils/llm_response_parser.py
"""
Robust LLM response parsing utilities for production-grade JSON handling.
Provides sanitization, validation, and fallback mechanisms for LLM outputs.
"""

import json
import re
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ParseResult(Enum):
    """Result status of JSON parsing attempt"""
    SUCCESS = "success"
    FALLBACK_USED = "fallback_used"
    VALIDATION_FAILED = "validation_failed"
    PARSE_FAILED = "parse_failed"


@dataclass
class LLMParseResponse:
    """Response object for LLM parsing operations"""
    data: Any
    status: ParseResult
    original_response: str
    sanitized_response: str
    error_message: Optional[str] = None
    fallback_used: bool = False


class LLMResponseParser:
    """Production-grade LLM response parser with robust error handling"""
    
    def __init__(self, enable_logging: bool = True):
        self.enable_logging = enable_logging
        self.parse_attempts = 0
        self.fallback_uses = 0
    
    def sanitize_response(self, response: str) -> str:
        """
        Sanitize LLM response to extract clean JSON.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Sanitized string ready for JSON parsing
        """
        if not response or not isinstance(response, str):
            return ""
        
        # Step 1: Basic cleanup
        sanitized = response.strip()
        
        # Step 2: Remove markdown code blocks
        # Handle ```json ... ``` or ``` ... ```
        sanitized = re.sub(r'```(?:json)?\s*\n?', '', sanitized)
        sanitized = re.sub(r'\n?\s*```', '', sanitized)
        
        # Step 3: Remove common LLM prefixes/suffixes
        prefixes_to_remove = [
            "Here's the JSON:",
            "Here is the JSON:",
            "The JSON response is:",
            "JSON:",
            "Response:",
            "Output:",
        ]
        
        for prefix in prefixes_to_remove:
            if sanitized.lower().startswith(prefix.lower()):
                sanitized = sanitized[len(prefix):].strip()
        
        # Step 4: Remove trailing explanations
        # Look for patterns like "This JSON represents..." after the closing brace
        json_end_pattern = r'(\}|\])\s*\n.*$'
        match = re.search(json_end_pattern, sanitized, re.DOTALL)
        if match:
            sanitized = sanitized[:match.end(1)]
        
        # Step 4.5: Handle incomplete JSON due to truncation
        # If we have an opening brace/bracket but no closing one, try to find the last complete structure
        if sanitized.count('{') > sanitized.count('}'):
            # Find the last complete JSON object
            last_brace = sanitized.rfind('}')
            if last_brace != -1:
                # Try to find a complete structure ending at this brace
                brace_count = 0
                for i in range(last_brace, -1, -1):
                    if sanitized[i] == '}':
                        brace_count += 1
                    elif sanitized[i] == '{':
                        brace_count -= 1
                        if brace_count == 0:
                            sanitized = sanitized[i:last_brace+1]
                            break
        
        if sanitized.count('[') > sanitized.count(']'):
            # Find the last complete JSON array
            last_bracket = sanitized.rfind(']')
            if last_bracket != -1:
                bracket_count = 0
                for i in range(last_bracket, -1, -1):
                    if sanitized[i] == ']':
                        bracket_count += 1
                    elif sanitized[i] == '[':
                        bracket_count -= 1
                        if bracket_count == 0:
                            sanitized = sanitized[i:last_bracket+1]
                            break
        
        # Step 5: Handle common encoding issues
        sanitized = sanitized.replace('"', '"').replace('"', '"')  # Smart quotes
        sanitized = sanitized.replace(''', "'").replace(''', "'")  # Smart apostrophes
        
        # Step 6: Ensure proper JSON structure
        sanitized = sanitized.strip()
        
        return sanitized
    
    def parse_json_robust(
        self,
        response: str,
        expected_type: type = dict,
        validator: Optional[Callable[[Any], bool]] = None,
        fallback_value: Any = None
    ) -> LLMParseResponse:
        """
        Robustly parse JSON from LLM response with multiple fallback strategies.
        
        Args:
            response: Raw LLM response
            expected_type: Expected Python type (dict, list, etc.)
            validator: Optional validation function
            fallback_value: Value to return if all parsing fails
            
        Returns:
            LLMParseResponse with parsed data and metadata
        """
        self.parse_attempts += 1
        original_response = response
        
        # Step 1: Sanitize the response
        sanitized = self.sanitize_response(response)
        
        if not sanitized:
            return LLMParseResponse(
                data=fallback_value,
                status=ParseResult.PARSE_FAILED,
                original_response=original_response,
                sanitized_response=sanitized,
                error_message="Empty response after sanitization",
                fallback_used=True
            )
        
        # Step 2: Attempt direct parsing
        try:
            parsed_data = json.loads(sanitized)
            
            # Type validation
            if not isinstance(parsed_data, expected_type):
                if self.enable_logging:
                    logger.warning(f"Parsed data type {type(parsed_data)} doesn't match expected {expected_type}")
                
                # Try to coerce if possible
                if expected_type == dict and isinstance(parsed_data, list) and len(parsed_data) > 0:
                    if isinstance(parsed_data[0], dict):
                        parsed_data = parsed_data[0]  # Take first dict from list
                elif expected_type == list and isinstance(parsed_data, dict):
                    parsed_data = [parsed_data]  # Wrap dict in list
                else:
                    raise ValueError(f"Cannot coerce {type(parsed_data)} to {expected_type}")
            
            # Custom validation
            if validator and not validator(parsed_data):
                return LLMParseResponse(
                    data=fallback_value,
                    status=ParseResult.VALIDATION_FAILED,
                    original_response=original_response,
                    sanitized_response=sanitized,
                    error_message="Custom validation failed",
                    fallback_used=True
                )
            
            # Success!
            if self.enable_logging:
                logger.debug(f"Successfully parsed JSON: {type(parsed_data)}")
            
            return LLMParseResponse(
                data=parsed_data,
                status=ParseResult.SUCCESS,
                original_response=original_response,
                sanitized_response=sanitized
            )
            
        except json.JSONDecodeError as e:
            if self.enable_logging:
                logger.warning(f"JSON decode error: {e}")
            
            # Step 3: Try fallback parsing strategies
            fallback_result = self._try_fallback_parsing(sanitized, expected_type)
            if fallback_result is not None:
                self.fallback_uses += 1
                return LLMParseResponse(
                    data=fallback_result,
                    status=ParseResult.FALLBACK_USED,
                    original_response=original_response,
                    sanitized_response=sanitized,
                    error_message=f"JSON parse failed, used fallback: {str(e)}",
                    fallback_used=True
                )
            
            # Step 4: Return fallback value
            self.fallback_uses += 1
            return LLMParseResponse(
                data=fallback_value,
                status=ParseResult.PARSE_FAILED,
                original_response=original_response,
                sanitized_response=sanitized,
                error_message=f"All parsing strategies failed: {str(e)}",
                fallback_used=True
            )
        
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Unexpected error during JSON parsing: {e}")
            
            self.fallback_uses += 1
            return LLMParseResponse(
                data=fallback_value,
                status=ParseResult.PARSE_FAILED,
                original_response=original_response,
                sanitized_response=sanitized,
                error_message=f"Unexpected parsing error: {str(e)}",
                fallback_used=True
            )
    
    def _try_fallback_parsing(self, sanitized: str, expected_type: type) -> Optional[Any]:
        """
        Try various fallback parsing strategies for malformed JSON.
        
        Args:
            sanitized: Sanitized response string
            expected_type: Expected return type
            
        Returns:
            Parsed data or None if all strategies fail
        """
        
        # Strategy 1: Fix common JSON issues
        try:
            # Fix trailing commas
            fixed = re.sub(r',(\s*[}\]])', r'\1', sanitized)
            
            # Fix single quotes to double quotes
            fixed = re.sub(r"'([^']*)':", r'"\1":', fixed)
            fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)
            
            return json.loads(fixed)
        except:
            pass
        
        # Strategy 2: Extract JSON-like patterns
        if expected_type == dict:
            # Look for key-value patterns
            try:
                # Extract content between first { and last }
                start = sanitized.find('{')
                end = sanitized.rfind('}')
                if start != -1 and end != -1 and end > start:
                    json_candidate = sanitized[start:end+1]
                    return json.loads(json_candidate)
            except:
                pass
        
        elif expected_type == list:
            # Look for array patterns
            try:
                # Extract content between first [ and last ]
                start = sanitized.find('[')
                end = sanitized.rfind(']')
                if start != -1 and end != -1 and end > start:
                    json_candidate = sanitized[start:end+1]
                    return json.loads(json_candidate)
            except:
                pass
        
        # Strategy 3: Regex-based extraction for specific patterns
        if expected_type == list:
            # Try to extract comma-separated values
            try:
                # Look for patterns like [1, 2, 3] or ["a", "b", "c"]
                numbers_match = re.search(r'\[([0-9,\s]+)\]', sanitized)
                if numbers_match:
                    numbers_str = numbers_match.group(1)
                    return [int(x.strip()) for x in numbers_str.split(',') if x.strip().isdigit()]
                
                strings_match = re.search(r'\[(["\'][^"\']*["\'][,\s]*)+\]', sanitized)
                if strings_match:
                    return json.loads(strings_match.group(0))
            except:
                pass
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parsing statistics for monitoring"""
        return {
            "total_attempts": self.parse_attempts,
            "fallback_uses": self.fallback_uses,
            "success_rate": (self.parse_attempts - self.fallback_uses) / max(self.parse_attempts, 1),
            "fallback_rate": self.fallback_uses / max(self.parse_attempts, 1)
        }


# Global parser instance
_global_parser = LLMResponseParser()


def parse_llm_json(
    response: str,
    expected_type: type = dict,
    validator: Optional[Callable[[Any], bool]] = None,
    fallback_value: Any = None
) -> LLMParseResponse:
    """
    Convenience function for robust LLM JSON parsing.
    
    Args:
        response: Raw LLM response
        expected_type: Expected Python type
        validator: Optional validation function
        fallback_value: Fallback value if parsing fails
        
    Returns:
        LLMParseResponse with parsed data and metadata
    """
    return _global_parser.parse_json_robust(
        response=response,
        expected_type=expected_type,
        validator=validator,
        fallback_value=fallback_value
    )


def sanitize_llm_response(response: str) -> str:
    """
    Convenience function for response sanitization.
    
    Args:
        response: Raw LLM response
        
    Returns:
        Sanitized response string
    """
    return _global_parser.sanitize_response(response)


def get_parser_stats() -> Dict[str, Any]:
    """Get global parser statistics"""
    return _global_parser.get_stats()


# Validation functions for common RAG node outputs
def validate_query_analysis(data: Dict[str, Any]) -> bool:
    """Validate query analysis JSON structure"""
    required_fields = ["query_type", "intent", "entities"]
    
    if not isinstance(data, dict):
        return False
    
    for field in required_fields:
        if field not in data:
            return False
    
    # Validate query_type values
    valid_types = ["simple", "conversational", "complex", "code_related"]
    if data["query_type"] not in valid_types:
        return False
    
    # Validate entities is a list
    if not isinstance(data["entities"], list):
        return False
    
    return True


def validate_reranking_indices(data: List[int]) -> bool:
    """Validate reranking indices structure"""
    if not isinstance(data, list):
        return False
    
    # All elements should be integers
    if not all(isinstance(x, int) for x in data):
        return False
    
    # Should have reasonable length
    if len(data) > 20:  # Sanity check
        return False
    
    return True
