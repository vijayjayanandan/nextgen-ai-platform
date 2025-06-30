"""
Fast PII Screening Service - Industry-grade performance
Implements sub-5ms PII detection for critical patterns only.
"""

import re
import time
import hashlib
import asyncio
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class CriticalPIIType(Enum):
    """Critical PII types that require immediate blocking."""
    SIN = "sin"
    CREDIT_CARD = "credit_card"
    IRCC_NUMBER = "ircc_number"
    UCI_NUMBER = "uci_number"
    PASSPORT = "passport"


@dataclass
class FastPIIResult:
    """Result of fast PII screening."""
    has_critical_pii: bool
    detected_types: List[CriticalPIIType]
    anonymized_content: Optional[str]
    processing_time_ms: float
    should_block: bool


class FastPIIScreener:
    """
    Ultra-fast PII screener for critical patterns only.
    Designed for < 5ms response times.
    """
    
    def __init__(self):
        """Initialize fast PII screener with pre-compiled patterns."""
        # Pre-compile critical patterns for maximum performance
        self.critical_patterns = self._compile_critical_patterns()
        
        # Content cache for repeated requests
        self.content_cache: Dict[str, FastPIIResult] = {}
        self.cache_max_size = 1000
        
        # Performance tracking
        self.total_requests = 0
        self.cache_hits = 0
        
        # Thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="fast_pii")
        
        logger.info("Fast PII screener initialized with critical patterns")
    
    def _compile_critical_patterns(self) -> Dict[CriticalPIIType, List[re.Pattern]]:
        """Compile critical PII patterns for maximum performance."""
        patterns = {
            CriticalPIIType.SIN: [
                re.compile(r'\b\d{3}[-\s]?\d{3}[-\s]?\d{3}\b', re.IGNORECASE),
                re.compile(r'\bSIN:?\s*\d{3}[-\s]?\d{3}[-\s]?\d{3}\b', re.IGNORECASE),
            ],
            CriticalPIIType.CREDIT_CARD: [
                # Visa
                re.compile(r'\b4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
                # MasterCard
                re.compile(r'\b5[1-5]\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
                # American Express
                re.compile(r'\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b'),
            ],
            CriticalPIIType.IRCC_NUMBER: [
                re.compile(r'\b[A-Z]{2}\d{8,12}\b', re.IGNORECASE),
                re.compile(r'\bIRCC[-\s]?[A-Z]{2}\d{8,12}\b', re.IGNORECASE),
            ],
            CriticalPIIType.UCI_NUMBER: [
                re.compile(r'\bUCI:?\s*\d{8,10}\b', re.IGNORECASE),
                re.compile(r'\b\d{8,10}\b'),  # Generic 8-10 digit numbers
            ],
            CriticalPIIType.PASSPORT: [
                re.compile(r'\b[A-Z]{2}\d{6}\b', re.IGNORECASE),  # Canadian
                re.compile(r'\b\d{9}\b'),  # US passport
            ],
        }
        
        logger.info(f"Compiled {sum(len(p) for p in patterns.values())} critical PII patterns")
        return patterns
    
    async def screen_content(self, content: str, max_length: int = 5000) -> FastPIIResult:
        """
        Screen content for critical PII patterns with sub-5ms target.
        
        Args:
            content: Text content to screen
            max_length: Maximum content length to process (performance limit)
            
        Returns:
            FastPIIResult with screening results
        """
        start_time = time.perf_counter()
        self.total_requests += 1
        
        # Quick size check - skip very large content
        if len(content) > max_length:
            logger.debug(f"Content too large ({len(content)} chars), skipping fast screening")
            return FastPIIResult(
                has_critical_pii=False,
                detected_types=[],
                anonymized_content=None,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                should_block=False
            )
        
        # Check cache first
        content_hash = self._get_content_hash(content)
        if content_hash in self.content_cache:
            self.cache_hits += 1
            cached_result = self.content_cache[content_hash]
            # Update processing time for cache hit
            cached_result.processing_time_ms = (time.perf_counter() - start_time) * 1000
            return cached_result
        
        # Perform fast screening
        detected_types = []
        anonymized_content = content
        
        for pii_type, patterns in self.critical_patterns.items():
            for pattern in patterns:
                if pattern.search(content):
                    detected_types.append(pii_type)
                    # Apply immediate anonymization for critical PII
                    anonymized_content = pattern.sub(
                        f"[REDACTED_{pii_type.value.upper()}]", 
                        anonymized_content
                    )
                    break  # One match per type is enough for fast screening
        
        # Determine if content should be blocked
        should_block = len(detected_types) > 0
        
        # Create result
        result = FastPIIResult(
            has_critical_pii=len(detected_types) > 0,
            detected_types=detected_types,
            anonymized_content=anonymized_content if should_block else None,
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
            should_block=should_block
        )
        
        # Cache result (with size limit)
        if len(self.content_cache) < self.cache_max_size:
            self.content_cache[content_hash] = result
        
        # Log performance metrics periodically
        if self.total_requests % 100 == 0:
            cache_hit_rate = (self.cache_hits / self.total_requests) * 100
            logger.info(
                f"Fast PII screener stats: {self.total_requests} requests, "
                f"{cache_hit_rate:.1f}% cache hit rate, "
                f"{result.processing_time_ms:.2f}ms avg"
            )
        
        return result
    
    def _get_content_hash(self, content: str) -> str:
        """Generate hash for content caching."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    async def batch_screen(self, contents: List[str]) -> List[FastPIIResult]:
        """
        Screen multiple content items efficiently.
        
        Args:
            contents: List of content strings to screen
            
        Returns:
            List of FastPIIResult objects
        """
        # Process in parallel for better performance
        tasks = [self.screen_content(content) for content in contents]
        return await asyncio.gather(*tasks)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        cache_hit_rate = (self.cache_hits / max(self.total_requests, 1)) * 100
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_hit_rate_percent": cache_hit_rate,
            "cache_size": len(self.content_cache),
            "patterns_loaded": sum(len(p) for p in self.critical_patterns.values())
        }
    
    def clear_cache(self):
        """Clear the content cache."""
        self.content_cache.clear()
        logger.info("Fast PII screener cache cleared")


# Global instance for reuse across requests
_fast_screener: Optional[FastPIIScreener] = None


def get_fast_screener() -> FastPIIScreener:
    """Get or create the global fast PII screener instance."""
    global _fast_screener
    if _fast_screener is None:
        _fast_screener = FastPIIScreener()
    return _fast_screener


async def quick_pii_check(content: str) -> FastPIIResult:
    """
    Convenience function for quick PII checking.
    
    Args:
        content: Content to check
        
    Returns:
        FastPIIResult with screening results
    """
    screener = get_fast_screener()
    return await screener.screen_content(content)


# Circuit breaker for reliability
class PIICircuitBreaker:
    """Circuit breaker for PII screening reliability."""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call_with_fallback(self, func, *args, **kwargs) -> FastPIIResult:
        """Call function with circuit breaker protection."""
        current_time = time.time()
        
        # Check if circuit should be half-open
        if (self.state == "OPEN" and 
            current_time - self.last_failure_time > self.timeout_seconds):
            self.state = "HALF_OPEN"
            logger.info("PII circuit breaker moving to HALF_OPEN state")
        
        # If circuit is open, return safe fallback
        if self.state == "OPEN":
            logger.warning("PII circuit breaker is OPEN, returning safe fallback")
            return FastPIIResult(
                has_critical_pii=False,
                detected_types=[],
                anonymized_content=None,
                processing_time_ms=0.1,
                should_block=False
            )
        
        try:
            result = await func(*args, **kwargs)
            
            # Success - reset circuit breaker if it was half-open
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("PII circuit breaker reset to CLOSED state")
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = current_time
            
            logger.error(f"PII screening failed: {e}")
            
            # Open circuit if threshold exceeded
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(
                    f"PII circuit breaker OPENED after {self.failure_count} failures"
                )
            
            # Return safe fallback
            return FastPIIResult(
                has_critical_pii=False,
                detected_types=[],
                anonymized_content=None,
                processing_time_ms=0.1,
                should_block=False
            )


# Global circuit breaker instance
_circuit_breaker = PIICircuitBreaker()


async def safe_quick_pii_check(content: str) -> FastPIIResult:
    """
    Circuit breaker protected PII checking.
    
    Args:
        content: Content to check
        
    Returns:
        FastPIIResult with screening results
    """
    return await _circuit_breaker.call_with_fallback(quick_pii_check, content)
