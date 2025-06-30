from typing import Dict, List, Optional, Any, Set, Tuple, AsyncIterator
import re
import asyncio
import hashlib
import json
import secrets
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
from fastapi import HTTPException

# Import your existing content filter
from .content_filter import ContentFilter
from app.core.config import settings
from app.core.logging import get_logger, audit_log

logger = get_logger(__name__)


class PIIEntityType(Enum):
    """Enumeration of PII entity types for Canadian immigration context."""
    SIN = "sin"  # Social Insurance Number
    IRCC_NUMBER = "ircc_number"  # IRCC case numbers
    UCI_NUMBER = "uci_number"  # Unique Client Identifier
    PHONE = "phone"
    EMAIL = "email"
    ADDRESS = "address"
    PERSON_NAME = "person_name"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    CREDIT_CARD = "credit_card"
    HEALTH_CARD = "health_card"
    DRIVERS_LICENSE = "drivers_license"
    POSTAL_CODE = "postal_code"
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"
    EMPLOYEE_ID = "employee_id"
    CASE_NUMBER = "case_number"


class RiskLevel(Enum):
    """Risk levels for PII exposure."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnonymizationMethod(Enum):
    """Methods for anonymizing detected PII."""
    REDACTION = "redaction"
    TOKENIZATION = "tokenization"
    GENERALIZATION = "generalization"
    SUPPRESSION = "suppression"


@dataclass
class PIIDetection:
    """Represents a detected PII instance."""
    entity_type: PIIEntityType
    text: str
    start: int
    end: int
    confidence: float
    risk_level: RiskLevel
    context: str
    detection_method: str
    
    def to_dict(self):
        return {
            'entity_type': self.entity_type.value,
            'text': self.text,
            'start': self.start,
            'end': self.end,
            'confidence': self.confidence,
            'risk_level': self.risk_level.value,
            'context': self.context,
            'detection_method': self.detection_method
        }


@dataclass
class FilterResult:
    """Result of content filtering operation."""
    filtered_content: str
    pii_detected: bool
    detections: List[PIIDetection]
    is_safe: bool
    risk_score: float
    anonymization_applied: bool
    original_length: int
    filtered_length: int
    processing_time_ms: float


class EnhancedPIIFilter:
    """
    Enhanced PII filter with Canadian immigration-specific patterns
    and ML-based detection capabilities.
    """
    
    def __init__(self):
        """Initialize the enhanced PII filter."""
        # Canadian-specific PII patterns
        self.canadian_patterns = {
            PIIEntityType.SIN: [
                # Social Insurance Number - Canadian format
                r"\b\d{3}[-\s]?\d{3}[-\s]?\d{3}\b",
                # Alternative SIN formats
                r"\bSIN:?\s*\d{3}[-\s]?\d{3}[-\s]?\d{3}\b"
            ],
            PIIEntityType.IRCC_NUMBER: [
                # IRCC case numbers
                r"\b[A-Z]{2}\d{8,12}\b",
                # IRCC with prefix
                r"\bIRCC[-\s]?[A-Z]{2}\d{8,12}\b"
            ],
            PIIEntityType.UCI_NUMBER: [
                # Unique Client Identifier
                r"\b\d{8,10}\b",
                # UCI with prefix
                r"\bUCI:?\s*\d{8,10}\b"
            ],
            PIIEntityType.POSTAL_CODE: [
                # Canadian postal code
                r"\b[A-Z]\d[A-Z][-\s]?\d[A-Z]\d\b",
                # Postal code variations
                r"\b[A-Z]\d[A-Z]\s\d[A-Z]\d\b"
            ],
            PIIEntityType.HEALTH_CARD: [
                # Ontario OHIP
                r"\b\d{4}[-\s]?\d{3}[-\s]?\d{3}[-\s]?[A-Z]{2}\b",
                # Generic Canadian health card
                r"\b\d{4}[-\s]?\d{3}[-\s]?\d{3}\b",
                # Health card with prefix
                r"\bHC:?\s*\d{4}[-\s]?\d{3}[-\s]?\d{3}\b"
            ],
            PIIEntityType.DRIVERS_LICENSE: [
                # Ontario
                r"\b[A-Z]\d{4}[-\s]?\d{5}[-\s]?\d{5}\b",
                # Quebec
                r"\b[A-Z]\d{4}[-\s]?\d{6}[-\s]?\d{2}\b",
                # British Columbia
                r"\b\d{7}\b",
                # Alberta
                r"\b\d{6}[-\s]?\d{3}\b"
            ],
            PIIEntityType.PASSPORT: [
                # Canadian passport
                r"\b[A-Z]{2}\d{6}\b",
                # US passport
                r"\b\d{9}\b",
                # Generic passport pattern
                r"\b[A-Z]{1,2}\d{6,9}\b"
            ],
            PIIEntityType.PHONE: [
                # North American format with country code
                r"\b\+1[-\s]?\(?([2-9]\d{2})\)?[-\s]?([2-9]\d{2})[-\s]?(\d{4})\b",
                # North American format without country code
                r"\b\(?([2-9]\d{2})\)?[-\s]?([2-9]\d{2})[-\s]?(\d{4})\b",
                # International format
                r"\b\+\d{1,3}[-\s]?\d{1,14}\b"
            ],
            PIIEntityType.EMAIL: [
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            ],
            PIIEntityType.CREDIT_CARD: [
                # Visa
                r"\b4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
                # MasterCard
                r"\b5[1-5]\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
                # American Express
                r"\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b",
                # Generic credit card
                r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
            ],
            PIIEntityType.IP_ADDRESS: [
                # IPv4
                r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
                # IPv6
                r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
            ],
            PIIEntityType.MAC_ADDRESS: [
                r"\b(?:[0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}\b"
            ],
            PIIEntityType.ADDRESS: [
                # Canadian postal code in address
                r"\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Circle|Cir|Court|Ct)\b",
                # Address with postal code
                r"\b\d+\s+[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*[A-Z]{2}\s*[A-Z]\d[A-Z][-\s]?\d[A-Z]\d\b"
            ],
            PIIEntityType.CASE_NUMBER: [
                # Immigration case numbers
                r"\bCase:?\s*[A-Z]{2,4}\d{6,12}\b",
                r"\bFile:?\s*[A-Z]{2,4}\d{6,12}\b",
                r"\bRef:?\s*[A-Z]{2,4}\d{6,12}\b"
            ]
        }
        
        # Risk level mapping
        self.risk_levels = {
            PIIEntityType.SIN: RiskLevel.CRITICAL,
            PIIEntityType.CREDIT_CARD: RiskLevel.CRITICAL,
            PIIEntityType.PASSPORT: RiskLevel.HIGH,
            PIIEntityType.IRCC_NUMBER: RiskLevel.HIGH,
            PIIEntityType.UCI_NUMBER: RiskLevel.HIGH,
            PIIEntityType.HEALTH_CARD: RiskLevel.HIGH,
            PIIEntityType.DRIVERS_LICENSE: RiskLevel.HIGH,
            PIIEntityType.PHONE: RiskLevel.MEDIUM,
            PIIEntityType.EMAIL: RiskLevel.MEDIUM,
            PIIEntityType.ADDRESS: RiskLevel.MEDIUM,
            PIIEntityType.POSTAL_CODE: RiskLevel.MEDIUM,
            PIIEntityType.CASE_NUMBER: RiskLevel.MEDIUM,
            PIIEntityType.IP_ADDRESS: RiskLevel.LOW,
            PIIEntityType.MAC_ADDRESS: RiskLevel.MEDIUM,
            PIIEntityType.EMPLOYEE_ID: RiskLevel.MEDIUM,
        }
        
        # Anonymization methods by entity type
        self.anonymization_methods = {
            PIIEntityType.SIN: AnonymizationMethod.TOKENIZATION,
            PIIEntityType.CREDIT_CARD: AnonymizationMethod.TOKENIZATION,
            PIIEntityType.PASSPORT: AnonymizationMethod.TOKENIZATION,
            PIIEntityType.IRCC_NUMBER: AnonymizationMethod.TOKENIZATION,
            PIIEntityType.UCI_NUMBER: AnonymizationMethod.TOKENIZATION,
            PIIEntityType.HEALTH_CARD: AnonymizationMethod.TOKENIZATION,
            PIIEntityType.PHONE: AnonymizationMethod.GENERALIZATION,
            PIIEntityType.EMAIL: AnonymizationMethod.REDACTION,
            PIIEntityType.ADDRESS: AnonymizationMethod.GENERALIZATION,
            PIIEntityType.POSTAL_CODE: AnonymizationMethod.GENERALIZATION,
            PIIEntityType.IP_ADDRESS: AnonymizationMethod.REDACTION,
            PIIEntityType.PERSON_NAME: AnonymizationMethod.GENERALIZATION,
        }
        
        # Token mapping for reversible anonymization
        self.token_mapping = {}
        
        # Initialize ML components (simulated for now)
        self.ml_models_loaded = self._initialize_ml_components()
    
    def _initialize_ml_components(self) -> bool:
        """Initialize ML-based PII detection components."""
        try:
            # In a real implementation, you would load spaCy models here
            # For now, we'll simulate ML detection capability
            logger.info("ML components initialized for enhanced PII detection")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize ML components: {e}")
            return False
    
    async def detect_pii_entities(self, text: str) -> List[PIIDetection]:
        """
        Detect PII entities using multiple detection methods.
        
        Args:
            text: Text content to analyze
            
        Returns:
            List of PII detections
        """
        detections = []
        
        # 1. Pattern-based detection for Canadian formats
        pattern_detections = self._detect_canadian_patterns(text)
        detections.extend(pattern_detections)
        
        # 2. ML-based detection (simulated for now)
        if self.ml_models_loaded:
            ml_detections = await self._detect_with_ml(text)
            detections.extend(ml_detections)
        
        # 3. Context-aware validation
        validated_detections = self._validate_detections(text, detections)
        
        # 4. Deduplicate and merge overlapping detections
        final_detections = self._deduplicate_detections(validated_detections)
        
        return final_detections
    
    def _detect_canadian_patterns(self, text: str) -> List[PIIDetection]:
        """Detect Canadian-specific PII patterns."""
        detections = []
        
        for entity_type, patterns in self.canadian_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    # Extract context around the match
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    
                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_pattern_confidence(entity_type, match.group())
                    
                    detection = PIIDetection(
                        entity_type=entity_type,
                        text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                        risk_level=self.risk_levels.get(entity_type, RiskLevel.MEDIUM),
                        context=context,
                        detection_method="pattern_matching"
                    )
                    
                    detections.append(detection)
        
        return detections
    
    async def _detect_with_ml(self, text: str) -> List[PIIDetection]:
        """
        ML-based PII detection using NER models.
        
        Args:
            text: Text content to analyze
            
        Returns:
            List of PII detections
        """
        detections = []
        
        # Simulate ML-based detection
        # In a real implementation, you would use spaCy, transformers, or custom models
        
        # Example: Detect person names, organizations, locations
        ml_entities = [
            # Simulated ML detections would go here
            # {"text": "John Doe", "label": "PERSON", "start": 0, "end": 8, "confidence": 0.95}
        ]
        
        for entity in ml_entities:
            # Map ML labels to our PII types
            pii_type = self._map_ml_label_to_pii_type(entity["label"])
            
            if pii_type:
                # Extract context
                start = max(0, entity["start"] - 50)
                end = min(len(text), entity["end"] + 50)
                context = text[start:end]
                
                detection = PIIDetection(
                    entity_type=pii_type,
                    text=entity["text"],
                    start=entity["start"],
                    end=entity["end"],
                    confidence=entity["confidence"],
                    risk_level=self.risk_levels.get(pii_type, RiskLevel.MEDIUM),
                    context=context,
                    detection_method="ml_detection"
                )
                
                detections.append(detection)
        
        return detections
    
    def _map_ml_label_to_pii_type(self, label: str) -> Optional[PIIEntityType]:
        """Map ML model labels to our PII entity types."""
        label_mapping = {
            "PERSON": PIIEntityType.PERSON_NAME,
            "ORG": None,  # Organizations are not PII
            "GPE": None,  # Geopolitical entities are not PII
            "DATE": PIIEntityType.DATE_OF_BIRTH,
            "PHONE": PIIEntityType.PHONE,
            "EMAIL": PIIEntityType.EMAIL,
        }
        return label_mapping.get(label)
    
    def _calculate_pattern_confidence(self, entity_type: PIIEntityType, text: str) -> float:
        """Calculate confidence score for pattern-based detections."""
        # Base confidence for pattern matching
        base_confidence = 0.85
        
        # Adjust based on entity type specificity
        if entity_type in [PIIEntityType.SIN, PIIEntityType.IRCC_NUMBER]:
            # Very specific Canadian patterns
            return 0.95
        elif entity_type in [PIIEntityType.CREDIT_CARD, PIIEntityType.PASSPORT]:
            # Well-defined international patterns
            return 0.90
        elif entity_type in [PIIEntityType.PHONE, PIIEntityType.EMAIL]:
            # Common patterns with some ambiguity
            return 0.80
        else:
            return base_confidence
    
    def _validate_detections(self, text: str, detections: List[PIIDetection]) -> List[PIIDetection]:
        """Validate detections using context analysis."""
        validated = []
        
        for detection in detections:
            # Context-based validation
            context_lower = detection.context.lower()
            
            # Check for false positives
            if self._is_likely_false_positive(detection, context_lower):
                # Reduce confidence for likely false positives
                detection.confidence *= 0.7
                if detection.confidence < 0.5:
                    continue  # Skip low-confidence detections
            
            # Check for high-confidence indicators
            if self._has_high_confidence_indicators(detection, context_lower):
                detection.confidence = min(0.98, detection.confidence * 1.1)
            
            validated.append(detection)
        
        return validated
    
    def _is_likely_false_positive(self, detection: PIIDetection, context: str) -> bool:
        """Check if detection is likely a false positive."""
        # Common false positive patterns
        false_positive_indicators = [
            "example", "sample", "test", "dummy", "fake",
            "xxx-xxx-xxxx", "000-000-0000", "123-456-7890"
        ]
        
        return any(indicator in context for indicator in false_positive_indicators)
    
    def _has_high_confidence_indicators(self, detection: PIIDetection, context: str) -> bool:
        """Check for high-confidence context indicators."""
        high_confidence_indicators = {
            PIIEntityType.SIN: ["sin", "social insurance", "insurance number"],
            PIIEntityType.IRCC_NUMBER: ["ircc", "immigration", "case number"],
            PIIEntityType.UCI_NUMBER: ["uci", "client identifier", "unique client"],
            PIIEntityType.PHONE: ["phone", "telephone", "call", "contact"],
            PIIEntityType.EMAIL: ["email", "e-mail", "contact", "@"],
        }
        
        indicators = high_confidence_indicators.get(detection.entity_type, [])
        return any(indicator in context for indicator in indicators)
    
    def _deduplicate_detections(self, detections: List[PIIDetection]) -> List[PIIDetection]:
        """Remove duplicate and overlapping detections."""
        if not detections:
            return []
        
        # Sort by start position
        sorted_detections = sorted(detections, key=lambda x: x.start)
        deduplicated = []
        
        for detection in sorted_detections:
            # Check for overlap with existing detections
            overlaps = False
            for existing in deduplicated:
                if self._detections_overlap(detection, existing):
                    # Keep the higher confidence detection
                    if detection.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(detection)
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(detection)
        
        return deduplicated
    
    def _detections_overlap(self, det1: PIIDetection, det2: PIIDetection) -> bool:
        """Check if two detections overlap."""
        return not (det1.end <= det2.start or det2.end <= det1.start)
    
    def apply_anonymization(self, text: str, detections: List[PIIDetection]) -> Tuple[str, Dict[str, Any]]:
        """
        Apply anonymization to detected PII entities.
        
        Args:
            text: Original text
            detections: List of PII detections
            
        Returns:
            Tuple of (anonymized_text, anonymization_details)
        """
        if not detections:
            return text, {"anonymized": False, "entities_processed": 0}
        
        # Sort detections by start position (reverse order for replacement)
        sorted_detections = sorted(detections, key=lambda x: x.start, reverse=True)
        
        anonymized_text = text
        anonymization_log = []
        
        for detection in sorted_detections:
            method = self.anonymization_methods.get(
                detection.entity_type, 
                AnonymizationMethod.REDACTION
            )
            
            # Apply anonymization method
            anonymized_value = self._apply_anonymization_method(
                detection.text, 
                detection.entity_type, 
                method
            )
            
            # Replace in text
            anonymized_text = (
                anonymized_text[:detection.start] + 
                anonymized_value + 
                anonymized_text[detection.end:]
            )
            
            # Log the anonymization
            anonymization_log.append({
                "entity_type": detection.entity_type.value,
                "original_text": detection.text,
                "anonymized_text": anonymized_value,
                "method": method.value,
                "start": detection.start,
                "end": detection.end,
                "risk_level": detection.risk_level.value
            })
        
        return anonymized_text, {
            "anonymized": True,
            "entities_processed": len(sorted_detections),
            "anonymization_log": anonymization_log,
            "original_length": len(text),
            "anonymized_length": len(anonymized_text)
        }
    
    def _apply_anonymization_method(
        self, 
        text: str, 
        entity_type: PIIEntityType, 
        method: AnonymizationMethod
    ) -> str:
        """Apply specific anonymization method to text."""
        
        if method == AnonymizationMethod.REDACTION:
            return f"[REDACTED_{entity_type.value.upper()}]"
        
        elif method == AnonymizationMethod.TOKENIZATION:
            # Generate reversible token
            token = f"TOKEN_{entity_type.value.upper()}_{secrets.token_hex(4).upper()}"
            self.token_mapping[token] = text
            return token
        
        elif method == AnonymizationMethod.GENERALIZATION:
            return self._generalize_entity(text, entity_type)
        
        elif method == AnonymizationMethod.SUPPRESSION:
            return ""
        
        return text
    
    def _generalize_entity(self, text: str, entity_type: PIIEntityType) -> str:
        """Apply generalization to specific entity types."""
        
        if entity_type == PIIEntityType.PHONE:
            return "XXX-XXX-XXXX"
        elif entity_type == PIIEntityType.EMAIL:
            return "[EMAIL_ADDRESS]"
        elif entity_type == PIIEntityType.ADDRESS:
            return "[ADDRESS]"
        elif entity_type == PIIEntityType.POSTAL_CODE:
            return "XXX XXX"
        elif entity_type == PIIEntityType.PERSON_NAME:
            return "[NAME]"
        else:
            return f"[{entity_type.value.upper()}]"


class EnterpriseContentFilter(ContentFilter):
    """
    Enterprise-grade content filter extending the base ContentFilter
    with advanced PII detection and Canadian immigration context.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        enable_enhanced_pii: bool = True,
        risk_threshold: float = 0.7
    ):
        """
        Initialize the enterprise content filter.
        
        Args:
            api_key: API key for OpenAI moderation API
            api_base: Base URL for OpenAI API
            enable_enhanced_pii: Enable enhanced PII detection
            risk_threshold: Risk threshold for content blocking
        """
        super().__init__(api_key, api_base)
        
        self.enable_enhanced_pii = enable_enhanced_pii
        self.risk_threshold = risk_threshold
        
        # Initialize enhanced PII filter
        if self.enable_enhanced_pii:
            self.pii_filter = EnhancedPIIFilter()
        
        logger.info("Enterprise content filter initialized with enhanced PII detection")
    
    async def filter_prompt(
        self, 
        prompt: str, 
        user_id: str, 
        context: Dict[str, Any] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Filter prompt content using enhanced PII detection.
        Maintains compatibility with orchestrator interface.
        
        Args:
            prompt: Prompt text to filter
            user_id: User ID for audit logging
            context: Additional context for filtering
            
        Returns:
            Tuple of (filtered_prompt, filter_details)
        """
        result = await self.enhanced_filter(prompt, user_id, context)
        
        # Convert FilterResult to orchestrator-compatible format
        filter_details = {
            "filtered": not result.is_safe,
            "pii_detected": result.pii_detected,
            "risk_score": result.risk_score,
            "detections": len(result.detections),
            "anonymization_applied": result.anonymization_applied,
            "processing_time_ms": result.processing_time_ms
        }
        
        return result.filtered_content, filter_details
    
    async def filter_response(
        self, 
        response: str, 
        user_id: str, 
        context: Dict[str, Any] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Filter response content using enhanced PII detection.
        Maintains compatibility with orchestrator interface.
        
        Args:
            response: Response text to filter
            user_id: User ID for audit logging
            context: Additional context for filtering
            
        Returns:
            Tuple of (filtered_response, filter_details)
        """
        result = await self.enhanced_filter(response, user_id, context)
        
        # Convert FilterResult to orchestrator-compatible format
        filter_details = {
            "filtered": not result.is_safe,
            "pii_detected": result.pii_detected,
            "risk_score": result.risk_score,
            "detections": len(result.detections),
            "anonymization_applied": result.anonymization_applied,
            "processing_time_ms": result.processing_time_ms
        }
        
        return result.filtered_content, filter_details
    
    async def enhanced_filter(
        self, 
        content: str, 
        user_id: str, 
        context: Dict[str, Any] = None
    ) -> FilterResult:
        """
        Enhanced filtering with advanced PII detection and risk assessment.
        
        Args:
            content: Text content to filter
            user_id: ID of the user submitting content
            context: Additional context for filtering
            
        Returns:
            FilterResult with comprehensive filtering details
        """
        start_time = datetime.utcnow()
        
        # 1. Basic content filtering (existing functionality)
        is_allowed, basic_details = await super().check_content(content, user_id, context)
        
        # 2. Enhanced PII detection
        pii_detections = []
        if self.enable_enhanced_pii:
            pii_detections = await self.pii_filter.detect_pii_entities(content)
        
        # 3. Calculate risk score
        risk_score = self._calculate_risk_score(pii_detections, basic_details)
        
        # 4. Apply anonymization if needed
        anonymized_content = content
        anonymization_applied = False
        anonymization_details = {}
        
        if pii_detections and risk_score > self.risk_threshold:
            anonymized_content, anonymization_details = self.pii_filter.apply_anonymization(
                content, pii_detections
            )
            anonymization_applied = True
        
        # 5. Determine final safety status
        is_safe = is_allowed and risk_score <= self.risk_threshold
        
        # 6. Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # 7. Create comprehensive audit log
        await self._log_enhanced_filtering(
            user_id=user_id,
            content=content,
            pii_detections=pii_detections,
            risk_score=risk_score,
            is_safe=is_safe,
            anonymization_applied=anonymization_applied,
            context=context
        )
        
        return FilterResult(
            filtered_content=anonymized_content,
            pii_detected=len(pii_detections) > 0,
            detections=pii_detections,
            is_safe=is_safe,
            risk_score=risk_score,
            anonymization_applied=anonymization_applied,
            original_length=len(content),
            filtered_length=len(anonymized_content),
            processing_time_ms=processing_time
        )
    
    def _calculate_risk_score(
        self, 
        pii_detections: List[PIIDetection], 
        basic_details: Dict[str, Any]
    ) -> float:
        """Calculate overall risk score based on detections."""
        
        if not pii_detections and not basic_details.get("flags", []):
            return 0.0
        
        # Base risk from basic filtering
        base_risk = 0.3 if basic_details.get("flags", []) else 0.0
        
        # PII risk calculation
        pii_risk = 0.0
        if pii_detections:
            # Weight by risk level and confidence
            risk_weights = {
                RiskLevel.CRITICAL: 1.0,
                RiskLevel.HIGH: 0.7,
                RiskLevel.MEDIUM: 0.4,
                RiskLevel.LOW: 0.1
            }
            
            total_weighted_risk = 0.0
            for detection in pii_detections:
                weight = risk_weights.get(detection.risk_level, 0.4)
                total_weighted_risk += weight * detection.confidence
            
            # Normalize by number of detections (diminishing returns)
            pii_risk = min(0.9, total_weighted_risk / (1 + len(pii_detections) * 0.1))
        
        # Combine risks (not simply additive)
        combined_risk = base_risk + pii_risk - (base_risk * pii_risk)
        
        return min(1.0, combined_risk)
    
    async def _log_enhanced_filtering(
        self,
        user_id: str,
        content: str,
        pii_detections: List[PIIDetection],
        risk_score: float,
        is_safe: bool,
        anonymization_applied: bool,
        context: Dict[str, Any] = None
    ):
        """Log enhanced filtering results for audit and compliance."""
        
        # Log PII detections if any
        if pii_detections:
            logger.warning(
                f"Enhanced PII detection: {len(pii_detections)} entities found",
                extra={
                    "user_id": user_id,
                    "pii_types": [d.entity_type.value for d in pii_detections],
                    "risk_score": risk_score,
                    "anonymization_applied": anonymization_applied
                }
            )
        
        # Create detailed audit log
        audit_log(
            user_id=user_id,
            action="enhanced_content_filtering",
            resource_type="content",
            resource_id="enhanced_filter",
            details={
                "content_length": len(content),
                "pii_detections_count": len(pii_detections),
                "pii_types": [d.entity_type.value for d in pii_detections],
                "risk_levels": [d.risk_level.value for d in pii_detections],
                "overall_risk_score": risk_score,
                "is_safe": is_safe,
                "anonymization_applied": anonymization_applied,
                "context": context or {}
            }
        )
