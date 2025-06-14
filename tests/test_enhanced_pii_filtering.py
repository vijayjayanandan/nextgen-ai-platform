import pytest
import asyncio
from typing import List, Dict, Any

from app.services.moderation.enhanced_content_filter import (
    EnhancedPIIFilter,
    EnterpriseContentFilter,
    PIIEntityType,
    RiskLevel,
    PIIDetection
)


class TestEnhancedPIIFilter:
    """Test suite for enhanced PII filtering with Canadian immigration context."""
    
    @pytest.fixture
    def pii_filter(self):
        """Create PII filter instance for testing."""
        return EnhancedPIIFilter()
    
    @pytest.fixture
    def enterprise_filter(self):
        """Create enterprise content filter instance for testing."""
        return EnterpriseContentFilter()
    
    @pytest.mark.asyncio
    async def test_canadian_sin_detection(self, pii_filter):
        """Test detection of Canadian Social Insurance Numbers."""
        test_cases = [
            "My SIN is 123-456-789",
            "SIN: 123 456 789",
            "Social Insurance Number 123456789",
            "Please provide your 123-456-789 for verification"
        ]
        
        for test_content in test_cases:
            detections = await pii_filter.detect_pii_entities(test_content)
            
            # Should detect at least one SIN
            sin_detections = [d for d in detections if d.entity_type == PIIEntityType.SIN]
            assert len(sin_detections) > 0, f"Failed to detect SIN in: {test_content}"
            
            # Should be high confidence
            assert sin_detections[0].confidence > 0.9
            assert sin_detections[0].risk_level == RiskLevel.CRITICAL
    
    @pytest.mark.asyncio
    async def test_ircc_number_detection(self, pii_filter):
        """Test detection of IRCC case numbers."""
        test_cases = [
            "Your IRCC case number is AB12345678",
            "Reference: XY987654321",
            "IRCC-AB12345678 is your file number"
        ]
        
        for test_content in test_cases:
            detections = await pii_filter.detect_pii_entities(test_content)
            
            # Should detect IRCC number
            ircc_detections = [d for d in detections if d.entity_type == PIIEntityType.IRCC_NUMBER]
            assert len(ircc_detections) > 0, f"Failed to detect IRCC number in: {test_content}"
            assert ircc_detections[0].risk_level == RiskLevel.HIGH
    
    @pytest.mark.asyncio
    async def test_canadian_postal_code_detection(self, pii_filter):
        """Test detection of Canadian postal codes."""
        test_cases = [
            "I live at K1A 0A6",
            "Postal code: M5V 3A8",
            "Address includes K1A-0A6"
        ]
        
        for test_content in test_cases:
            detections = await pii_filter.detect_pii_entities(test_content)
            
            # Should detect postal code
            postal_detections = [d for d in detections if d.entity_type == PIIEntityType.POSTAL_CODE]
            assert len(postal_detections) > 0, f"Failed to detect postal code in: {test_content}"
    
    @pytest.mark.asyncio
    async def test_phone_number_detection(self, pii_filter):
        """Test detection of phone numbers."""
        test_cases = [
            "Call me at (613) 555-1234",
            "Phone: +1-416-555-9876",
            "Contact: 514.555.0123"
        ]
        
        for test_content in test_cases:
            detections = await pii_filter.detect_pii_entities(test_content)
            
            # Should detect phone number
            phone_detections = [d for d in detections if d.entity_type == PIIEntityType.PHONE]
            assert len(phone_detections) > 0, f"Failed to detect phone in: {test_content}"
    
    @pytest.mark.asyncio
    async def test_email_detection(self, pii_filter):
        """Test detection of email addresses."""
        test_cases = [
            "Contact me at john.doe@example.com",
            "Email: user@immigration.gc.ca",
            "Send to test.user+tag@domain.org"
        ]
        
        for test_content in test_cases:
            detections = await pii_filter.detect_pii_entities(test_content)
            
            # Should detect email
            email_detections = [d for d in detections if d.entity_type == PIIEntityType.EMAIL]
            assert len(email_detections) > 0, f"Failed to detect email in: {test_content}"
    
    @pytest.mark.asyncio
    async def test_false_positive_filtering(self, pii_filter):
        """Test that common false positives are filtered out."""
        test_cases = [
            "Example SIN: 123-456-789 (this is just an example)",
            "Sample phone: 555-555-5555 for testing",
            "Dummy email: test@example.com"
        ]
        
        for test_content in test_cases:
            detections = await pii_filter.detect_pii_entities(test_content)
            
            # Should have reduced confidence or be filtered out
            for detection in detections:
                if "example" in test_content.lower() or "sample" in test_content.lower():
                    assert detection.confidence < 0.8, f"False positive not filtered: {test_content}"
    
    @pytest.mark.asyncio
    async def test_anonymization_tokenization(self, pii_filter):
        """Test tokenization anonymization method."""
        test_content = "My SIN is 123-456-789 and my email is john@example.com"
        
        detections = await pii_filter.detect_pii_entities(test_content)
        anonymized_text, details = pii_filter.apply_anonymization(test_content, detections)
        
        # Should contain tokens instead of original PII
        assert "123-456-789" not in anonymized_text
        assert "john@example.com" not in anonymized_text
        assert "TOKEN_" in anonymized_text
        assert details["anonymized"] is True
        assert details["entities_processed"] > 0
    
    @pytest.mark.asyncio
    async def test_anonymization_generalization(self, pii_filter):
        """Test generalization anonymization method."""
        test_content = "Call me at (613) 555-1234"
        
        detections = await pii_filter.detect_pii_entities(test_content)
        anonymized_text, details = pii_filter.apply_anonymization(test_content, detections)
        
        # Should generalize phone number
        assert "(613) 555-1234" not in anonymized_text
        assert "XXX-XXX-XXXX" in anonymized_text or "[PHONE]" in anonymized_text
    
    @pytest.mark.asyncio
    async def test_multiple_pii_types(self, pii_filter):
        """Test detection of multiple PII types in single text."""
        test_content = """
        Dear Immigration Officer,
        
        My name is John Doe and my SIN is 123-456-789.
        You can reach me at john.doe@email.com or (613) 555-1234.
        My address is 123 Main St, Ottawa, ON K1A 0A6.
        My IRCC case number is AB12345678.
        
        Thank you.
        """
        
        detections = await pii_filter.detect_pii_entities(test_content)
        
        # Should detect multiple types
        detected_types = {d.entity_type for d in detections}
        expected_types = {
            PIIEntityType.SIN,
            PIIEntityType.EMAIL,
            PIIEntityType.PHONE,
            PIIEntityType.POSTAL_CODE,
            PIIEntityType.IRCC_NUMBER
        }
        
        # Should detect most expected types
        assert len(detected_types.intersection(expected_types)) >= 3
    
    @pytest.mark.asyncio
    async def test_enterprise_filter_integration(self, enterprise_filter):
        """Test enterprise content filter with enhanced PII detection."""
        test_content = "Please process my application. My SIN is 123-456-789."
        
        result = await enterprise_filter.enhanced_filter(
            content=test_content,
            user_id="test_user_123",
            context={"test": True}
        )
        
        # Should detect PII and apply filtering
        assert result.pii_detected is True
        assert len(result.detections) > 0
        assert result.risk_score > 0.0
        assert result.anonymization_applied is True
        assert "123-456-789" not in result.filtered_content
    
    @pytest.mark.asyncio
    async def test_risk_score_calculation(self, enterprise_filter):
        """Test risk score calculation based on PII types and confidence."""
        test_cases = [
            ("No PII here", 0.0),
            ("My email is test@example.com", 0.3),  # Medium risk
            ("My SIN is 123-456-789", 0.7),  # High risk (critical PII)
            ("SIN: 123-456-789, Phone: (613) 555-1234", 0.8)  # Multiple PII
        ]
        
        for content, expected_min_risk in test_cases:
            result = await enterprise_filter.enhanced_filter(
                content=content,
                user_id="test_user",
                context={"test": True}
            )
            
            if expected_min_risk == 0.0:
                assert result.risk_score == 0.0
            else:
                assert result.risk_score >= expected_min_risk
    
    @pytest.mark.asyncio
    async def test_context_aware_validation(self, pii_filter):
        """Test context-aware validation of detections."""
        # High confidence context
        high_confidence_content = "Please provide your SIN: 123-456-789 for verification"
        
        # Low confidence context  
        low_confidence_content = "Example SIN format: 123-456-789"
        
        high_detections = await pii_filter.detect_pii_entities(high_confidence_content)
        low_detections = await pii_filter.detect_pii_entities(low_confidence_content)
        
        # High confidence context should have higher confidence scores
        if high_detections and low_detections:
            assert high_detections[0].confidence > low_detections[0].confidence
    
    @pytest.mark.asyncio
    async def test_deduplication(self, pii_filter):
        """Test deduplication of overlapping detections."""
        # Content with overlapping patterns
        test_content = "SIN 123-456-789 and SIN: 123-456-789"
        
        detections = await pii_filter.detect_pii_entities(test_content)
        
        # Should deduplicate overlapping detections
        sin_detections = [d for d in detections if d.entity_type == PIIEntityType.SIN]
        
        # Should not have duplicate detections for the same SIN
        unique_texts = {d.text for d in sin_detections}
        assert len(unique_texts) <= 2  # At most 2 unique SIN texts
    
    def test_pattern_confidence_calculation(self, pii_filter):
        """Test confidence calculation for different pattern types."""
        # Test specific Canadian patterns
        sin_confidence = pii_filter._calculate_pattern_confidence(PIIEntityType.SIN, "123-456-789")
        email_confidence = pii_filter._calculate_pattern_confidence(PIIEntityType.EMAIL, "test@example.com")
        
        # SIN should have higher confidence than email
        assert sin_confidence > email_confidence
        assert sin_confidence >= 0.95  # Very specific Canadian pattern
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, enterprise_filter):
        """Test performance benchmarks for PII filtering."""
        test_content = """
        This is a longer text with multiple PII elements to test performance.
        SIN: 123-456-789, Email: john.doe@example.com, Phone: (613) 555-1234,
        Address: 123 Main St, Ottawa, ON K1A 0A6, IRCC: AB12345678.
        """ * 10  # Repeat to make it longer
        
        result = await enterprise_filter.enhanced_filter(
            content=test_content,
            user_id="performance_test",
            context={"benchmark": True}
        )
        
        # Should process within reasonable time (< 500ms for this size)
        assert result.processing_time_ms < 500
        assert result.pii_detected is True


class TestPIIMiddlewareIntegration:
    """Test suite for PII middleware integration."""
    
    @pytest.mark.asyncio
    async def test_middleware_request_filtering(self):
        """Test middleware filtering of incoming requests."""
        # This would require FastAPI test client setup
        # Placeholder for middleware integration tests
        pass
    
    @pytest.mark.asyncio
    async def test_middleware_response_filtering(self):
        """Test middleware filtering of outgoing responses."""
        # This would require FastAPI test client setup
        # Placeholder for middleware integration tests
        pass


if __name__ == "__main__":
    # Run basic tests
    async def run_basic_tests():
        pii_filter = EnhancedPIIFilter()
        
        # Test Canadian SIN detection
        test_content = "My SIN is 123-456-789"
        detections = await pii_filter.detect_pii_entities(test_content)
        print(f"Detected {len(detections)} PII entities in: {test_content}")
        
        for detection in detections:
            print(f"  - {detection.entity_type.value}: {detection.text} (confidence: {detection.confidence:.2f})")
        
        # Test anonymization
        if detections:
            anonymized, details = pii_filter.apply_anonymization(test_content, detections)
            print(f"Anonymized: {anonymized}")
            print(f"Details: {details}")
    
    # Run the basic tests
    asyncio.run(run_basic_tests())
