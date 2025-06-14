from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.services.moderation.enhanced_content_filter import (
    EnterpriseContentFilter, 
    FilterResult, 
    PIIDetection,
    PIIEntityType,
    RiskLevel
)
from app.core.security import get_current_user
from app.core.logging import get_logger, audit_log
from app.schemas.user import UserInDB

logger = get_logger(__name__)
router = APIRouter()


class PIIDetectionRequest(BaseModel):
    """Request model for PII detection."""
    content: str = Field(..., description="Text content to analyze for PII")
    sensitivity_level: str = Field(
        default="high", 
        description="Detection sensitivity: low, medium, high"
    )
    anonymize: bool = Field(
        default=False, 
        description="Whether to apply anonymization to detected PII"
    )


class PIIDetectionResponse(BaseModel):
    """Response model for PII detection results."""
    content: str = Field(description="Original or filtered content")
    pii_detected: bool = Field(description="Whether PII was detected")
    detections: List[Dict[str, Any]] = Field(description="List of PII detections")
    risk_score: float = Field(description="Overall risk score (0.0 to 1.0)")
    is_safe: bool = Field(description="Whether content is considered safe")
    anonymization_applied: bool = Field(description="Whether anonymization was applied")
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    compliance_status: Dict[str, Any] = Field(description="Compliance assessment")


class ContentModerationRequest(BaseModel):
    """Request model for comprehensive content moderation."""
    content: str = Field(..., description="Text content to moderate")
    context: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Additional context for moderation"
    )
    apply_filtering: bool = Field(
        default=True, 
        description="Whether to apply content filtering"
    )


class ContentModerationResponse(BaseModel):
    """Response model for content moderation results."""
    filtered_content: str = Field(description="Content after filtering")
    is_safe: bool = Field(description="Whether content passed moderation")
    moderation_flags: List[str] = Field(description="List of moderation flags")
    pii_detections: List[Dict[str, Any]] = Field(description="PII detection results")
    risk_assessment: Dict[str, Any] = Field(description="Risk assessment details")
    recommendations: List[str] = Field(description="Recommendations for content")


class ComplianceReportRequest(BaseModel):
    """Request model for compliance report generation."""
    content: str = Field(..., description="Content to assess for compliance")
    frameworks: List[str] = Field(
        default=["PIPEDA", "GDPR"], 
        description="Compliance frameworks to check"
    )
    user_context: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="User context for compliance assessment"
    )


class ComplianceReportResponse(BaseModel):
    """Response model for compliance assessment."""
    compliance_score: float = Field(description="Overall compliance score")
    framework_results: Dict[str, Dict[str, Any]] = Field(
        description="Results by compliance framework"
    )
    violations: List[str] = Field(description="Compliance violations found")
    recommendations: List[str] = Field(description="Compliance recommendations")
    data_subject_rights: List[str] = Field(description="Applicable data subject rights")


@router.post("/detect-pii", response_model=PIIDetectionResponse)
async def detect_pii(
    request: PIIDetectionRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Detect PII entities in text content with Canadian immigration context.
    
    This endpoint provides advanced PII detection specifically tuned for
    Canadian immigration documents and communications.
    """
    try:
        # Initialize enhanced content filter
        content_filter = EnterpriseContentFilter()
        
        # Adjust risk threshold based on sensitivity level
        risk_thresholds = {
            "low": 0.9,
            "medium": 0.7,
            "high": 0.5
        }
        content_filter.risk_threshold = risk_thresholds.get(
            request.sensitivity_level, 0.7
        )
        
        # Perform enhanced filtering
        filter_result = await content_filter.enhanced_filter(
            content=request.content,
            user_id=str(current_user.id),
            context={
                "endpoint": "detect_pii",
                "sensitivity_level": request.sensitivity_level,
                "anonymize_requested": request.anonymize
            }
        )
        
        # Apply anonymization if requested and PII detected
        final_content = request.content
        anonymization_applied = False
        
        if request.anonymize and filter_result.pii_detected:
            final_content = filter_result.filtered_content
            anonymization_applied = filter_result.anonymization_applied
        
        # Generate compliance status
        compliance_status = _generate_compliance_status(filter_result)
        
        # Log API usage
        logger.info(
            f"PII detection API used by user {current_user.id}",
            extra={
                "user_id": current_user.id,
                "content_length": len(request.content),
                "pii_detected": filter_result.pii_detected,
                "risk_score": filter_result.risk_score,
                "sensitivity_level": request.sensitivity_level
            }
        )
        
        return PIIDetectionResponse(
            content=final_content,
            pii_detected=filter_result.pii_detected,
            detections=[detection.to_dict() for detection in filter_result.detections],
            risk_score=filter_result.risk_score,
            is_safe=filter_result.is_safe,
            anonymization_applied=anonymization_applied,
            processing_time_ms=filter_result.processing_time_ms,
            compliance_status=compliance_status
        )
        
    except Exception as e:
        logger.error(f"Error in PII detection API: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing PII detection request"
        )


@router.post("/moderate-content", response_model=ContentModerationResponse)
async def moderate_content(
    request: ContentModerationRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Comprehensive content moderation with PII filtering and safety checks.
    
    This endpoint provides complete content moderation including PII detection,
    harmful content filtering, and compliance assessment.
    """
    try:
        # Initialize enhanced content filter
        content_filter = EnterpriseContentFilter()
        
        # Perform enhanced filtering
        filter_result = await content_filter.enhanced_filter(
            content=request.content,
            user_id=str(current_user.id),
            context={
                "endpoint": "moderate_content",
                "apply_filtering": request.apply_filtering,
                **(request.context if request.context else {})
            }
        )
        
        # Get basic moderation flags
        is_allowed, basic_details = await content_filter.check_content(
            content=request.content,
            user_id=str(current_user.id),
            context=request.context
        )
        
        # Generate recommendations
        recommendations = _generate_recommendations(filter_result, basic_details)
        
        # Create risk assessment
        risk_assessment = {
            "overall_risk_score": filter_result.risk_score,
            "pii_risk": len(filter_result.detections) > 0,
            "content_safety_risk": not is_allowed,
            "high_risk_entities": [
                d.entity_type.value for d in filter_result.detections 
                if d.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
            ],
            "processing_time_ms": filter_result.processing_time_ms
        }
        
        # Log moderation activity
        logger.info(
            f"Content moderation API used by user {current_user.id}",
            extra={
                "user_id": current_user.id,
                "content_length": len(request.content),
                "is_safe": filter_result.is_safe,
                "pii_detected": filter_result.pii_detected,
                "moderation_flags": basic_details.get("flags", [])
            }
        )
        
        return ContentModerationResponse(
            filtered_content=filter_result.filtered_content if request.apply_filtering else request.content,
            is_safe=filter_result.is_safe and is_allowed,
            moderation_flags=basic_details.get("flags", []),
            pii_detections=[detection.to_dict() for detection in filter_result.detections],
            risk_assessment=risk_assessment,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error in content moderation API: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing content moderation request"
        )


@router.post("/compliance-report", response_model=ComplianceReportResponse)
async def generate_compliance_report(
    request: ComplianceReportRequest,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Generate comprehensive compliance report for content.
    
    This endpoint assesses content against specified compliance frameworks
    and provides detailed recommendations for compliance.
    """
    try:
        # Initialize enhanced content filter
        content_filter = EnterpriseContentFilter()
        
        # Perform enhanced filtering for PII detection
        filter_result = await content_filter.enhanced_filter(
            content=request.content,
            user_id=str(current_user.id),
            context={
                "endpoint": "compliance_report",
                "frameworks": request.frameworks,
                **(request.user_context if request.user_context else {})
            }
        )
        
        # Generate framework-specific results
        framework_results = {}
        overall_violations = []
        overall_recommendations = []
        
        for framework in request.frameworks:
            framework_result = _assess_framework_compliance(
                framework, filter_result, request.content
            )
            framework_results[framework] = framework_result
            overall_violations.extend(framework_result.get("violations", []))
            overall_recommendations.extend(framework_result.get("recommendations", []))
        
        # Calculate overall compliance score
        compliance_score = _calculate_compliance_score(framework_results, filter_result)
        
        # Determine applicable data subject rights
        data_subject_rights = _get_data_subject_rights(request.frameworks, filter_result)
        
        # Log compliance assessment
        logger.info(
            f"Compliance report generated for user {current_user.id}",
            extra={
                "user_id": current_user.id,
                "frameworks": request.frameworks,
                "compliance_score": compliance_score,
                "violations_count": len(overall_violations),
                "pii_detected": filter_result.pii_detected
            }
        )
        
        return ComplianceReportResponse(
            compliance_score=compliance_score,
            framework_results=framework_results,
            violations=list(set(overall_violations)),  # Remove duplicates
            recommendations=list(set(overall_recommendations)),  # Remove duplicates
            data_subject_rights=data_subject_rights
        )
        
    except Exception as e:
        logger.error(f"Error in compliance report API: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating compliance report"
        )


def _generate_compliance_status(filter_result: FilterResult) -> Dict[str, Any]:
    """Generate compliance status based on filter results."""
    return {
        "pipeda_compliant": filter_result.risk_score < 0.3,
        "gdpr_compliant": filter_result.risk_score < 0.2,
        "data_minimization": len(filter_result.detections) == 0,
        "anonymization_available": filter_result.anonymization_applied,
        "audit_trail_complete": True,
        "retention_policy_applicable": filter_result.pii_detected
    }


def _generate_recommendations(
    filter_result: FilterResult, 
    basic_details: Dict[str, Any]
) -> List[str]:
    """Generate recommendations based on filtering results."""
    recommendations = []
    
    if filter_result.pii_detected:
        recommendations.append("Consider anonymizing detected PII before storage")
        recommendations.append("Implement data retention policies for personal information")
        
        # Specific recommendations based on PII types
        critical_pii = [
            d for d in filter_result.detections 
            if d.risk_level == RiskLevel.CRITICAL
        ]
        if critical_pii:
            recommendations.append("Critical PII detected - immediate anonymization recommended")
    
    if basic_details.get("flags"):
        recommendations.append("Content flagged for manual review")
        recommendations.append("Consider revising content to meet policy guidelines")
    
    if filter_result.risk_score > 0.7:
        recommendations.append("High risk content - additional security measures recommended")
    
    if not recommendations:
        recommendations.append("Content appears compliant with current policies")
    
    return recommendations


def _assess_framework_compliance(
    framework: str, 
    filter_result: FilterResult, 
    content: str
) -> Dict[str, Any]:
    """Assess compliance against specific framework."""
    
    if framework.upper() == "PIPEDA":
        return {
            "compliant": filter_result.risk_score < 0.3,
            "score": max(0.0, 1.0 - filter_result.risk_score),
            "violations": [
                "Personal information detected without explicit consent mechanism"
            ] if filter_result.pii_detected else [],
            "recommendations": [
                "Implement consent collection for personal information",
                "Apply purpose limitation principles",
                "Establish data retention schedules"
            ] if filter_result.pii_detected else ["Content appears PIPEDA compliant"],
            "requirements_met": {
                "consent": not filter_result.pii_detected,
                "purpose_limitation": True,
                "data_minimization": not filter_result.pii_detected,
                "accuracy": True,
                "safeguards": filter_result.anonymization_applied,
                "openness": True,
                "individual_access": True,
                "challenging_compliance": True
            }
        }
    
    elif framework.upper() == "GDPR":
        return {
            "compliant": filter_result.risk_score < 0.2,
            "score": max(0.0, 1.0 - filter_result.risk_score * 1.2),
            "violations": [
                "Personal data processing without legal basis"
            ] if filter_result.pii_detected else [],
            "recommendations": [
                "Establish legal basis for processing",
                "Implement data subject rights mechanisms",
                "Conduct privacy impact assessment"
            ] if filter_result.pii_detected else ["Content appears GDPR compliant"],
            "requirements_met": {
                "lawfulness": not filter_result.pii_detected,
                "fairness": True,
                "transparency": True,
                "purpose_limitation": True,
                "data_minimization": not filter_result.pii_detected,
                "accuracy": True,
                "storage_limitation": True,
                "integrity_confidentiality": filter_result.anonymization_applied,
                "accountability": True
            }
        }
    
    else:
        return {
            "compliant": True,
            "score": 1.0,
            "violations": [],
            "recommendations": [f"Framework {framework} assessment not implemented"],
            "requirements_met": {}
        }


def _calculate_compliance_score(
    framework_results: Dict[str, Dict[str, Any]], 
    filter_result: FilterResult
) -> float:
    """Calculate overall compliance score across frameworks."""
    if not framework_results:
        return 1.0
    
    scores = [result.get("score", 0.0) for result in framework_results.values()]
    base_score = sum(scores) / len(scores)
    
    # Adjust for PII risk
    if filter_result.pii_detected:
        base_score *= (1.0 - filter_result.risk_score * 0.5)
    
    return max(0.0, min(1.0, base_score))


def _get_data_subject_rights(
    frameworks: List[str], 
    filter_result: FilterResult
) -> List[str]:
    """Determine applicable data subject rights based on frameworks and PII detection."""
    rights = []
    
    if not filter_result.pii_detected:
        return ["No personal data detected - data subject rights not applicable"]
    
    if "PIPEDA" in [f.upper() for f in frameworks]:
        rights.extend([
            "Right to access personal information",
            "Right to correct inaccurate information",
            "Right to withdraw consent",
            "Right to file complaints with Privacy Commissioner"
        ])
    
    if "GDPR" in [f.upper() for f in frameworks]:
        rights.extend([
            "Right to access personal data",
            "Right to rectification",
            "Right to erasure (right to be forgotten)",
            "Right to restrict processing",
            "Right to data portability",
            "Right to object to processing",
            "Rights related to automated decision making"
        ])
    
    if "CCPA" in [f.upper() for f in frameworks]:
        rights.extend([
            "Right to know about personal information collected",
            "Right to delete personal information",
            "Right to opt-out of sale of personal information",
            "Right to non-discrimination"
        ])
    
    return list(set(rights))  # Remove duplicates
