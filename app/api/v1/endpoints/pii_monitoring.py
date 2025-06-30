"""
PII Monitoring Endpoints
Provides status and metrics for the PII filtering system.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List
from datetime import datetime

from app.core.dependencies import get_current_user
from app.services.pii import get_fast_screener, get_background_processor
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/pii/status", tags=["PII Monitoring"])
async def get_pii_system_status(
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get comprehensive PII filtering system status.
    
    Returns:
        System status including performance metrics and health checks
    """
    try:
        # Get fast screener stats
        fast_screener = get_fast_screener()
        fast_stats = fast_screener.get_performance_stats()
        
        # Get background processor stats
        background_processor = get_background_processor()
        background_stats = background_processor.get_queue_stats()
        
        # System health checks
        health_checks = {
            "fast_screener_loaded": fast_screener is not None,
            "background_processor_running": background_stats["is_running"],
            "patterns_compiled": fast_stats["patterns_loaded"] > 0,
            "workers_active": background_stats["workers_running"] > 0
        }
        
        overall_health = all(health_checks.values())
        
        return {
            "status": "healthy" if overall_health else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "fast_screening": {
                "total_requests": fast_stats["total_requests"],
                "cache_hits": fast_stats["cache_hits"],
                "cache_hit_rate_percent": fast_stats["cache_hit_rate_percent"],
                "cache_size": fast_stats["cache_size"],
                "patterns_loaded": fast_stats["patterns_loaded"]
            },
            "background_processing": {
                "queue_size": background_stats["queue_size"],
                "max_queue_size": background_stats["max_queue_size"],
                "active_tasks": background_stats["active_tasks"],
                "total_processed": background_stats["total_processed"],
                "failed_tasks": background_stats["failed_tasks"],
                "avg_processing_time": background_stats["avg_processing_time"],
                "workers_running": background_stats["workers_running"],
                "is_running": background_stats["is_running"]
            },
            "health_checks": health_checks,
            "overall_health": overall_health
        }
        
    except Exception as e:
        logger.error(f"Error getting PII system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get PII system status")


@router.get("/pii/performance", tags=["PII Monitoring"])
async def get_pii_performance_metrics(
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get detailed performance metrics for PII filtering.
    
    Returns:
        Performance metrics and benchmarks
    """
    try:
        fast_screener = get_fast_screener()
        background_processor = get_background_processor()
        
        fast_stats = fast_screener.get_performance_stats()
        background_stats = background_processor.get_queue_stats()
        
        # Calculate performance indicators
        cache_efficiency = fast_stats["cache_hit_rate_percent"]
        queue_utilization = (background_stats["queue_size"] / 
                           max(background_stats["max_queue_size"], 1)) * 100
        
        # Performance thresholds (industry standards)
        thresholds = {
            "cache_hit_rate_good": 70.0,
            "queue_utilization_warning": 80.0,
            "avg_processing_time_warning": 2.0  # seconds
        }
        
        # Performance assessment
        performance_issues = []
        
        if cache_efficiency < thresholds["cache_hit_rate_good"]:
            performance_issues.append("Low cache hit rate")
        
        if queue_utilization > thresholds["queue_utilization_warning"]:
            performance_issues.append("High queue utilization")
        
        if background_stats["avg_processing_time"] > thresholds["avg_processing_time_warning"]:
            performance_issues.append("Slow background processing")
        
        performance_grade = "excellent" if not performance_issues else \
                          "good" if len(performance_issues) == 1 else "needs_attention"
        
        return {
            "performance_grade": performance_grade,
            "performance_issues": performance_issues,
            "metrics": {
                "cache_efficiency_percent": cache_efficiency,
                "queue_utilization_percent": queue_utilization,
                "avg_background_processing_time": background_stats["avg_processing_time"],
                "total_requests_processed": fast_stats["total_requests"],
                "background_tasks_completed": background_stats["total_processed"],
                "error_rate_percent": (
                    (background_stats["failed_tasks"] / 
                     max(background_stats["total_processed"], 1)) * 100
                )
            },
            "thresholds": thresholds,
            "recommendations": _get_performance_recommendations(
                fast_stats, background_stats, performance_issues
            )
        }
        
    except Exception as e:
        logger.error(f"Error getting PII performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance metrics")


@router.post("/pii/cache/clear", tags=["PII Monitoring"])
async def clear_pii_cache(
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, str]:
    """
    Clear the PII screening cache.
    
    Returns:
        Confirmation of cache clearing
    """
    try:
        fast_screener = get_fast_screener()
        fast_screener.clear_cache()
        
        logger.info(f"PII cache cleared by user {current_user.get('user_id', 'unknown')}")
        
        return {
            "status": "success",
            "message": "PII screening cache cleared successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error clearing PII cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear PII cache")


@router.post("/pii/background/cleanup", tags=["PII Monitoring"])
async def cleanup_background_tasks(
    max_age_hours: int = 24,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Clean up completed background PII analysis tasks.
    
    Args:
        max_age_hours: Maximum age of completed tasks to keep (default: 24 hours)
    
    Returns:
        Cleanup results
    """
    try:
        background_processor = get_background_processor()
        
        # Get stats before cleanup
        stats_before = background_processor.get_queue_stats()
        
        # Perform cleanup
        await background_processor.cleanup_completed_tasks(max_age_hours)
        
        # Get stats after cleanup
        stats_after = background_processor.get_queue_stats()
        
        tasks_cleaned = stats_before["active_tasks"] - stats_after["active_tasks"]
        
        logger.info(
            f"Background task cleanup performed by user {current_user.get('user_id', 'unknown')}: "
            f"{tasks_cleaned} tasks cleaned"
        )
        
        return {
            "status": "success",
            "tasks_cleaned": tasks_cleaned,
            "active_tasks_before": stats_before["active_tasks"],
            "active_tasks_after": stats_after["active_tasks"],
            "max_age_hours": max_age_hours,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up background tasks: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup background tasks")


def _get_performance_recommendations(
    fast_stats: Dict[str, Any],
    background_stats: Dict[str, Any],
    issues: List[str]
) -> List[str]:
    """Generate performance recommendations based on current metrics."""
    
    recommendations = []
    
    if "Low cache hit rate" in issues:
        recommendations.append(
            "Consider increasing cache size or reviewing content patterns for better caching"
        )
    
    if "High queue utilization" in issues:
        recommendations.append(
            "Consider increasing background worker count or queue size"
        )
    
    if "Slow background processing" in issues:
        recommendations.append(
            "Review background processing performance and consider optimization"
        )
    
    if background_stats["failed_tasks"] > 0:
        recommendations.append(
            "Investigate failed background tasks and address underlying issues"
        )
    
    if not recommendations:
        recommendations.append("System is performing well - no immediate action needed")
    
    return recommendations


@router.get("/pii/test", tags=["PII Monitoring"])
async def test_pii_detection(
    content: str,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Test PII detection on provided content.
    
    Args:
        content: Content to test for PII
    
    Returns:
        PII detection results
    """
    try:
        from app.services.pii import safe_quick_pii_check
        
        # Perform fast PII screening
        result = await safe_quick_pii_check(content)
        
        logger.info(
            f"PII test performed by user {current_user.get('user_id', 'unknown')}: "
            f"critical_pii={result.has_critical_pii}"
        )
        
        return {
            "content_length": len(content),
            "has_critical_pii": result.has_critical_pii,
            "should_block": result.should_block,
            "detected_types": [t.value for t in result.detected_types],
            "processing_time_ms": result.processing_time_ms,
            "anonymized_content": result.anonymized_content,
            "test_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error testing PII detection: {e}")
        raise HTTPException(status_code=500, detail="Failed to test PII detection")
