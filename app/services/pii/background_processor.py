"""
Background PII Processing Service
Handles comprehensive PII analysis without blocking main request flow.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from app.core.logging import get_logger, audit_log
from app.core.config import settings
from app.services.moderation.enhanced_content_filter import EnterpriseContentFilter, FilterResult

logger = get_logger(__name__)


class ProcessingPriority(Enum):
    """Priority levels for background processing."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PIIAnalysisTask:
    """Task for background PII analysis."""
    task_id: str
    content: str
    user_id: str
    endpoint: str
    priority: ProcessingPriority
    context: Dict[str, Any]
    created_at: datetime
    processed_at: Optional[datetime] = None
    result: Optional[FilterResult] = None


class BackgroundPIIProcessor:
    """
    Background processor for comprehensive PII analysis.
    Processes tasks asynchronously without blocking API responses.
    """
    
    def __init__(self):
        """Initialize background PII processor."""
        self.task_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.processing_tasks: Dict[str, PIIAnalysisTask] = {}
        self.is_running = False
        self.worker_tasks: List[asyncio.Task] = []
        self.num_workers = 3
        
        # Initialize enhanced content filter for comprehensive analysis
        self.content_filter = EnterpriseContentFilter(
            enable_enhanced_pii=True,
            risk_threshold=settings.PII_RISK_THRESHOLD
        )
        
        # Performance metrics
        self.total_processed = 0
        self.total_processing_time = 0.0
        self.failed_tasks = 0
        
        logger.info("Background PII processor initialized")
    
    async def start(self):
        """Start background processing workers."""
        if self.is_running:
            logger.warning("Background PII processor already running")
            return
        
        self.is_running = True
        
        # Start worker tasks
        for i in range(self.num_workers):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self.worker_tasks.append(task)
        
        logger.info(f"Started {self.num_workers} background PII processing workers")
    
    async def stop(self):
        """Stop background processing workers."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
        
        logger.info("Background PII processor stopped")
    
    async def submit_task(
        self,
        content: str,
        user_id: str,
        endpoint: str,
        priority: ProcessingPriority = ProcessingPriority.MEDIUM,
        context: Dict[str, Any] = None
    ) -> str:
        """
        Submit content for background PII analysis.
        
        Args:
            content: Content to analyze
            user_id: User ID for audit logging
            endpoint: API endpoint that triggered analysis
            priority: Processing priority
            context: Additional context information
            
        Returns:
            Task ID for tracking
        """
        task_id = f"pii_{int(time.time() * 1000)}_{hash(content) % 10000}"
        
        task = PIIAnalysisTask(
            task_id=task_id,
            content=content,
            user_id=user_id,
            endpoint=endpoint,
            priority=priority,
            context=context or {},
            created_at=datetime.utcnow()
        )
        
        try:
            # Add to queue (non-blocking)
            self.task_queue.put_nowait(task)
            self.processing_tasks[task_id] = task
            
            logger.debug(f"Submitted PII analysis task {task_id} for user {user_id}")
            return task_id
            
        except asyncio.QueueFull:
            logger.warning("PII analysis queue full, dropping task")
            return ""
    
    async def get_task_result(self, task_id: str) -> Optional[FilterResult]:
        """
        Get result of background PII analysis task.
        
        Args:
            task_id: Task ID to check
            
        Returns:
            FilterResult if task is complete, None otherwise
        """
        task = self.processing_tasks.get(task_id)
        if task and task.result:
            return task.result
        return None
    
    async def _worker(self, worker_name: str):
        """Background worker for processing PII analysis tasks."""
        logger.info(f"PII worker {worker_name} started")
        
        while self.is_running:
            try:
                # Get task from queue with timeout
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                # Process the task
                await self._process_task(task, worker_name)
                
            except asyncio.TimeoutError:
                # No tasks available, continue
                continue
            except Exception as e:
                logger.error(f"PII worker {worker_name} error: {e}")
                await asyncio.sleep(1)  # Brief pause on error
        
        logger.info(f"PII worker {worker_name} stopped")
    
    async def _process_task(self, task: PIIAnalysisTask, worker_name: str):
        """
        Process a single PII analysis task.
        
        Args:
            task: Task to process
            worker_name: Name of processing worker
        """
        start_time = time.time()
        
        try:
            logger.debug(f"Worker {worker_name} processing task {task.task_id}")
            
            # Perform comprehensive PII analysis
            result = await self.content_filter.enhanced_filter(
                content=task.content,
                user_id=task.user_id,
                context={
                    **task.context,
                    "endpoint": task.endpoint,
                    "background_processing": True,
                    "worker": worker_name
                }
            )
            
            # Update task with result
            task.result = result
            task.processed_at = datetime.utcnow()
            
            # Update metrics
            processing_time = time.time() - start_time
            self.total_processed += 1
            self.total_processing_time += processing_time
            
            # Log significant findings
            if result.pii_detected:
                logger.info(
                    f"Background PII analysis found {len(result.detections)} PII entities "
                    f"in task {task.task_id} (risk: {result.risk_score:.2f})"
                )
                
                # Create detailed audit log for PII findings
                await self._audit_pii_findings(task, result)
            
            # Log performance metrics periodically
            if self.total_processed % 50 == 0:
                avg_time = self.total_processing_time / self.total_processed
                logger.info(
                    f"Background PII processor stats: {self.total_processed} tasks, "
                    f"{avg_time:.2f}s avg processing time"
                )
            
        except Exception as e:
            self.failed_tasks += 1
            logger.error(f"Failed to process PII task {task.task_id}: {e}")
            
            # Create error result
            task.result = FilterResult(
                filtered_content=task.content,
                pii_detected=False,
                detections=[],
                is_safe=True,  # Fail safe
                risk_score=0.0,
                anonymization_applied=False,
                original_length=len(task.content),
                filtered_length=len(task.content),
                processing_time_ms=(time.time() - start_time) * 1000
            )
            task.processed_at = datetime.utcnow()
    
    async def _audit_pii_findings(self, task: PIIAnalysisTask, result: FilterResult):
        """
        Create detailed audit log for PII findings.
        
        Args:
            task: Processed task
            result: PII analysis result
        """
        # Create comprehensive audit entry
        audit_log(
            user_id=task.user_id,
            action="background_pii_analysis",
            resource_type="content",
            resource_id=task.task_id,
            details={
                "endpoint": task.endpoint,
                "content_length": len(task.content),
                "pii_detections_count": len(result.detections),
                "pii_types": [d.entity_type.value for d in result.detections],
                "risk_levels": [d.risk_level.value for d in result.detections],
                "overall_risk_score": result.risk_score,
                "anonymization_applied": result.anonymization_applied,
                "processing_time_ms": result.processing_time_ms,
                "priority": task.priority.value,
                "created_at": task.created_at.isoformat(),
                "processed_at": task.processed_at.isoformat() if task.processed_at else None,
                "context": task.context
            }
        )
        
        # Log high-risk findings separately
        if result.risk_score > 0.8:
            logger.warning(
                f"HIGH RISK PII detected in background analysis: "
                f"task={task.task_id}, user={task.user_id}, "
                f"risk={result.risk_score:.2f}, "
                f"types={[d.entity_type.value for d in result.detections]}"
            )
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue and processing statistics."""
        return {
            "queue_size": self.task_queue.qsize(),
            "max_queue_size": self.task_queue.maxsize,
            "active_tasks": len(self.processing_tasks),
            "total_processed": self.total_processed,
            "failed_tasks": self.failed_tasks,
            "avg_processing_time": (
                self.total_processing_time / max(self.total_processed, 1)
            ),
            "workers_running": len(self.worker_tasks),
            "is_running": self.is_running
        }
    
    async def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """
        Clean up completed tasks older than specified age.
        
        Args:
            max_age_hours: Maximum age of completed tasks to keep
        """
        cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        
        tasks_to_remove = []
        for task_id, task in self.processing_tasks.items():
            if (task.processed_at and 
                task.processed_at.timestamp() < cutoff_time):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.processing_tasks[task_id]
        
        if tasks_to_remove:
            logger.info(f"Cleaned up {len(tasks_to_remove)} completed PII analysis tasks")


# Global background processor instance
_background_processor: Optional[BackgroundPIIProcessor] = None


def get_background_processor() -> BackgroundPIIProcessor:
    """Get or create the global background PII processor."""
    global _background_processor
    if _background_processor is None:
        _background_processor = BackgroundPIIProcessor()
    return _background_processor


async def submit_for_analysis(
    content: str,
    user_id: str,
    endpoint: str,
    priority: ProcessingPriority = ProcessingPriority.MEDIUM,
    context: Dict[str, Any] = None
) -> str:
    """
    Submit content for background PII analysis.
    
    Args:
        content: Content to analyze
        user_id: User ID for audit logging
        endpoint: API endpoint that triggered analysis
        priority: Processing priority
        context: Additional context information
        
    Returns:
        Task ID for tracking
    """
    processor = get_background_processor()
    return await processor.submit_task(content, user_id, endpoint, priority, context)


async def get_analysis_result(task_id: str) -> Optional[FilterResult]:
    """
    Get result of background PII analysis.
    
    Args:
        task_id: Task ID to check
        
    Returns:
        FilterResult if available, None otherwise
    """
    processor = get_background_processor()
    return await processor.get_task_result(task_id)
