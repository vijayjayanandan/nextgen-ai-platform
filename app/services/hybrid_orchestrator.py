"""
Hybrid Orchestrator Service - Two-Tier PII Architecture
Implements enterprise-grade security with optimal performance.
"""

from typing import Dict, List, Optional, Any, Union, Tuple, AsyncIterator
import uuid
import time
import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from fastapi import HTTPException, Depends
from fastapi.encoders import jsonable_encoder

from app.core.config import settings
from app.core.logging import get_logger, audit_log
from app.db.session import get_db
from app.models.chat import Conversation, Message, MessageRole
from app.services.model_router import ModelRouter, get_model_router
from app.services.moderation.enhanced_content_filter import EnterpriseContentFilter
from app.services.pii.fast_pii_screener import FastPIIScreener, get_fast_screener, safe_quick_pii_check
from app.services.pii.background_processor import BackgroundPIIProcessor, get_background_processor, submit_for_analysis, ProcessingPriority
from app.services.retrieval.vector_db_service import VectorDBService
from app.services.explanation.attribution_service import AttributionService
from app.services.embeddings.embedding_service import EmbeddingService
from app.schemas.chat import ChatMessage, ChatCompletionRequest, ChatCompletionResponse
from app.schemas.completion import CompletionRequest, CompletionResponse

logger = get_logger(__name__)


class HybridOrchestratorService:
    """
    Hybrid orchestrator implementing two-tier PII architecture:
    
    Tier 1: FastPIIScreener (0-5ms) - Immediate critical PII blocking
    Tier 2: BackgroundProcessor - Comprehensive analysis + audit logging
    
    This provides optimal performance with enterprise-grade security.
    """
    
    def __init__(
        self,
        db: AsyncSession,
        model_router: ModelRouter
    ):
        """
        Initialize the hybrid orchestrator service.
        
        Args:
            db: Database session
            model_router: Service for routing requests to appropriate LLM services
        """
        self.db = db
        self.model_router = model_router
        
        # Initialize two-tier PII architecture
        self.fast_screener = get_fast_screener()
        self.background_processor = get_background_processor()
        self.comprehensive_filter = EnterpriseContentFilter()
        
        # Initialize supporting services
        self.vector_db_service = VectorDBService()
        self.embedding_service = EmbeddingService()
        self.attribution_service = AttributionService(db)
        
        logger.info("Hybrid orchestrator initialized with two-tier PII architecture")
    
    async def process_completion(
        self,
        request: CompletionRequest,
        user_id: str
    ) -> CompletionResponse:
        """
        Process completion with two-tier PII filtering.
        
        Args:
            request: Completion request
            user_id: ID of the user making the request
            
        Returns:
            Completion response
        """
        start_time = time.time()
        
        # TIER 1: Fast PII screening (0-5ms) - Critical PII blocking
        fast_result = await safe_quick_pii_check(request.prompt)
        
        if fast_result.should_block:
            logger.warning(
                f"CRITICAL PII BLOCKED: User {user_id} prompt blocked by fast screener",
                extra={
                    "user_id": user_id,
                    "detected_types": [t.value for t in fast_result.detected_types],
                    "processing_time_ms": fast_result.processing_time_ms
                }
            )
            
            # Submit to background for comprehensive analysis
            await submit_for_analysis(
                content=request.prompt,
                user_id=user_id,
                endpoint="completion",
                priority=ProcessingPriority.HIGH,
                context={"blocked_by_fast_screener": True, "model": request.model}
            )
            
            return CompletionResponse(
                id=str(uuid.uuid4()),
                object="text_completion",
                created=int(time.time()),
                model=request.model,
                choices=[{
                    "text": "I cannot process this request as it contains sensitive information that violates our security policies. Please remove any personal identifiers and try again.",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "content_filter"
                }],
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            )
        
        # Content passed fast screening - continue processing
        filtered_prompt = fast_result.anonymized_content or request.prompt
        
        # Submit to background for comprehensive analysis (async)
        background_task_id = await submit_for_analysis(
            content=request.prompt,
            user_id=user_id,
            endpoint="completion",
            priority=ProcessingPriority.MEDIUM,
            context={"model": request.model, "fast_screening_passed": True}
        )
        
        # Step 2: Retrieve relevant documents if RAG is enabled
        source_chunks = []
        if settings.ENABLE_RETRIEVAL_AUGMENTATION and request.retrieve:
            retrieval_options = request.retrieval_options or {}
            top_k = retrieval_options.get("top_k", 5)
            filters = retrieval_options.get("filters")
            
            try:
                source_chunks = await self.embedding_service.semantic_search(
                    filtered_prompt,
                    top_k=top_k,
                    filters=filters
                )
                
                logger.info(f"Retrieved {len(source_chunks)} chunks for prompt")
            except Exception as e:
                logger.error(f"Error during retrieval: {str(e)}")
                # Continue without retrieved content
        
        # Step 3: Augment prompt with retrieved content if available
        augmented_prompt = filtered_prompt
        if source_chunks:
            # Build context from retrieved chunks
            context_text = "\n\n".join([
                f"Document: {chunk.metadata.get('document_title', 'Untitled') if chunk.metadata else 'Untitled'}\n"
                f"Content: {chunk.content}"
                for chunk in source_chunks
            ])
            
            # Augment the prompt with context
            augmented_prompt = (
                f"Context information:\n{context_text}\n\n"
                f"User query: {filtered_prompt}\n\n"
                f"Please answer the query based on the context information provided."
            )
        
        # Step 4: Generate completion using the model router
        completion_response = await self.model_router.route_completion_request(
            prompt=augmented_prompt,
            model=request.model,
            user_id=user_id,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            stop=request.stop,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            user=request.user,
            metadata={
                "retrieval_enabled": bool(source_chunks),
                "chunks_retrieved": len(source_chunks),
                "fast_screening_time_ms": fast_result.processing_time_ms,
                "background_task_id": background_task_id
            }
        )
        
        # Step 5: Fast screen the response
        response_choices = completion_response.choices
        for i, choice in enumerate(response_choices):
            # Fast screen response content
            response_fast_result = await safe_quick_pii_check(choice.text)
            
            if response_fast_result.should_block:
                logger.warning(f"Response blocked by fast screener for user {user_id}")
                choice.text = "I apologize, but I cannot provide this response as it may contain sensitive information."
                choice.finish_reason = "content_filter"
            elif response_fast_result.anonymized_content:
                # Apply fast anonymization if needed
                choice.text = response_fast_result.anonymized_content
            
            # Submit response to background analysis
            await submit_for_analysis(
                content=choice.text,
                user_id=user_id,
                endpoint="completion_response",
                priority=ProcessingPriority.LOW,
                context={
                    "model": request.model,
                    "completion_id": completion_response.id,
                    "choice_index": i
                }
            )
        
        # Step 6: Add attributions if using retrieval augmentation
        if source_chunks and settings.ENABLE_EXPLANATION:
            for i, choice in enumerate(response_choices):
                # Add attributions to each choice
                choice.text = await self.attribution_service.add_citations(
                    choice.text,
                    source_chunks
                )
            
            # Also add source document information to response
            completion_response.source_documents = [
                {
                    "id": str(chunk.document_id),
                    "title": chunk.metadata.get("document_title", "Untitled") if chunk.metadata else "Untitled",
                    "content": chunk.content,
                    "metadata": chunk.metadata or {}
                }
                for chunk in source_chunks
            ][:5]  # Limit to top 5 to avoid large responses
        
        # Record processing time
        processing_time = time.time() - start_time
        logger.info(f"Completion processed in {processing_time:.2f}s with two-tier PII filtering")
        
        # Audit log the completion
        audit_log(
            user_id=user_id,
            action="hybrid_completion",
            resource_type="completion",
            resource_id=completion_response.id,
            details={
                "model": request.model,
                "processing_time": processing_time,
                "prompt_tokens": completion_response.usage.prompt_tokens,
                "completion_tokens": completion_response.usage.completion_tokens,
                "total_tokens": completion_response.usage.total_tokens,
                "retrieval_used": bool(source_chunks),
                "chunks_retrieved": len(source_chunks),
                "fast_screening_time_ms": fast_result.processing_time_ms,
                "background_task_id": background_task_id,
                "critical_pii_blocked": fast_result.should_block
            }
        )
        
        return completion_response
    
    async def process_chat_completion(
        self,
        request: ChatCompletionRequest,
        user_id: str
    ) -> ChatCompletionResponse:
        """
        Process chat completion with two-tier PII filtering.
        
        Args:
            request: Chat completion request
            user_id: ID of the user making the request
            
        Returns:
            Chat completion response
        """
        start_time = time.time()
        
        # Process messages
        messages = request.messages
        
        # TIER 1: Fast PII screening for the last user message
        last_user_message = None
        for msg in reversed(messages):
            if msg.role == MessageRole.USER:
                last_user_message = msg
                break
        
        if last_user_message:
            # Fast screen the user message
            fast_result = await safe_quick_pii_check(last_user_message.content)
            
            if fast_result.should_block:
                logger.warning(
                    f"CRITICAL PII BLOCKED: User {user_id} chat message blocked by fast screener",
                    extra={
                        "user_id": user_id,
                        "detected_types": [t.value for t in fast_result.detected_types],
                        "processing_time_ms": fast_result.processing_time_ms
                    }
                )
                
                # Submit to background for comprehensive analysis
                await submit_for_analysis(
                    content=last_user_message.content,
                    user_id=user_id,
                    endpoint="chat_completion",
                    priority=ProcessingPriority.HIGH,
                    context={"blocked_by_fast_screener": True, "model": request.model}
                )
                
                return ChatCompletionResponse(
                    id=str(uuid.uuid4()),
                    object="chat.completion",
                    created=int(time.time()),
                    model=request.model,
                    choices=[{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "I cannot process this message as it contains sensitive information that violates our security policies. Please remove any personal identifiers and try again."
                        },
                        "finish_reason": "content_filter"
                    }],
                    usage={
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                )
            
            # Apply fast anonymization if needed
            if fast_result.anonymized_content:
                last_user_message.content = fast_result.anonymized_content
            
            # Submit to background for comprehensive analysis (async)
            background_task_id = await submit_for_analysis(
                content=last_user_message.content,
                user_id=user_id,
                endpoint="chat_completion",
                priority=ProcessingPriority.MEDIUM,
                context={"model": request.model, "fast_screening_passed": True}
            )
        
        # Step 2: Retrieve relevant documents if RAG is enabled
        source_chunks = []
        if settings.ENABLE_RETRIEVAL_AUGMENTATION and request.retrieve:
            # Use last user message for retrieval
            if last_user_message:
                retrieval_options = request.retrieval_options or {}
                top_k = retrieval_options.get("top_k", 5)
                filters = retrieval_options.get("filters")
                
                try:
                    source_chunks = await self.embedding_service.semantic_search(
                        last_user_message.content,
                        top_k=top_k,
                        filters=filters
                    )
                    
                    logger.info(f"Retrieved {len(source_chunks)} chunks for chat")
                except Exception as e:
                    logger.error(f"Error during retrieval: {str(e)}")
                    # Continue without retrieved content
        
        # Step 3: Augment messages with retrieved content if available
        augmented_messages = list(messages)  # Create a copy
        if source_chunks:
            # Build context from retrieved chunks
            context_text = "\n\n".join([
                f"Document: {chunk.metadata.get('document_title', 'Untitled') if chunk.metadata else 'Untitled'}\n"
                f"Content: {chunk.content}"
                for chunk in source_chunks
            ])
            
            # Add system message with context if not already present
            has_system = any(msg.role == MessageRole.SYSTEM for msg in messages)
            
            if has_system:
                # Update existing system message
                for i, msg in enumerate(augmented_messages):
                    if msg.role == MessageRole.SYSTEM:
                        augmented_messages[i].content = (
                            f"{msg.content}\n\n"
                            f"Context information:\n{context_text}"
                        )
                        break
            else:
                # Add new system message with context
                system_message = ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=f"Context information:\n{context_text}\n\nPlease use this context to answer the user's questions."
                )
                augmented_messages.insert(0, system_message)
        
        # Step 4: Generate chat completion using the model router
        chat_completion_response = await self.model_router.route_chat_completion_request(
            messages=augmented_messages,
            model=request.model,
            user_id=user_id,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            n=request.n,
            stop=request.stop,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            user=request.user,
            metadata={
                "retrieval_enabled": bool(source_chunks),
                "chunks_retrieved": len(source_chunks),
                "fast_screening_time_ms": fast_result.processing_time_ms if last_user_message else 0,
                "background_task_id": background_task_id if last_user_message else None
            }
        )
        
        # Step 5: Fast screen the response
        response_choices = chat_completion_response.choices
        for i, choice in enumerate(response_choices):
            # Fast screen response content
            response_fast_result = await safe_quick_pii_check(choice.message.content)
            
            if response_fast_result.should_block:
                logger.warning(f"Chat response blocked by fast screener for user {user_id}")
                choice.message.content = "I apologize, but I cannot provide this response as it may contain sensitive information."
                choice.finish_reason = "content_filter"
            elif response_fast_result.anonymized_content:
                # Apply fast anonymization if needed
                choice.message.content = response_fast_result.anonymized_content
            
            # Submit response to background analysis
            await submit_for_analysis(
                content=choice.message.content,
                user_id=user_id,
                endpoint="chat_completion_response",
                priority=ProcessingPriority.LOW,
                context={
                    "model": request.model,
                    "completion_id": chat_completion_response.id,
                    "choice_index": i
                }
            )
        
        # Step 6: Add attributions if using retrieval augmentation
        if source_chunks and settings.ENABLE_EXPLANATION:
            for i, choice in enumerate(response_choices):
                # Add attributions to each choice
                choice.message.content = await self.attribution_service.add_citations(
                    choice.message.content,
                    source_chunks
                )
            
            # Also add source document information to response
            chat_completion_response.source_documents = [
                {
                    "id": str(chunk.document_id),
                    "title": chunk.metadata.get("document_title", "Untitled") if chunk.metadata else "Untitled",
                    "content": chunk.content,
                    "metadata": chunk.metadata or {}
                }
                for chunk in source_chunks
            ][:5]  # Limit to top 5 to avoid large responses
        
        # Record processing time
        processing_time = time.time() - start_time
        logger.info(f"Chat completion processed in {processing_time:.2f}s with two-tier PII filtering")
        
        # Audit log the chat completion
        audit_log(
            user_id=user_id,
            action="hybrid_chat_completion",
            resource_type="chat_completion",
            resource_id=chat_completion_response.id,
            details={
                "model": request.model,
                "processing_time": processing_time,
                "prompt_tokens": chat_completion_response.usage.prompt_tokens,
                "completion_tokens": chat_completion_response.usage.completion_tokens,
                "total_tokens": chat_completion_response.usage.total_tokens,
                "retrieval_used": bool(source_chunks),
                "chunks_retrieved": len(source_chunks),
                "fast_screening_time_ms": fast_result.processing_time_ms if last_user_message else 0,
                "background_task_id": background_task_id if last_user_message else None,
                "critical_pii_blocked": fast_result.should_block if last_user_message else False
            }
        )
        
        return chat_completion_response
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the hybrid orchestrator.
        
        Returns:
            Dictionary containing performance metrics
        """
        # Get fast screener metrics
        fast_metrics = self.fast_screener.get_performance_stats()
        
        # Get background processor metrics
        background_metrics = self.background_processor.get_queue_stats()
        
        return {
            "fast_screener": fast_metrics,
            "background_processor": background_metrics,
            "architecture": "two_tier_hybrid",
            "tier_1": "FastPIIScreener (0-5ms critical blocking)",
            "tier_2": "BackgroundProcessor (comprehensive analysis)"
        }


# Dependency function for FastAPI
async def get_hybrid_orchestrator(
    db: AsyncSession = Depends(get_db),
    model_router: ModelRouter = Depends(get_model_router)
) -> HybridOrchestratorService:
    """
    Dependency function to get hybrid orchestrator service.
    
    Args:
        db: Database session
        model_router: Model router service
        
    Returns:
        HybridOrchestratorService instance
    """
    return HybridOrchestratorService(db, model_router)
