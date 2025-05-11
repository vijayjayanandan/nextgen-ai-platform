from typing import Dict, List, Optional, Any
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sse_starlette.sse import EventSourceResponse

from app.core.security import get_current_user
from app.schemas.user import UserInDB
from app.schemas.completion import CompletionRequest, CompletionResponse
from app.services.orchestrator import OrchestratorService, get_orchestrator

router = APIRouter()


@router.post("/", response_model=CompletionResponse)
async def create_completion(
    request: CompletionRequest,
    current_user: UserInDB = Depends(get_current_user),
    orchestrator: OrchestratorService = Depends(get_orchestrator)
):
    """
    Generate a text completion for the given prompt.
    
    This endpoint provides an interface similar to OpenAI's completions API.
    """
    try:
        # Process the completion request through the orchestrator
        response = await orchestrator.process_completion(request, str(current_user.id))
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating completion: {str(e)}"
        )


@router.post("/stream")
async def stream_completion(
    request: CompletionRequest,
    current_user: UserInDB = Depends(get_current_user),
    orchestrator: OrchestratorService = Depends(get_orchestrator)
):
    """
    Stream a text completion for the given prompt.
    
    This endpoint returns a Server-Sent Events (SSE) stream of completion chunks.
    """
    if not request.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'stream' parameter must be set to true for streaming"
        )
    
    async def event_generator():
        try:
            # Apply content filtering to prompt
            filtered_prompt, filter_details = await orchestrator.content_filter.filter_prompt(
                request.prompt, 
                str(current_user.id),
                context={"model": request.model}
            )
            
            # If filtering blocked the content, return error
            if filter_details.get("filtered", False) and filtered_prompt != request.prompt:
                yield {
                    "data": {
                        "error": "Content policy violation",
                        "message": "Your prompt contains content that violates our usage policies."
                    }
                }
                return
            
            # Check for retrieval options
            source_chunks = []
            if request.retrieve:
                # Apply retrieval augmentation
                retrieval_options = request.retrieval_options or {}
                top_k = retrieval_options.get("top_k", 5)
                filters = retrieval_options.get("filters")
                
                try:
                    source_chunks = await orchestrator.embedding_service.semantic_search(
                        filtered_prompt,
                        top_k=top_k,
                        filters=filters
                    )
                except Exception as e:
                    # Continue without retrieved content
                    pass
            
            # Augment prompt with retrieved content if available
            augmented_prompt = filtered_prompt
            if source_chunks:
                # Build context from retrieved chunks
                context_text = "\n\n".join([
                    f"Document: {chunk.get('metadata', {}).get('document_title', 'Untitled')}\n"
                    f"Content: {chunk.get('content', '')}"
                    for chunk in source_chunks
                ])
                
                # Augment the prompt with context
                augmented_prompt = (
                    f"Context information:\n{context_text}\n\n"
                    f"User query: {filtered_prompt}\n\n"
                    f"Please answer the query based on the context information provided."
                )
            
            # Stream the completion
            stream = orchestrator.model_router.route_stream_completion_request(
                prompt=augmented_prompt,
                model=request.model,
                user_id=str(current_user.id),
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                user=request.user,
                metadata={
                    "retrieval_enabled": bool(source_chunks),
                    "chunks_retrieved": len(source_chunks)
                }
            )
            
            # Send initial metadata event
            yield {
                "event": "metadata",
                "data": {
                    "model": request.model,
                    "created": int(uuid.uuid4().time_low),  # Simple timestamp approximation
                    "retrieval_used": bool(source_chunks),
                    "chunks_retrieved": len(source_chunks)
                }
            }
            
            # Stream completion chunks
            async for chunk in stream:
                # Check if chunk contains sensitive content
                filtered_chunk, filter_details = await orchestrator.content_filter.filter_response(
                    chunk,
                    str(current_user.id),
                    context={"model": request.model, "streaming": True}
                )
                
                # If content is filtered, send a warning
                if filter_details.get("filtered", False) and filtered_chunk != chunk:
                    yield {
                        "event": "warning",
                        "data": {
                            "message": "Content filtered due to policy violation",
                            "type": "content_filter"
                        }
                    }
                    # Use the filtered chunk
                    chunk = filtered_chunk
                
                # Only send non-empty chunks
                if chunk:
                    yield {"data": chunk}
            
            # Send done event
            yield {
                "event": "done",
                "data": {}
            }
            
        except Exception as e:
            # Send error event
            yield {
                "event": "error",
                "data": {
                    "message": f"Error generating completion: {str(e)}"
                }
            }
    
    # Return SSE response
    return EventSourceResponse(event_generator())