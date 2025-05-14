from typing import Dict, List, Optional, Any
import uuid
import json
from fastapi import APIRouter, Depends, HTTPException, status, Path, Query
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.core.security import get_current_user
from app.schemas.user import UserInDB
from app.schemas.chat import (
    ChatCompletionRequest, 
    ChatCompletionResponse,
    ConversationBase,
    ConversationCreate, 
    ConversationUpdate,
    ConversationInDB, 
    ConversationWithMessages,
    MessageCreate
)
from app.services.orchestrator import OrchestratorService, get_orchestrator
from app.models.chat import MessageRole

router = APIRouter()


# New conversation model
class ConversationRequest(BaseModel):
    title: Optional[str] = None
    model_name: str
    system_prompt: Optional[str] = None


# User message model
class UserMessageRequest(BaseModel):
    message: str


@router.post("/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    current_user: UserInDB = Depends(get_current_user),
    orchestrator: OrchestratorService = Depends(get_orchestrator)
):
    """
    Generate a chat completion for the given messages.
    
    This endpoint provides an interface similar to OpenAI's chat completions API.
    """
    try:
        # Process the chat completion request through the orchestrator
        response = await orchestrator.process_chat_completion(request, str(current_user.id))
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating chat completion: {str(e)}"
        )


@router.post("/stream")
async def stream_chat_completion(
    request: ChatCompletionRequest,
    current_user: UserInDB = Depends(get_current_user),
    orchestrator: OrchestratorService = Depends(get_orchestrator)
):
    """
    Stream a chat completion for the given messages.
    
    This endpoint returns a Server-Sent Events (SSE) stream of chat completion chunks.
    """
    if not request.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'stream' parameter must be set to true for streaming"
        )
    
    async def event_generator():
        try:
            # Process messages
            messages = request.messages
            
            # Filter the last user message
            last_user_message = None
            for msg in reversed(messages):
                if msg.role == MessageRole.USER:
                    last_user_message = msg
                    break
            
            if last_user_message:
                # Apply content filtering to last user message
                filtered_content, filter_details = await orchestrator.content_filter.filter_prompt(
                    last_user_message.content,
                    str(current_user.id),
                    context={"model": request.model}
                )
                
                # If filtering blocked the content, return error
                if filter_details.get("filtered", False) and filtered_content != last_user_message.content:
                    yield {
                        "data": {
                            "error": "Content policy violation",
                            "message": "Your message contains content that violates our usage policies."
                        }
                    }
                    return
                
                # Update message with filtered content
                last_user_message.content = filtered_content
            
            # Check for retrieval options
            source_chunks = []
            if request.retrieve and last_user_message:
                # Apply retrieval augmentation
                retrieval_options = request.retrieval_options or {}
                top_k = retrieval_options.get("top_k", 5)
                filters = retrieval_options.get("filters")
                
                try:
                    source_chunks = await orchestrator.embedding_service.semantic_search(
                        last_user_message.content,
                        top_k=top_k,
                        filters=filters
                    )
                except Exception as e:
                    # Continue without retrieved content
                    pass
            
            # Augment messages with retrieved content if available
            augmented_messages = list(messages)  # Create a copy
            if source_chunks:
                # Build context from retrieved chunks
                context_text = "\n\n".join([
                    f"Document: {chunk.get('metadata', {}).get('document_title', 'Untitled')}\n"
                    f"Content: {chunk.get('content', '')}"
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
                    system_message = {
                        "role": MessageRole.SYSTEM,
                        "content": f"Context information:\n{context_text}\n\nPlease use this context to answer the user's questions."
                    }
                    augmented_messages.insert(0, system_message)
            
            # Stream the chat completion - await the coroutine to get the async iterator
            stream = await orchestrator.model_router.route_stream_chat_completion_request(
                messages=augmented_messages,
                model=request.model,
                user_id=str(current_user.id),
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                functions=request.functions,
                function_call=request.function_call,
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
                    "message": f"Error generating chat completion: {str(e)}"
                }
            }
    
    # Return SSE response - call the generator function to get an iterator
    return EventSourceResponse(
        event_generator(),  # Call the function to get an iterator
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Important for Nginx
        }
    )


@router.post("/conversations", response_model=Dict[str, Any])
async def create_conversation(
    request: ConversationRequest,
    current_user: UserInDB = Depends(get_current_user),
    orchestrator: OrchestratorService = Depends(get_orchestrator)
):
    """
    Create a new conversation.
    """
    try:
        conversation = await orchestrator.create_conversation(
            title=request.title,
            user_id=str(current_user.id),
            model_name=request.model_name,
            system_prompt=request.system_prompt
        )
        return conversation
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating conversation: {str(e)}"
        )


@router.get("/conversations", response_model=Dict[str, Any])
async def list_conversations(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: UserInDB = Depends(get_current_user),
    orchestrator: OrchestratorService = Depends(get_orchestrator)
):
    """
    List all conversations for the current user.
    """
    try:
        conversations = await orchestrator.list_conversations(
            user_id=str(current_user.id),
            limit=limit,
            offset=offset
        )
        return conversations
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing conversations: {str(e)}"
        )


@router.get("/conversations/{conversation_id}", response_model=Dict[str, Any])
async def get_conversation(
    conversation_id: str = Path(..., description="ID of the conversation"),
    current_user: UserInDB = Depends(get_current_user),
    orchestrator: OrchestratorService = Depends(get_orchestrator)
):
    """
    Get a conversation by ID.
    """
    try:
        conversation = await orchestrator.get_conversation(
            conversation_id=conversation_id,
            user_id=str(current_user.id)
        )
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting conversation: {str(e)}"
        )


@router.post("/conversations/{conversation_id}/messages", response_model=Dict[str, Any])
async def add_message_to_conversation(
    request: UserMessageRequest,
    conversation_id: str = Path(..., description="ID of the conversation"),
    current_user: UserInDB = Depends(get_current_user),
    orchestrator: OrchestratorService = Depends(get_orchestrator)
):
    """
    Add a user message to a conversation and get the AI's response.
    """
    try:
        # Process the conversation message
        response = await orchestrator.process_conversation_message(
            conversation_id=conversation_id,
            user_message=request.message,
            user_id=str(current_user.id)
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )


@router.get("/conversations/{conversation_id}/messages", response_model=List[Dict[str, Any]])
async def get_conversation_messages(
    conversation_id: str = Path(..., description="ID of the conversation"),
    current_user: UserInDB = Depends(get_current_user),
    orchestrator: OrchestratorService = Depends(get_orchestrator)
):
    """
    Get all messages in a conversation.
    """
    try:
        conversation = await orchestrator.get_conversation(
            conversation_id=conversation_id,
            user_id=str(current_user.id)
        )
        return conversation.get("messages", [])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting conversation messages: {str(e)}"
        )
