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
from app.services.moderation.content_filter import ContentFilter
from app.services.retrieval.vector_db_service import VectorDBService
from app.services.explanation.attribution_service import AttributionService
from app.services.embeddings.embedding_service import EmbeddingService
from app.schemas.chat import ChatMessage, ChatCompletionRequest, ChatCompletionResponse
from app.schemas.completion import CompletionRequest, CompletionResponse

logger = get_logger(__name__)


class OrchestratorService:
    """
    Main service that orchestrates the entire flow:
    - Content filtering
    - Retrieval augmented generation
    - LLM inference
    - Attribution and explanation
    """
    
    def __init__(
        self,
        db: AsyncSession,
        model_router: ModelRouter
    ):
        """
        Initialize the orchestrator service.
        
        Args:
            db: Database session
            model_router: Service for routing requests to appropriate LLM services
        """
        self.db = db
        self.model_router = model_router
        
        # Initialize supporting services
        self.content_filter = ContentFilter()
        self.vector_db_service = VectorDBService()
        self.embedding_service = EmbeddingService()
        self.attribution_service = AttributionService(db)
    
    async def process_completion(
        self,
        request: CompletionRequest,
        user_id: str
    ) -> CompletionResponse:
        """
        Process a completion request with the full pipeline.
        
        Args:
            request: Completion request
            user_id: ID of the user making the request
            
        Returns:
            Completion response
        """
        start_time = time.time()
        
        # Step 1: Filter the prompt
        filtered_prompt, filter_details = await self.content_filter.filter_prompt(
            request.prompt, 
            user_id,
            context={"model": request.model}
        )
        
        # If filtering blocked the content, return early
        if filter_details.get("filtered", False) and filtered_prompt != request.prompt:
            logger.warning(f"Prompt from user {user_id} was filtered")
            return CompletionResponse(
                id=str(uuid.uuid4()),
                object="text_completion",
                created=int(time.time()),
                model=request.model,
                choices=[{
                    "text": "I apologize, but your request contains content that violates our usage policies. Please rephrase your request.",
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
                "chunks_retrieved": len(source_chunks)
            }
        )
        
        # Step 5: Filter the response content if needed
        choices = completion_response.choices
        for i, choice in enumerate(choices):
            # Apply content filtering to each choice
            filtered_text, response_filter_details = await self.content_filter.filter_response(
                choice.text,
                user_id,
                context={"model": request.model}
            )
            
            # Update choice with filtered text
            choice.text = filtered_text
            if response_filter_details.get("filtered", False):
                choice.finish_reason = "content_filter"
        
        # Step 6: Add attributions if using retrieval augmentation
        if source_chunks and settings.ENABLE_EXPLANATION:
            for i, choice in enumerate(choices):
                # Add attributions to each choice
                choice.text = await self.attribution_service.add_citations(
                    choice.text,
                    source_chunks
                )
            
            # Also add source document information to response
            completion_response.source_documents = [
                {
                    "id": chunk.get("document_id", ""),
                    "title": chunk.get("metadata", {}).get("document_title", "Untitled"),
                    "content": chunk.get("content", ""),
                    "metadata": chunk.get("metadata", {})
                }
                for chunk in source_chunks
            ][:5]  # Limit to top 5 to avoid large responses
        
        # Record processing time
        processing_time = time.time() - start_time
        logger.info(f"Completion processed in {processing_time:.2f}s")
        
        # Audit log the completion
        audit_log(
            user_id=user_id,
            action="completion",
            resource_type="completion",
            resource_id=completion_response.id,
            details={
                "model": request.model,
                "processing_time": processing_time,
                "prompt_tokens": completion_response.usage.prompt_tokens,
                "completion_tokens": completion_response.usage.completion_tokens,
                "total_tokens": completion_response.usage.total_tokens,
                "retrieval_used": bool(source_chunks),
                "chunks_retrieved": len(source_chunks)
            }
        )
        
        return completion_response
    
    async def process_chat_completion(
        self,
        request: ChatCompletionRequest,
        user_id: str
    ) -> ChatCompletionResponse:
        """
        Process a chat completion request with the full pipeline.
        
        Args:
            request: Chat completion request
            user_id: ID of the user making the request
            
        Returns:
            Chat completion response
        """
        start_time = time.time()
        
        # Process messages
        messages = request.messages
        
        # Step 1: Apply content filtering to the last user message
        last_user_message = None
        for msg in reversed(messages):
            if msg.role == MessageRole.USER:
                last_user_message = msg
                break
        
        if last_user_message:
            filtered_content, filter_details = await self.content_filter.filter_prompt(
                last_user_message.content,
                user_id,
                context={"model": request.model}
            )
            
            # If filtering blocked the content, return early
            if filter_details.get("filtered", False) and filtered_content != last_user_message.content:
                logger.warning(f"Chat message from user {user_id} was filtered")
                return ChatCompletionResponse(
                    id=str(uuid.uuid4()),
                    object="chat.completion",
                    created=int(time.time()),
                    model=request.model,
                    choices=[{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "I apologize, but your message contains content that violates our usage policies. Please rephrase your request."
                        },
                        "finish_reason": "content_filter"
                    }],
                    usage={
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                )
            
            # Update message with filtered content
            last_user_message.content = filtered_content
        
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
            functions=request.functions,
            function_call=request.function_call,
            user=request.user,
            metadata={
                "retrieval_enabled": bool(source_chunks),
                "chunks_retrieved": len(source_chunks)
            }
        )
        
        # Step 5: Filter the response content if needed
        choices = chat_completion_response.choices
        for i, choice in enumerate(choices):
            # Skip if the content is None (e.g., for function calls)
            if choice.message.content is None:
                continue
                
            # Apply content filtering
            filtered_text, response_filter_details = await self.content_filter.filter_response(
                choice.message.content,
                user_id,
                context={"model": request.model}
            )
            
            # Update choice with filtered text
            choice.message.content = filtered_text
            if response_filter_details.get("filtered", False):
                choice.finish_reason = "content_filter"
        
        # Step 6: Add attributions if using retrieval augmentation
        if source_chunks and settings.ENABLE_EXPLANATION:
            for i, choice in enumerate(choices):
                # Skip if the content is None
                if choice.message.content is None:
                    continue
                    
                # Add attributions
                choice.message.content = await self.attribution_service.add_citations(
                    choice.message.content,
                    source_chunks
                )
            
            # Add source document information to response
            chat_completion_response.source_documents = [
                {
                    "id": chunk.get("document_id", ""),
                    "title": chunk.get("metadata", {}).get("document_title", "Untitled"),
                    "content": chunk.get("content", ""),
                    "metadata": chunk.get("metadata", {})
                }
                for chunk in source_chunks
            ][:5]  # Limit to top 5 to avoid large responses
        
        # Record processing time
        processing_time = time.time() - start_time
        logger.info(f"Chat completion processed in {processing_time:.2f}s")
        
        # Audit log the chat completion
        audit_log(
            user_id=user_id,
            action="chat_completion",
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
                "message_count": len(messages)
            }
        )
        
        return chat_completion_response
    
    async def create_conversation(
        self,
        title: Optional[str],
        user_id: str,
        model_name: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new conversation.
        
        Args:
            title: Title for the conversation
            user_id: ID of the user creating the conversation
            model_name: Name of the model to use
            system_prompt: Optional system prompt
            
        Returns:
            Dictionary with conversation information
        """
        try:
            # Create conversation record
            conversation = Conversation(
                user_id=uuid.UUID(user_id),
                title=title,
                model_name=model_name,
                system_prompt=system_prompt,
                is_active=True
            )
            
            self.db.add(conversation)
            await self.db.commit()
            await self.db.refresh(conversation)
            
            # Add system message if provided
            if system_prompt:
                system_message = Message(
                    conversation_id=conversation.id,
                    role=MessageRole.SYSTEM,
                    content=system_prompt,
                    sequence=0
                )
                
                self.db.add(system_message)
                await self.db.commit()
            
            # Return conversation info
            return {
                "id": str(conversation.id),
                "title": conversation.title,
                "created_at": conversation.created_at.isoformat(),
                "model_name": conversation.model_name,
                "system_prompt": conversation.system_prompt
            }
            
        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error creating conversation: {str(e)}"
            )
    
    async def add_message_to_conversation(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        function_name: Optional[str] = None,
        function_arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a message to an existing conversation.
        
        Args:
            conversation_id: ID of the conversation
            role: Role of the message sender
            content: Message content
            function_name: Name of the function (for function calls)
            function_arguments: Function arguments (for function calls)
            
        Returns:
            Dictionary with message information
        """
        try:
            # Check if conversation exists
            uuid_id = uuid.UUID(conversation_id)
            result = await self.db.execute(select(Conversation).where(Conversation.id == uuid_id))
            conversation = result.scalars().first()
            
            if not conversation:
                raise HTTPException(
                    status_code=404,
                    detail=f"Conversation {conversation_id} not found"
                )
            
            # Get the next sequence number
            result = await self.db.execute(
                select(Message)
                .where(Message.conversation_id == uuid_id)
                .order_by(Message.sequence.desc())
            )
            last_message = result.scalars().first()
            
            sequence = 0 if not last_message else last_message.sequence + 1
            
            # Create message
            message = Message(
                conversation_id=uuid_id,
                role=role,
                content=content,
                sequence=sequence,
                function_name=function_name,
                function_arguments=jsonable_encoder(function_arguments) if function_arguments else None
            )
            
            self.db.add(message)
            await self.db.commit()
            await self.db.refresh(message)
            
            # Return message info
            return {
                "id": str(message.id),
                "conversation_id": str(message.conversation_id),
                "role": message.role.value,
                "content": message.content,
                "sequence": message.sequence,
                "created_at": message.created_at.isoformat(),
                "function_name": message.function_name,
                "function_arguments": message.function_arguments
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error adding message to conversation: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error adding message to conversation: {str(e)}"
            )
    
    async def get_conversation(
        self,
        conversation_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Get a conversation with all its messages.
        
        Args:
            conversation_id: ID of the conversation
            user_id: ID of the user requesting the conversation
            
        Returns:
            Dictionary with conversation information and messages
        """
        try:
            # Check if conversation exists
            uuid_id = uuid.UUID(conversation_id)
            uuid_user = uuid.UUID(user_id)
            
            result = await self.db.execute(
                select(Conversation)
                .where(Conversation.id == uuid_id)
                .where(Conversation.user_id == uuid_user)
            )
            conversation = result.scalars().first()
            
            if not conversation:
                raise HTTPException(
                    status_code=404,
                    detail=f"Conversation {conversation_id} not found or access denied"
                )
            
            # Get all messages for the conversation
            result = await self.db.execute(
                select(Message)
                .where(Message.conversation_id == uuid_id)
                .order_by(Message.sequence)
            )
            messages = result.scalars().all()
            
            # Format response
            return {
                "id": str(conversation.id),
                "title": conversation.title,
                "description": conversation.description,
                "created_at": conversation.created_at.isoformat(),
                "updated_at": conversation.updated_at.isoformat(),
                "model_name": conversation.model_name,
                "model_params": conversation.model_params,
                "system_prompt": conversation.system_prompt,
                "is_active": conversation.is_active,
                "retrieved_document_ids": [
                    str(doc_id) for doc_id in conversation.retrieved_document_ids
                ],
                "messages": [
                    {
                        "id": str(message.id),
                        "role": message.role.value,
                        "content": message.content,
                        "sequence": message.sequence,
                        "created_at": message.created_at.isoformat(),
                        "function_name": message.function_name,
                        "function_arguments": message.function_arguments,
                        "source_documents": [
                            str(doc_id) for doc_id in message.source_documents
                        ] if message.source_documents else []
                    }
                    for message in messages
                ]
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting conversation: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error getting conversation: {str(e)}"
            )
    
    async def list_conversations(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        List conversations for a user.
        
        Args:
            user_id: ID of the user
            limit: Maximum number of conversations to return
            offset: Offset for pagination
            
        Returns:
            Dictionary with list of conversations
        """
        try:
            uuid_user = uuid.UUID(user_id)
            
            # Get total count
            result = await self.db.execute(
                select(Conversation)
                .where(Conversation.user_id == uuid_user)
                .order_by(Conversation.updated_at.desc())
            )
            all_conversations = result.scalars().all()
            total_count = len(all_conversations)
            
            # Get paginated conversations
            result = await self.db.execute(
                select(Conversation)
                .where(Conversation.user_id == uuid_user)
                .order_by(Conversation.updated_at.desc())
                .limit(limit)
                .offset(offset)
            )
            conversations = result.scalars().all()
            
            # Format response
            return {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "conversations": [
                    {
                        "id": str(conv.id),
                        "title": conv.title,
                        "created_at": conv.created_at.isoformat(),
                        "updated_at": conv.updated_at.isoformat(),
                        "model_name": conv.model_name,
                        "is_active": conv.is_active
                    }
                    for conv in conversations
                ]
            }
            
        except Exception as e:
            logger.error(f"Error listing conversations: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error listing conversations: {str(e)}"
            )
    
    async def process_conversation_message(
        self,
        conversation_id: str,
        user_message: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Process a new user message in a conversation.
        
        Args:
            conversation_id: ID of the conversation
            user_message: User message content
            user_id: ID of the user
            
        Returns:
            Dictionary with assistant's response
        """
        try:
            # Check if conversation exists
            uuid_id = uuid.UUID(conversation_id)
            uuid_user = uuid.UUID(user_id)
            
            result = await self.db.execute(
                select(Conversation)
                .where(Conversation.id == uuid_id)
                .where(Conversation.user_id == uuid_user)
            )
            conversation = result.scalars().first()
            
            if not conversation:
                raise HTTPException(
                    status_code=404,
                    detail=f"Conversation {conversation_id} not found or access denied"
                )
            
            # Get conversation history
            result = await self.db.execute(
                select(Message)
                .where(Message.conversation_id == uuid_id)
                .order_by(Message.sequence)
            )
            messages = result.scalars().all()
            
            # Add user message to conversation
            user_message_info = await self.add_message_to_conversation(
                conversation_id=conversation_id,
                role=MessageRole.USER,
                content=user_message
            )
            
            # Prepare messages for chat completion
            chat_messages = []
            
            for message in messages:
                chat_messages.append(ChatMessage(
                    role=message.role,
                    content=message.content,
                    function_name=message.function_name,
                    function_arguments=message.function_arguments
                ))
            
            # Add the new user message
            chat_messages.append(ChatMessage(
                role=MessageRole.USER,
                content=user_message
            ))
            
            # Prepare chat completion request
            chat_request = ChatCompletionRequest(
                messages=chat_messages,
                model=conversation.model_name,
                temperature=conversation.model_params.get("temperature", 0.7),
                max_tokens=conversation.model_params.get("max_tokens"),
                retrieve=True,  # Enable retrieval by default
                retrieval_options={
                    "top_k": 5,
                    "filters": None
                }
            )
            
            # Process the chat completion
            response = await self.process_chat_completion(chat_request, user_id)
            
            # Get the assistant's response
            if not response.choices or len(response.choices) == 0:
                raise HTTPException(
                    status_code=500,
                    detail="No response generated"
                )
            
            assistant_message = response.choices[0].message
            
            # Add assistant's response to conversation
            assistant_message_info = await self.add_message_to_conversation(
                conversation_id=conversation_id,
                role=MessageRole.ASSISTANT,
                content=assistant_message.content or "",
                function_name=assistant_message.function_call.name if assistant_message.function_call else None,
                function_arguments=json.loads(assistant_message.function_call.arguments) 
                    if assistant_message.function_call and assistant_message.function_call.arguments 
                    else None
            )
            
            # Update conversation with retrieved document IDs
            if response.source_documents:
                source_doc_ids = []
                for doc in response.source_documents:
                    if "id" in doc:
                        try:
                            doc_id = uuid.UUID(doc["id"])
                            source_doc_ids.append(doc_id)
                        except ValueError:
                            continue
                
                if source_doc_ids:
                    # Update the message with source documents
                    message_id = uuid.UUID(assistant_message_info["id"])
                    result = await self.db.execute(select(Message).where(Message.id == message_id))
                    message = result.scalars().first()
                    
                    if message:
                        message.source_documents = source_doc_ids
                        await self.db.commit()
                    
                    # Also update the conversation's retrieved document IDs
                    existing_ids = conversation.retrieved_document_ids or []
                    conversation.retrieved_document_ids = list(set(existing_ids + source_doc_ids))
                    await self.db.commit()
            
            # Return the assistant's response
            return {
                "conversation_id": conversation_id,
                "message": assistant_message_info,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "source_documents": response.source_documents
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing conversation message: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing conversation message: {str(e)}"
            )


async def get_orchestrator(
    db: AsyncSession = Depends(get_db),
    model_router: ModelRouter = Depends(get_model_router)
) -> OrchestratorService:
    """
    Dependency that provides an OrchestratorService instance.
    
    Args:
        db: Database session
        model_router: Model router service
        
    Returns:
        OrchestratorService instance
    """
    return OrchestratorService(db, model_router)