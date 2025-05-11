from typing import Dict, List, Optional, Any, Union, Tuple, AsyncIterator
from fastapi import HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.config import settings
from app.core.logging import get_logger, audit_log
from app.db.session import get_db
from app.models.model import Model, ModelProvider, ModelDeploymentType
from app.services.llm.base import LLMService
from app.services.llm.openai_service import OpenAIService
from app.services.llm.anthropic_service import AnthropicService
from app.services.llm.on_prem_service import OnPremLLMService
from app.schemas.chat import ChatMessage, ChatCompletionResponse
from app.schemas.completion import CompletionResponse

logger = get_logger(__name__)


class ModelRouter:
    """
    Service that routes requests to the appropriate LLM service based on model, 
    data sensitivity, and other factors.
    """
    
    def __init__(self, db: AsyncSession):
        """
        Initialize the model router.
        
        Args:
            db: Database session for querying model information
        """
        self.db = db
        self.llm_services: Dict[str, LLMService] = {}
        
        # Initialize LLM services
        if settings.OPENAI_API_KEY:
            self.llm_services["openai"] = OpenAIService()
            
        if settings.ANTHROPIC_API_KEY:
            self.llm_services["anthropic"] = AnthropicService()
            
        if settings.ON_PREM_MODEL_ENABLED:
            self.llm_services["on_prem"] = OnPremLLMService()
    
    async def get_model_info(self, model_name: str) -> Tuple[Model, LLMService]:
        """
        Get model information and the appropriate LLM service.
        
        Args:
            model_name: Name of the model to use
            
        Returns:
            Tuple of (model_info, llm_service)
            
        Raises:
            HTTPException: If model not found or service not available
        """
        # Check if this is an on-prem model first
        if hasattr(settings, 'ON_PREM_MODELS') and model_name in settings.ON_PREM_MODELS:
            if "on_prem" not in self.llm_services:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"On-premises service not configured for model {model_name}"
                )
            # Create a specific instance of the OnPremLLMService for this model
            on_prem_service = OnPremLLMService(model_name=model_name)
            return None, on_prem_service
        
        # Query the model from the database
        result = await self.db.execute(select(Model).filter(Model.name == model_name))
        model = result.scalars().first()
        
        if not model:
            logger.warning(f"Model {model_name} not found in database")
            
            # Try to infer provider from model name for graceful handling
            if any(name in model_name for name in ["gpt", "davinci", "curie", "babbage", "ada"]):
                provider = "openai"
            elif any(name in model_name for name in ["claude"]):
                provider = "anthropic"
            elif any(name in model_name for name in ["llama"]):
                provider = "on_prem"
                # Create a specific Llama instance
                if "on_prem" in self.llm_services:
                    return None, OnPremLLMService(model_name="llama-7b")
            elif any(name in model_name for name in ["deepseek"]):
                provider = "on_prem"
                # Create a specific Deepseek instance
                if "on_prem" in self.llm_services:
                    return None, OnPremLLMService(model_name="deepseek-7b")
            else:
                provider = "on_prem"
                
            # Check if we have the service for this provider
            if provider not in self.llm_services:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model {model_name} not found and {provider} service not configured"
                )
                
            return None, self.llm_services[provider]
        
        # Get the appropriate service based on the model provider
        provider_key = model.provider.value
        
        if provider_key not in self.llm_services:
            raise HTTPException(
                status_code=400,
                detail=f"Service for provider {provider_key} not configured"
            )
            
        return model, self.llm_services[provider_key]
    
    async def route_sensitive_query(
        self, 
        query_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Determine if a query contains sensitive data that should be routed to on-prem models.
        
        Args:
            query_text: The query text to analyze
            metadata: Additional metadata about the query
            
        Returns:
            True if query should be routed to on-prem models, False otherwise
        """
        # Check metadata for explicit sensitivity flag
        if metadata and metadata.get("sensitive", False):
            return True
            
        # Check metadata for security classification
        if metadata and metadata.get("security_classification") in ["protected_a", "protected_b"]:
            return True
            
        # Simple keyword-based check
        sensitive_keywords = [
            "SIN", "social insurance", "passport", "health card", 
            "credit card", "protected", "confidential", "classified"
        ]
        
        if any(keyword.lower() in query_text.lower() for keyword in sensitive_keywords):
            return True
            
        # For more sophisticated detection, we could call a content filtering service here
        
        # Default to non-sensitive
        return False
    
    async def route_completion_request(
        self,
        prompt: str,
        model: str,
        user_id: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        n: int = 1,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        user: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> CompletionResponse:
        """
        Route a completion request to the appropriate LLM service.
        
        Args:
            prompt: The text prompt to complete
            model: The model to use
            user_id: ID of the user making the request
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            n: Number of completions to generate
            stop: Sequence(s) at which to stop generation
            presence_penalty: Presence penalty (-2 to 2)
            frequency_penalty: Frequency penalty (-2 to 2)
            user: End-user identifier
            metadata: Additional metadata about the request
            kwargs: Additional provider-specific parameters
            
        Returns:
            CompletionResponse object with generated completion(s)
        """
        # Check if query contains sensitive data
        is_sensitive = await self.route_sensitive_query(prompt, metadata)
        
        # Get model info and service
        model_info, llm_service = await self.get_model_info(model)
        
        # Override routing for sensitive data to on-prem if needed
        if is_sensitive and model_info and not model_info.allowed_for_protected_b:
            if "on_prem" not in self.llm_services:
                raise HTTPException(
                    status_code=400,
                    detail="Query contains Protected B data but on-prem service not available"
                )
                
            logger.info(f"Routing sensitive query to on-prem service instead of {model}")
            llm_service = self.llm_services["on_prem"]
            
            # Use the default on-prem model
            if model_info:
                model = "on_prem_default"  # This should be configured in settings
        
        # Log the request
        logger.info(f"Routing completion request for model {model} to {llm_service.__class__.__name__}")
        
        # Audit log the request
        audit_log(
            user_id=user_id,
            action="inference",
            resource_type="completion",
            resource_id=model,
            details={
                "model": model,
                "is_sensitive": is_sensitive,
                "provider": llm_service.__class__.__name__,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        
        # Call the LLM service
        return await llm_service.generate_completion(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            user=user,
            **kwargs
        )
    
    async def route_chat_completion_request(
        self,
        messages: List[ChatMessage],
        model: str,
        user_id: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        n: int = 1,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        user: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChatCompletionResponse:
        """
        Route a chat completion request to the appropriate LLM service.
        
        Args:
            messages: List of chat messages in the conversation
            model: The model to use
            user_id: ID of the user making the request
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            n: Number of completions to generate
            stop: Sequence(s) at which to stop generation
            presence_penalty: Presence penalty (-2 to 2)
            frequency_penalty: Frequency penalty (-2 to 2)
            functions: List of function definitions
            function_call: Controls function calling behavior
            user: End-user identifier
            metadata: Additional metadata about the request
            kwargs: Additional provider-specific parameters
            
        Returns:
            ChatCompletionResponse object with generated response(s)
        """
        # Check if query contains sensitive data (using the last user message)
        user_messages = [msg for msg in messages if msg.role == "user"]
        is_sensitive = False
        
        if user_messages:
            is_sensitive = await self.route_sensitive_query(
                user_messages[-1].content, 
                metadata
            )
        
        # Get model info and service
        model_info, llm_service = await self.get_model_info(model)
        
        # Override routing for sensitive data to on-prem if needed
        if is_sensitive and model_info and not model_info.allowed_for_protected_b:
            if "on_prem" not in self.llm_services:
                raise HTTPException(
                    status_code=400,
                    detail="Query contains Protected B data but on-prem service not available"
                )
                
            logger.info(f"Routing sensitive chat to on-prem service instead of {model}")
            llm_service = self.llm_services["on_prem"]
            
            # Use the default on-prem model
            if model_info:
                model = "on_prem_default"  # This should be configured in settings
        
        # Log the request
        logger.info(f"Routing chat completion request for model {model} to {llm_service.__class__.__name__}")
        
        # Audit log the request
        audit_log(
            user_id=user_id,
            action="inference",
            resource_type="chat_completion",
            resource_id=model,
            details={
                "model": model,
                "is_sensitive": is_sensitive,
                "provider": llm_service.__class__.__name__,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "message_count": len(messages),
            }
        )
        
        # Call the LLM service
        return await llm_service.generate_chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            functions=functions,
            function_call=function_call,
            user=user,
            **kwargs
        )
    
    async def route_stream_completion_request(
        self,
        prompt: str,
        model: str,
        user_id: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        user: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Route a streaming completion request to the appropriate LLM service.
        
        Args:
            prompt: The text prompt to complete
            model: The model to use
            user_id: ID of the user making the request
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            stop: Sequence(s) at which to stop generation
            presence_penalty: Presence penalty (-2 to 2)
            frequency_penalty: Frequency penalty (-2 to 2)
            user: End-user identifier
            metadata: Additional metadata about the request
            kwargs: Additional provider-specific parameters
            
        Returns:
            AsyncIterator yielding completion text chunks
        """
        # Check if query contains sensitive data
        is_sensitive = await self.route_sensitive_query(prompt, metadata)
        
        # Get model info and service
        model_info, llm_service = await self.get_model_info(model)
        
        # Override routing for sensitive data to on-prem if needed
        if is_sensitive and model_info and not model_info.allowed_for_protected_b:
            if "on_prem" not in self.llm_services:
                raise HTTPException(
                    status_code=400,
                    detail="Query contains Protected B data but on-prem service not available"
                )
                
            logger.info(f"Routing sensitive streaming query to on-prem service instead of {model}")
            llm_service = self.llm_services["on_prem"]
            
            # Use the default on-prem model
            if model_info:
                model = "on_prem_default"  # This should be configured in settings
        
        # Log the request
        logger.info(f"Routing streaming completion request for model {model} to {llm_service.__class__.__name__}")
        
        # Audit log the request
        audit_log(
            user_id=user_id,
            action="inference",
            resource_type="stream_completion",
            resource_id=model,
            details={
                "model": model,
                "is_sensitive": is_sensitive,
                "provider": llm_service.__class__.__name__,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        
        # Call the LLM service
        return llm_service.stream_completion(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            user=user,
            **kwargs
        )
    
    async def route_stream_chat_completion_request(
        self,
        messages: List[ChatMessage],
        model: str,
        user_id: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        user: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Route a streaming chat completion request to the appropriate LLM service.
        
        Args:
            messages: List of chat messages in the conversation
            model: The model to use
            user_id: ID of the user making the request
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            stop: Sequence(s) at which to stop generation
            presence_penalty: Presence penalty (-2 to 2)
            frequency_penalty: Frequency penalty (-2 to 2)
            functions: List of function definitions
            function_call: Controls function calling behavior
            user: End-user identifier
            metadata: Additional metadata about the request
            kwargs: Additional provider-specific parameters
            
        Returns:
            AsyncIterator yielding chat response text chunks
        """
        # Check if query contains sensitive data (using the last user message)
        user_messages = [msg for msg in messages if msg.role == "user"]
        is_sensitive = False
        
        if user_messages:
            is_sensitive = await self.route_sensitive_query(
                user_messages[-1].content, 
                metadata
            )
        
        # Get model info and service
        model_info, llm_service = await self.get_model_info(model)
        
        # Override routing for sensitive data to on-prem if needed
        if is_sensitive and model_info and not model_info.allowed_for_protected_b:
            if "on_prem" not in self.llm_services:
                raise HTTPException(
                    status_code=400,
                    detail="Query contains Protected B data but on-prem service not available"
                )
                
            logger.info(f"Routing sensitive streaming chat to on-prem service instead of {model}")
            llm_service = self.llm_services["on_prem"]
            
            # Use the default on-prem model
            if model_info:
                model = "on_prem_default"  # This should be configured in settings
        
        # Log the request
        logger.info(f"Routing streaming chat completion request for model {model} to {llm_service.__class__.__name__}")
        
        # Audit log the request
        audit_log(
            user_id=user_id,
            action="inference",
            resource_type="stream_chat_completion",
            resource_id=model,
            details={
                "model": model,
                "is_sensitive": is_sensitive,
                "provider": llm_service.__class__.__name__,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "message_count": len(messages),
            }
        )
        
        # Call the LLM service
        return llm_service.stream_chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            functions=functions,
            function_call=function_call,
            user=user,
            **kwargs
        )


async def get_model_router(db: AsyncSession = Depends(get_db)) -> ModelRouter:
    """
    Dependency that provides a ModelRouter instance.
    
    Args:
        db: Database session
        
    Returns:
        ModelRouter instance
    """
    return ModelRouter(db)