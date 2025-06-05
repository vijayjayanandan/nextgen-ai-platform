import time
import uuid
import json
import httpx
from typing import Dict, List, Optional, Any, AsyncIterator, Union
from fastapi import HTTPException

from app.core.config import settings
from app.core.logging import get_logger
from app.services.llm.base import LLMService
from app.services.llm.adapters.base import ModelAdapter
from app.services.llm.adapters.openai_compatible import OpenAICompatibleAdapter
from app.services.llm.adapters.llama import LlamaAdapter
from app.services.llm.adapters.deepseek import DeepseekAdapter
from app.services.llm.adapters.ollama import OllamaAdapter
from app.schemas.chat import ChatMessage, ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseMessage, ChatCompletionResponseUsage, FunctionCall
from app.schemas.completion import CompletionResponse, CompletionResponseChoice, CompletionResponseUsage
from app.models.chat import MessageRole

logger = get_logger(__name__)


class OnPremLLMService(LLMService):
    """
    Service for interacting with on-premises LLM inference endpoints.
    This service supports secure inference for Protected B data.
    """
    
    def __init__(self, model_name: Optional[str] = None, endpoint_url: Optional[str] = None):
        """
        Initialize the on-prem LLM service.
        
        Args:
            model_name: Name of the specific model to use
            endpoint_url: URL of the on-prem inference service. If not provided, 
                         uses settings from ON_PREM_MODELS configuration
        """
        self.model_name = model_name or "on_prem_default"
        
        # Get model configuration
        if hasattr(settings, 'ON_PREM_MODELS') and self.model_name in settings.ON_PREM_MODELS:
            self.model_config = settings.ON_PREM_MODELS[self.model_name]
            self.endpoint_url = endpoint_url or self.model_config.get("endpoint")
            self.model_type = self.model_config.get("model_type", "default")
            # Auto-detect Ollama models
            if self.model_config.get("base_model") == "ollama" or "ollama" in self.model_name:
                self.model_type = "ollama"
        else:
            # Fallback to legacy configuration
            self.endpoint_url = endpoint_url or settings.ON_PREM_MODEL_ENDPOINT
            self.model_type = "default"
            self.model_config = {}
        
        if not self.endpoint_url and settings.ON_PREM_MODEL_ENABLED:
            logger.error("On-premises model endpoint not provided")
            raise ValueError("On-premises model endpoint is required when ON_PREM_MODEL_ENABLED is True")
            
        # Get the appropriate adapter for this model type
        self.adapter = self._get_adapter(self.model_type)
        
    def update_from_model(self, model):
        """Update configuration from database model."""
        if model:
            self.endpoint_url = model.endpoint_url
            if model.base_model == "ollama":
                self.model_type = "ollama"
                # IMPORTANT: Update the adapter when model type changes
                self.adapter = self._get_adapter("ollama")
            self.model_config = model.default_parameters or {}

    def _get_adapter(self, model_type: str) -> ModelAdapter:
        """
        Get the appropriate adapter for the model type.
        
        Args:
            model_type: Type of model ("llama", "deepseek", etc.)
            
        Returns:
            ModelAdapter instance for the specified model type
        """
        adapters = {
            "llama": LlamaAdapter(),
            "deepseek": DeepseekAdapter(),
            "ollama": OllamaAdapter(),
            "default": OpenAICompatibleAdapter()
        }
        return adapters.get(model_type, adapters["default"])
    
    async def generate_completion(
        self,
        prompt: str,
        model: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        n: int = 1,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        user: Optional[str] = None,
        **kwargs
    ) -> CompletionResponse:
        """
        Generate a completion using the on-prem inference service.
        """
        if not settings.ON_PREM_MODEL_ENABLED:
            raise HTTPException(
                status_code=400,
                detail="On-premises model inference is not enabled"
            )
        
        # For Ollama, the endpoint already includes the full path
        if self.model_type == "ollama" or self.endpoint_url.endswith("/api/generate"):
            url = self.endpoint_url
        else:
            url = f"{self.endpoint_url}/completions"
        
        # Get actual model max tokens if not specified
        if max_tokens is None and self.model_config:
            max_tokens = self.model_config.get("max_tokens")
        
        # Prepare request payload using the adapter
        payload = await self.adapter.prepare_completion_payload(
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
        
        # Call on-prem API
        try:
            async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code != 200:
                    logger.error(f"On-prem API error: {response.status_code} - {response.text}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"On-prem API error: {response.text}"
                    )
                
                result = response.json()
                
                # Parse response using the adapter
                logger.info(f"Using adapter: {self.adapter.__class__.__name__} for model_type: {self.model_type}")
                logger.debug(f"Raw result from API: {result}")
                parsed_result = await self.adapter.parse_completion_response(result)
                
                # Map response to our schema
                choices = [
                    CompletionResponseChoice(
                        text=choice["text"],
                        index=choice["index"],
                        logprobs=choice.get("logprobs"),
                        finish_reason=choice["finish_reason"]
                    )
                    for choice in parsed_result["choices"]
                ]
                
                usage = CompletionResponseUsage(
                    prompt_tokens=parsed_result["usage"]["prompt_tokens"],
                    completion_tokens=parsed_result["usage"]["completion_tokens"],
                    total_tokens=parsed_result["usage"]["total_tokens"]
                )
                
                return CompletionResponse(
                    id=parsed_result.get("id", str(uuid.uuid4())),
                    object=parsed_result.get("object", "text_completion"),
                    created=parsed_result.get("created", int(time.time())),
                    model=parsed_result.get("model", model),
                    choices=choices,
                    usage=usage
                )
                
        except httpx.TimeoutException:
            logger.error(f"On-prem API timeout for model {model}")
            raise HTTPException(
                status_code=504,
                detail="Request to on-prem inference service timed out"
            )
        except Exception as e:
            logger.error(f"Error calling on-prem inference service: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error calling on-prem inference service: {str(e)}"
            )
    
    async def generate_chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
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
        **kwargs
    ) -> ChatCompletionResponse:
        """
        Generate a chat completion using the on-prem inference service.
        """
        if not settings.ON_PREM_MODEL_ENABLED:
            raise HTTPException(
                status_code=400,
                detail="On-premises model inference is not enabled"
            )
        
        # For Ollama, use the same endpoint for chat
        if self.model_type == "ollama" or self.endpoint_url.endswith("/api/generate"):
            url = self.endpoint_url
        else:
            url = f"{self.endpoint_url}/chat/completions"
        
        # Get actual model max tokens if not specified
        if max_tokens is None and self.model_config:
            max_tokens = self.model_config.get("max_tokens")
        
        # Convert messages to the format expected by the on-prem service
        formatted_messages = []
        for msg in messages:
            message_dict = {
                "role": msg.role.value,
                "content": msg.content
            }
            
            if msg.name:
                message_dict["name"] = msg.name
                
            if msg.function_name and msg.role.value == "function":
                message_dict["name"] = msg.function_name
                message_dict["content"] = (
                    msg.function_arguments 
                    if isinstance(msg.function_arguments, str) 
                    else json.dumps(msg.function_arguments)
                )
                
            formatted_messages.append(message_dict)
        
        # Prepare request payload using the adapter
        payload = await self.adapter.prepare_chat_completion_payload(
            messages=formatted_messages,
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
        
        # Call on-prem API
        try:
            async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code != 200:
                    logger.error(f"On-prem API error: {response.status_code} - {response.text}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"On-prem API error: {response.text}"
                    )
                
                result = response.json()
                
                # Parse response using the adapter
                parsed_result = await self.adapter.parse_chat_completion_response(result)
                
                # Map on-prem response to our schema
                choices = []
                for choice in parsed_result["choices"]:
                    message = choice["message"]
                    
                    # Handle function calls
                    function_call_obj = None
                    if "function_call" in message:
                        function_call_obj = FunctionCall(
                            name=message["function_call"]["name"],
                            arguments=message["function_call"]["arguments"]
                        )
                    
                    response_message = ChatCompletionResponseMessage(
                        role=message["role"],
                        content=message.get("content"),
                        function_call=function_call_obj
                    )
                    
                    choices.append(
                        ChatCompletionResponseChoice(
                            index=choice["index"],
                            message=response_message,
                            finish_reason=choice["finish_reason"]
                        )
                    )
                
                usage = ChatCompletionResponseUsage(
                    prompt_tokens=parsed_result["usage"]["prompt_tokens"],
                    completion_tokens=parsed_result["usage"]["completion_tokens"],
                    total_tokens=parsed_result["usage"]["total_tokens"]
                )
                
                return ChatCompletionResponse(
                    id=parsed_result.get("id", str(uuid.uuid4())),
                    object=parsed_result.get("object", "chat.completion"),
                    created=parsed_result.get("created", int(time.time())),
                    model=parsed_result.get("model", model),
                    choices=choices,
                    usage=usage
                )
                
        except httpx.TimeoutException:
            logger.error(f"On-prem API timeout for model {model}")
            raise HTTPException(
                status_code=504,
                detail="Request to on-prem inference service timed out"
            )
        except Exception as e:
            logger.error(f"Error calling on-prem inference service: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error calling on-prem inference service: {str(e)}"
            )
    
    async def stream_completion(
        self,
        prompt: str,
        model: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        user: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream a completion from the on-prem inference service.
        """
        if not settings.ON_PREM_MODEL_ENABLED:
            raise HTTPException(
                status_code=400,
                detail="On-premises model inference is not enabled"
            )
        
        # For Ollama, the endpoint already includes the full path
        if self.model_type == "ollama" or self.endpoint_url.endswith("/api/generate"):
            url = self.endpoint_url
        else:
            url = f"{self.endpoint_url}/completions"
        
        # Get actual model max tokens if not specified
        if max_tokens is None and self.model_config:
            max_tokens = self.model_config.get("max_tokens")
            
        # Prepare request payload using the adapter
        payload = await self.adapter.prepare_completion_payload(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            user=user,
            **kwargs
        )
        
        # Call on-prem API with streaming
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.text()
                        logger.error(f"On-prem API error: {response.status_code} - {error_text}")
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"On-prem API error: {error_text}"
                        )
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            line = line[6:]  # Remove "data: " prefix
                            
                            if line.strip() == "[DONE]":
                                break
                            
                            try:
                                chunk = json.loads(line)
                                # Use adapter to parse streaming chunk
                                content = await self.adapter.parse_streaming_chunk(chunk)
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue
                
        except httpx.TimeoutException:
            logger.error(f"On-prem API timeout for streaming model {model}")
            raise HTTPException(
                status_code=504,
                detail="Streaming request to on-prem inference service timed out"
            )
        except Exception as e:
            logger.error(f"Error streaming from on-prem inference service: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error streaming from on-prem inference service: {str(e)}"
            )
    
    async def stream_chat_completion(
        self,
        messages: List[ChatMessage],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        user: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream a chat completion from the on-prem inference service.
        """
        if not settings.ON_PREM_MODEL_ENABLED:
            raise HTTPException(
                status_code=400,
                detail="On-premises model inference is not enabled"
            )
        
        # For Ollama, use the same endpoint for chat
        if self.model_type == "ollama" or self.endpoint_url.endswith("/api/generate"):
            url = self.endpoint_url
        else:
            url = f"{self.endpoint_url}/chat/completions"
        
        # Get actual model max tokens if not specified
        if max_tokens is None and self.model_config:
            max_tokens = self.model_config.get("max_tokens")
            
        # Convert messages to the format expected by the on-prem service
        formatted_messages = []
        for msg in messages:
            message_dict = {
                "role": msg.role.value,
                "content": msg.content
            }
            
            if msg.name:
                message_dict["name"] = msg.name
                
            if msg.function_name and msg.role.value == "function":
                message_dict["name"] = msg.function_name
                message_dict["content"] = (
                    msg.function_arguments 
                    if isinstance(msg.function_arguments, str) 
                    else json.dumps(msg.function_arguments)
                )
                
            formatted_messages.append(message_dict)
        
        # Prepare request payload using the adapter
        payload = await self.adapter.prepare_chat_completion_payload(
            messages=formatted_messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            functions=functions,
            function_call=function_call,
            user=user,
            **kwargs
        )
        
        # Call on-prem API with streaming
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.text()
                        logger.error(f"On-prem API error: {response.status_code} - {error_text}")
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"On-prem API error: {error_text}"
                        )
                    
                    function_name = None
                    function_args = ""
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            line = line[6:]  # Remove "data: " prefix
                            
                            if line.strip() == "[DONE]":
                                break
                            
                            try:
                                chunk = json.loads(line)
                                # Use adapter to parse streaming chunk
                                content = await self.adapter.parse_streaming_chunk(chunk)
                                if content:
                                    yield content
                                    
                                # Check for function calls in the chunk
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    delta = chunk["choices"][0].get("delta", {})
                                    
                                    # Handle function calls (OpenAI format)
                                    if "function_call" in delta:
                                        if "name" in delta["function_call"]:
                                            function_name = delta["function_call"]["name"]
                                        
                                        if "arguments" in delta["function_call"]:
                                            function_args += delta["function_call"]["arguments"]
                                    
                                    # Handle tool calls (Claude/Llama format)
                                    if "tool_calls" in delta and len(delta["tool_calls"]) > 0:
                                        tool_call = delta["tool_calls"][0]
                                        if "function" in tool_call:
                                            if "name" in tool_call["function"]:
                                                function_name = tool_call["function"]["name"]
                                            
                                            if "arguments" in tool_call["function"]:
                                                function_args += tool_call["function"]["arguments"]
                                
                            except json.JSONDecodeError:
                                continue
                    
                    # After streaming is done, if we have a function call, yield it as JSON
                    if function_name:
                        yield f"\nFunction call: {function_name}({function_args})"
                
        except httpx.TimeoutException:
            logger.error(f"On-prem API timeout for streaming model {model}")
            raise HTTPException(
                status_code=504,
                detail="Streaming request to on-prem inference service timed out"
            )
        except Exception as e:
            logger.error(f"Error streaming from on-prem inference service: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error streaming from on-prem inference service: {str(e)}"
            )
    
    async def count_tokens(self, text: str, model: str) -> int:
        """
        Count the number of tokens in a text string.
        For on-prem models, we might need to call the tokenize endpoint if available,
        otherwise use an approximation.
        """
        # If tokenize endpoint is available, use it
        url = f"{self.endpoint_url}/tokenize"
        
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.post(
                    url,
                    json={"text": text, "model": model},
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("token_count", 0)
        except Exception as e:
            logger.warning(f"Error using tokenize endpoint: {str(e)}")
            
        # If adapter has a specialized token counting method, use it
        if hasattr(self.adapter, "count_tokens"):
            return await self.adapter.count_tokens(text)
            
        # Fallback to model-specific approximation based on type
        if self.model_type == "llama":
            # Llama models typically use ~4.5 tokens per word
            return len(text.split()) * 9 // 2
        elif self.model_type == "deepseek":
            # Deepseek models average tokenization
            return len(text.split()) * 5 // 3
        else:
            # Generic approximation: ~1.3 tokens per word for English text
            return len(text.split()) * 4 // 3