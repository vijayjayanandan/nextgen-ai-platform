import time
import uuid
from typing import Dict, List, Optional, Any, AsyncIterator, Union, cast
import httpx
import tiktoken
from fastapi import HTTPException

from app.core.config import settings
from app.core.logging import get_logger
from app.services.llm.base import LLMService
from app.schemas.chat import ChatMessage, ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseMessage, ChatCompletionResponseUsage, FunctionCall
from app.schemas.completion import CompletionResponse, CompletionResponseChoice, CompletionResponseUsage

logger = get_logger(__name__)


class OpenAIService(LLMService):
    """
    Service for interacting with OpenAI API.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None):
        """
        Initialize the OpenAI service.
        
        Args:
            api_key: OpenAI API key. If not provided, uses settings.OPENAI_API_KEY
            api_base: OpenAI API base URL. If not provided, uses settings.OPENAI_API_BASE
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.api_base = api_base or settings.OPENAI_API_BASE
        
        if not self.api_key:
            logger.error("OpenAI API key not provided")
            raise ValueError("OpenAI API key is required")
    
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
        Generate a completion using OpenAI API.
        """
        url = f"{self.api_base}/completions"
        
        # Prepare request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
            
        if stop is not None:
            payload["stop"] = stop
            
        if user is not None:
            payload["user"] = user
            
        # Add any additional parameters
        payload.update(kwargs)
        
        # Call OpenAI API
        try:
            async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"OpenAI API error: {response.text}"
                    )
                
                result = response.json()
                
                # Map OpenAI response to our schema
                choices = [
                    CompletionResponseChoice(
                        text=choice["text"],
                        index=choice["index"],
                        logprobs=choice.get("logprobs"),
                        finish_reason=choice["finish_reason"]
                    )
                    for choice in result["choices"]
                ]
                
                usage = CompletionResponseUsage(
                    prompt_tokens=result["usage"]["prompt_tokens"],
                    completion_tokens=result["usage"]["completion_tokens"],
                    total_tokens=result["usage"]["total_tokens"]
                )
                
                return CompletionResponse(
                    id=result["id"],
                    object=result["object"],
                    created=result["created"],
                    model=result["model"],
                    choices=choices,
                    usage=usage
                )
                
        except httpx.TimeoutException:
            logger.error(f"OpenAI API timeout for model {model}")
            raise HTTPException(
                status_code=504,
                detail="Request to OpenAI API timed out"
            )
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error calling OpenAI API: {str(e)}"
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
        Generate a chat completion using OpenAI API.
        """
        url = f"{self.api_base}/chat/completions"
        
        # Convert messages to OpenAI format
        openai_messages = []
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
                    else str(msg.function_arguments)
                )
                
            openai_messages.append(message_dict)
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": openai_messages,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
            
        if stop is not None:
            payload["stop"] = stop
            
        if functions is not None:
            payload["functions"] = functions
            
        if function_call is not None:
            payload["function_call"] = function_call
            
        if user is not None:
            payload["user"] = user
            
        # Add any additional parameters
        payload.update(kwargs)
        
        # Call OpenAI API
        try:
            async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"OpenAI API error: {response.text}"
                    )
                
                result = response.json()
                
                # Map OpenAI response to our schema
                choices = []
                for choice in result["choices"]:
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
                    prompt_tokens=result["usage"]["prompt_tokens"],
                    completion_tokens=result["usage"]["completion_tokens"],
                    total_tokens=result["usage"]["total_tokens"]
                )
                
                return ChatCompletionResponse(
                    id=result["id"],
                    object=result["object"],
                    created=result["created"],
                    model=result["model"],
                    choices=choices,
                    usage=usage
                )
                
        except httpx.TimeoutException:
            logger.error(f"OpenAI API timeout for model {model}")
            raise HTTPException(
                status_code=504,
                detail="Request to OpenAI API timed out"
            )
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error calling OpenAI API: {str(e)}"
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
        Stream a completion from OpenAI API.
        """
        url = f"{self.api_base}/completions"
        
        # Prepare request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
            
        if stop is not None:
            payload["stop"] = stop
            
        if user is not None:
            payload["user"] = user
            
        # Add any additional parameters
        payload.update(kwargs)
        
        # Call OpenAI API with streaming
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.text()
                        logger.error(f"OpenAI API error: {response.status_code} - {error_text}")
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"OpenAI API error: {error_text}"
                        )
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            line = line[6:]  # Remove "data: " prefix
                            
                            if line.strip() == "[DONE]":
                                break
                            
                            try:
                                chunk = httpx.loads(line)
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    content = chunk["choices"][0].get("text", "")
                                    if content:
                                        yield content
                            except Exception as e:
                                logger.warning(f"Error parsing stream chunk: {e}")
                                continue
                
        except httpx.TimeoutException:
            logger.error(f"OpenAI API timeout for streaming model {model}")
            raise HTTPException(
                status_code=504,
                detail="Streaming request to OpenAI API timed out"
            )
        except Exception as e:
            logger.error(f"Error streaming from OpenAI API: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error streaming from OpenAI API: {str(e)}"
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
        Stream a chat completion from OpenAI API.
        """
        url = f"{self.api_base}/chat/completions"
        
        # Convert messages to OpenAI format
        openai_messages = []
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
                    else str(msg.function_arguments)
                )
                
            openai_messages.append(message_dict)
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": openai_messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
            
        if stop is not None:
            payload["stop"] = stop
            
        if functions is not None:
            payload["functions"] = functions
            
        if function_call is not None:
            payload["function_call"] = function_call
            
        if user is not None:
            payload["user"] = user
            
        # Add any additional parameters
        payload.update(kwargs)
        
        # Call OpenAI API with streaming
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.text()
                        logger.error(f"OpenAI API error: {response.status_code} - {error_text}")
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"OpenAI API error: {error_text}"
                        )
                    
                    # For function calling, we'll accumulate the function call
                    function_name = None
                    function_args = ""
                    
                    # For content, we'll stream normally
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            line = line[6:]  # Remove "data: " prefix
                            
                            if line.strip() == "[DONE]":
                                break
                            
                            try:
                                chunk = httpx.loads(line)
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    delta = chunk["choices"][0].get("delta", {})
                                    
                                    # Handle function calls
                                    if "function_call" in delta:
                                        if "name" in delta["function_call"]:
                                            function_name = delta["function_call"]["name"]
                                        
                                        if "arguments" in delta["function_call"]:
                                            function_args += delta["function_call"]["arguments"]
                                            # Don't yield anything for function calls
                                            continue
                                    
                                    # Handle normal content
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                            except Exception as e:
                                logger.warning(f"Error parsing stream chunk: {e}")
                                continue
                
                    # After streaming is done, if we have a function call, yield it as JSON
                    if function_name:
                        yield f"\nFunction call: {function_name}({function_args})"
                
        except httpx.TimeoutException:
            logger.error(f"OpenAI API timeout for streaming model {model}")
            raise HTTPException(
                status_code=504,
                detail="Streaming request to OpenAI API timed out"
            )
        except Exception as e:
            logger.error(f"Error streaming from OpenAI API: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error streaming from OpenAI API: {str(e)}"
            )
    
    async def count_tokens(self, text: str, model: str) -> int:
        """
        Count the number of tokens in a text string.
        Uses tiktoken library for OpenAI models.
        """
        try:
            # Get the right encoding for the model
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except KeyError:
            # Fallback to cl100k_base for unknown models
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            # Rough approximation if tiktoken fails
            return len(text.split()) * 4 // 3  # Rough approximation of 4 chars per 3 tokens