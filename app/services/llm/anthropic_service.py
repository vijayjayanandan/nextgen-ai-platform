import time
import uuid
import json
from typing import Dict, List, Optional, Any, AsyncIterator, Union
import httpx
import re
from fastapi import HTTPException

from app.core.config import settings
from app.core.logging import get_logger
from app.services.llm.base import LLMService
from app.schemas.chat import ChatMessage, ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseMessage, ChatCompletionResponseUsage, FunctionCall
from app.schemas.completion import CompletionResponse, CompletionResponseChoice, CompletionResponseUsage
from app.models.chat import MessageRole

logger = get_logger(__name__)


class AnthropicService(LLMService):
    """
    Service for interacting with Anthropic Claude API.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None):
        """
        Initialize the Anthropic service.
        
        Args:
            api_key: Anthropic API key. If not provided, uses settings.ANTHROPIC_API_KEY
            api_base: Anthropic API base URL. If not provided, uses settings.ANTHROPIC_API_BASE
        """
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        self.api_base = api_base or settings.ANTHROPIC_API_BASE
        
        if not self.api_key:
            logger.error("Anthropic API key not provided")
            raise ValueError("Anthropic API key is required")
    
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
        Generate a completion using Anthropic API.
        Note: Anthropic doesn't support traditional completions, only chat completions.
        This method adapts the completion interface to Anthropic's chat API.
        """
        # Convert to chat completion format
        chat_message = ChatMessage(role=MessageRole.USER, content=prompt)
        chat_response = await self.generate_chat_completion(
            messages=[chat_message],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stop=stop,
            # Anthropic doesn't directly support these, but we'll include them for
            # parameter compatibility
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            user=user,
            **kwargs
        )
        
        # Convert chat completion to completion format
        choices = []
        for i, choice in enumerate(chat_response.choices):
            choices.append(
                CompletionResponseChoice(
                    text=choice.message.content or "",
                    index=i,
                    logprobs=None,
                    finish_reason=choice.finish_reason
                )
            )
        
        return CompletionResponse(
            id=chat_response.id,
            object="text_completion",
            created=chat_response.created,
            model=chat_response.model,
            choices=choices,
            usage=chat_response.usage,
            source_documents=chat_response.source_documents
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
        Generate a chat completion using Anthropic API.
        """
        url = f"{self.api_base}/v1/messages"
        
        # Convert messages to Anthropic format
        # Anthropic expects a specific format and doesn't support all the roles that OpenAI does
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_message = msg.content
            elif msg.role == MessageRole.USER:
                anthropic_messages.append({
                    "role": "user",
                    "content": msg.content
                })
            elif msg.role == MessageRole.ASSISTANT:
                anthropic_messages.append({
                    "role": "assistant",
                    "content": msg.content
                })
            elif msg.role == MessageRole.FUNCTION:
                # Anthropic doesn't directly support function messages
                # Convert to user messages with a specific format
                anthropic_messages.append({
                    "role": "user",
                    "content": f"Function {msg.function_name} returned: {msg.content}"
                })
        
        # Ensure we have at least one user message
        if not any(msg["role"] == "user" for msg in anthropic_messages):
            raise HTTPException(
                status_code=400,
                detail="At least one user message is required for Anthropic API"
            )
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens or 1024,
        }
        
        if system_message:
            payload["system"] = system_message
            
        if stop is not None:
            payload["stop_sequences"] = stop if isinstance(stop, list) else [stop]
        
        # Handle tool use with Claude's native tools feature if available
        if functions is not None and len(functions) > 0:
            # Check if this is a Claude model that supports tools
            # Currently Claude 3 Opus, Sonnet and Haiku support tools
            if any(model_name in model for model_name in ["claude-3", "claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]):
                payload["tools"] = functions
                
                if function_call is not None and function_call != "auto":
                    if isinstance(function_call, dict) and "name" in function_call:
                        payload["tool_choice"] = {"type": "tool", "name": function_call["name"]}
            else:
                logger.warning(f"Function calling not supported for model {model}, ignoring functions parameter")
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in payload:
                payload[key] = value
        
        # Call Anthropic API
        try:
            async with httpx.AsyncClient(timeout=settings.REQUEST_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json"
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Anthropic API error: {response.status_code} - {response.text}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Anthropic API error: {response.text}"
                    )
                
                result = response.json()
                
                # Extract tool calls if present
                content = result["content"][0]["text"]
                function_call_obj = None
                
                # Check for tool use in response
                if "tool_use" in result:
                    tool_use = result["tool_use"]
                    function_call_obj = FunctionCall(
                        name=tool_use["name"],
                        arguments=json.dumps(tool_use["input"])
                    )
                    # Remove tool use marker from content if present
                    content = re.sub(r'<tool_use>.*?</tool_use>', '', content, flags=re.DOTALL).strip()
                
                # Map Anthropic response to our schema
                message = ChatCompletionResponseMessage(
                    role=MessageRole.ASSISTANT,
                    content=content,
                    function_call=function_call_obj
                )
                
                # Anthropic doesn't provide detailed token usage, so we estimate
                prompt_tokens = await self.count_tokens("".join([m.content for m in messages]), model)
                completion_tokens = await self.count_tokens(content, model)
                
                choices = [
                    ChatCompletionResponseChoice(
                        index=0,
                        message=message,
                        finish_reason=result.get("stop_reason", "stop")
                    )
                ]
                
                usage = ChatCompletionResponseUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )
                
                return ChatCompletionResponse(
                    id=result["id"],
                    object="chat.completion",
                    created=int(time.time()),
                    model=result["model"],
                    choices=choices,
                    usage=usage
                )
                
        except httpx.TimeoutException:
            logger.error(f"Anthropic API timeout for model {model}")
            raise HTTPException(
                status_code=504,
                detail="Request to Anthropic API timed out"
            )
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error calling Anthropic API: {str(e)}"
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
        Stream a completion from Anthropic API.
        Adapts the completion interface to Anthropic's chat API.
        """
        # Convert to chat message format
        chat_message = ChatMessage(role=MessageRole.USER, content=prompt)
        
        # Use the chat streaming method
        async for chunk in self.stream_chat_completion(
            messages=[chat_message],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            user=user,
            **kwargs
        ):
            yield chunk
    
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
        Stream a chat completion from Anthropic API.
        """
        url = f"{self.api_base}/v1/messages"
        
        # Convert messages to Anthropic format
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_message = msg.content
            elif msg.role == MessageRole.USER:
                anthropic_messages.append({
                    "role": "user",
                    "content": msg.content
                })
            elif msg.role == MessageRole.ASSISTANT:
                anthropic_messages.append({
                    "role": "assistant",
                    "content": msg.content
                })
            elif msg.role == MessageRole.FUNCTION:
                # Convert function messages to user messages
                anthropic_messages.append({
                    "role": "user",
                    "content": f"Function {msg.function_name} returned: {msg.content}"
                })
        
        # Ensure we have at least one user message
        if not any(msg["role"] == "user" for msg in anthropic_messages):
            raise HTTPException(
                status_code=400,
                detail="At least one user message is required for Anthropic API"
            )
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens or 1024,
            "stream": True,
        }
        
        if system_message:
            payload["system"] = system_message
            
        if stop is not None:
            payload["stop_sequences"] = stop if isinstance(stop, list) else [stop]
        
        # Handle tool use with Claude's native tools feature
        if functions is not None and len(functions) > 0:
            # Check if this is a Claude model that supports tools
            if any(model_name in model for model_name in ["claude-3", "claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]):
                payload["tools"] = functions
                
                if function_call is not None and function_call != "auto":
                    if isinstance(function_call, dict) and "name" in function_call:
                        payload["tool_choice"] = {"type": "tool", "name": function_call["name"]}
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in payload:
                payload[key] = value
        
        # Call Anthropic API with streaming
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json"
                    }
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.text()
                        logger.error(f"Anthropic API error: {response.status_code} - {error_text}")
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"Anthropic API error: {error_text}"
                        )
                    
                    buffer = ""
                    tool_use_buffer = None
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            line = line[6:]  # Remove "data: " prefix
                            
                            if line.strip() == "[DONE]":
                                break
                                
                            try:
                                if line.strip():
                                    chunk = json.loads(line)
                                    
                                    # Handle tool use events
                                    if chunk.get("type") == "tool_use":
                                        tool_use_buffer = chunk
                                        continue
                                        
                                    # For content deltas, yield the text
                                    if chunk.get("type") == "content_block_delta" and "delta" in chunk:
                                        content = chunk["delta"].get("text", "")
                                        if content:
                                            # Check for and filter out tool use XML tags
                                            if "<tool_use>" in content or "</tool_use>" in content:
                                                # Add to buffer to check for complete tags
                                                buffer += content
                                                # Extract only non-tool parts for yielding
                                                parts = re.sub(r'<tool_use>.*?</tool_use>', '', buffer, flags=re.DOTALL).strip()
                                                # Only yield if we have non-tool content
                                                if parts and parts != buffer:
                                                    yield parts
                                                    buffer = ""
                                            else:
                                                yield content
                            except json.JSONDecodeError as e:
                                logger.warning(f"Error parsing stream chunk: {e}")
                                continue
                                
                    # If we have a tool use at the end, provide a simple text representation
                    if tool_use_buffer and "id" in tool_use_buffer:
                        tool_name = tool_use_buffer.get("name", "unknown")
                        tool_input = json.dumps(tool_use_buffer.get("input", {}))
                        yield f"\nFunction call: {tool_name}({tool_input})"
                
        except httpx.TimeoutException:
            logger.error(f"Anthropic API timeout for streaming model {model}")
            raise HTTPException(
                status_code=504,
                detail="Streaming request to Anthropic API timed out"
            )
        except Exception as e:
            logger.error(f"Error streaming from Anthropic API: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error streaming from Anthropic API: {str(e)}"
            )
    
    async def count_tokens(self, text: str, model: str) -> int:
        """
        Count the number of tokens in a text string.
        For Anthropic, we use a simple approximation since they don't provide a tokenizer.
        """
        # Simple approximation for Claude models
        # Claude docs suggest ~4 characters per token as a rough estimate
        return len(text) // 4 + 1