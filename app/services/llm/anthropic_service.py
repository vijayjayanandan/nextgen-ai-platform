import logging
import time
from typing import Dict, List, Optional, Any, AsyncIterator, Union
import traceback

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_exception

from app.core.config import settings
from app.core.logging import get_logger
from app.services.llm.base import LLMService
from app.schemas.chat import ChatMessage, ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseMessage
from app.schemas.completion import CompletionResponse, CompletionResponseChoice

logger = get_logger(__name__)

class AnthropicService(LLMService):
    """
    Service for interacting with Anthropic's Claude API.
    Implements streaming and non-streaming completions using Anthropic's API.
    """
    
    def __init__(self):
        """Initialize the Anthropic service with API credentials."""
        self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.api_base = settings.ANTHROPIC_API_BASE
        logger.info("Anthropic service initialized")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(lambda e: isinstance(e, (ConnectionError, TimeoutError)) or 
                                  (hasattr(e, 'status_code') and e.status_code >= 500)),
        reraise=True
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
        Generate a chat completion using Anthropic's Claude API.
        
        Args:
            messages: List of chat messages in the conversation
            model: Model identifier (e.g., "claude-3-7-sonnet-20250219")
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Nucleus sampling parameter
            n: Number of completions to generate
            stop: Sequence(s) at which to stop generation
            presence_penalty: Presence penalty (-2 to 2) - Not used by Anthropic
            frequency_penalty: Frequency penalty (-2 to 2) - Not used by Anthropic
            functions: List of function definitions (Claude calls these "tools")
            function_call: Controls function calling behavior (Claude calls this "tool_choice")
            user: End-user identifier
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            ChatCompletionResponse object with generated response(s)
        """
        try:
            # Use user parameter as user_id for tracking
            user_id = user or "anonymous_user"
            
            # Convert messages to dict format for internal processing
            converted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            
            # Extract system prompt (if any)
            system_prompt = self._get_system_prompt(converted_messages)
            
            # Format regular messages for Anthropic API
            formatted_messages = self._format_messages(converted_messages)
            
            # Log request details (without sensitive content)
            logger.debug(
                f"Anthropic request: model={model}, user={user_id}, "
                f"max_tokens={max_tokens or 1000}, temp={temperature}, "
                f"message_count={len(messages)}, has_system={system_prompt is not None}"
            )
            
            # Prepare parameters dictionary with only non-None values
            api_params = {
                "model": model,
                "messages": formatted_messages,
                "max_tokens": max_tokens or 1000,
                "temperature": temperature,
                "metadata": {"user_id": user_id}
            }
            
            # Add optional parameters only if they exist and have appropriate values
            if top_p is not None and top_p != 1.0:
                api_params["top_p"] = top_p
                
            if stop:
                api_params["stop_sequences"] = stop if isinstance(stop, list) else [stop]
                
            if system_prompt:
                api_params["system"] = system_prompt
            
            # Handle tools (Claude's version of functions) and tool_choice
            is_supported_model = any(name in model.lower() for name in ["claude-3", "claude-3-opus", "claude-3-sonnet", "claude-3-haiku"])
            
            if functions and len(functions) > 0 and is_supported_model:
                api_params["tools"] = functions
                
                # Only add tool_choice if we have tools and a specific function is requested
                if function_call and function_call != "auto" and isinstance(function_call, dict) and "name" in function_call:
                    api_params["tool_choice"] = {"type": "tool", "name": function_call["name"]}
            
            # Add remaining kwargs
            api_params.update(kwargs)
            
            # Make API call
            start_time = time.time()
            response = await self.client.messages.create(**api_params)
            duration = time.time() - start_time
            
            # Log successful response
            logger.info(
                f"Successful completion: model={model}, tokens={response.usage.output_tokens}, "
                f"duration={duration:.2f}s, user={user_id}"
            )
            
            # Create a timestamp from current time
            current_timestamp = int(time.time())
            
            # Check for tool use in the response
            function_call_data = None
            response_content = ""
            
            if hasattr(response, "content") and response.content:
                for block in response.content:
                    if block.type == "text":
                        response_content = block.text
                    elif block.type == "tool_use":
                        function_call_data = {
                            "name": block.id,
                            "arguments": block.input if hasattr(block, "input") else "{}"
                        }
            
            # Create a message object for the response
            message = ChatCompletionResponseMessage(
                role="assistant",
                content=response_content,
                function_call=function_call_data
            )
            
            # Create a choice object
            choice = ChatCompletionResponseChoice(
                index=0,
                message=message,
                finish_reason="stop" if not function_call_data else "function_call"
            )
            
            # Create standardized response
            result = ChatCompletionResponse(
                id=response.id,
                object="chat.completion",
                created=current_timestamp,
                model=model,
                choices=[choice],
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            )
            
            return result
            
        except Exception as e:
            # Log detailed error with traceback
            logger.error(
                f"Anthropic API error: {type(e).__name__}: {str(e)}\n"
                f"Model: {model}, User: {user}"
            )
            logger.debug(f"Error traceback: {traceback.format_exc()}")
            
            # Re-raise the exception so it can be handled by the caller
            raise
    
    @retry(
        stop=stop_after_attempt(2),  # Fewer retries for streaming to avoid long waits
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception(lambda e: isinstance(e, ConnectionError) or 
                                 (hasattr(e, 'status_code') and e.status_code >= 500)),
        reraise=True
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
        Stream a chat completion from Anthropic's Claude API.
        
        Args similar to generate_chat_completion, but returns a streaming response.
        
        Yields:
            Chunks of the generated response as they become available
        """
        try:
            user_id = user or "anonymous_user"
            converted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            system_prompt = self._get_system_prompt(converted_messages)
            formatted_messages = self._format_messages(converted_messages)
            
            logger.debug(
                f"Anthropic streaming request: model={model}, user={user_id}, "
                f"max_tokens={max_tokens or 1000}, temp={temperature}"
            )
            
            # Prepare API parameters with only valid values
            api_params = {
                "model": model,
                "messages": formatted_messages,
                "max_tokens": max_tokens or 1000,
                "temperature": temperature,
                "metadata": {"user_id": user_id},
                "stream": True  # Always true for streaming
            }
            
            # Add optional parameters
            if top_p is not None and top_p != 1.0:
                api_params["top_p"] = top_p
                
            if stop:
                api_params["stop_sequences"] = stop if isinstance(stop, list) else [stop]
                
            if system_prompt:
                api_params["system"] = system_prompt
            
            # Handle tools and tool_choice
            is_supported_model = any(name in model.lower() for name in ["claude-3", "claude-3-opus", "claude-3-sonnet", "claude-3-haiku"])
            
            if functions and len(functions) > 0 and is_supported_model:
                api_params["tools"] = functions
                
                if function_call and function_call != "auto" and isinstance(function_call, dict) and "name" in function_call:
                    api_params["tool_choice"] = {"type": "tool", "name": function_call["name"]}
            
            # Add remaining kwargs
            api_params.update(kwargs)
            
            # Make streaming API call
            stream = await self.client.messages.create(**api_params)
            
            # Track streaming metrics
            chunk_count = 0
            start_time = time.time()
            
            # Process and yield the streaming response
            async for chunk in stream:
                if chunk.type == "content_block_delta" and chunk.delta.text:
                    chunk_count += 1
                    yield chunk.delta.text
            
            duration = time.time() - start_time
            logger.info(
                f"Completed streaming: model={model}, chunks={chunk_count}, "
                f"duration={duration:.2f}s, user={user_id}"
            )
            
        except Exception as e:
            logger.error(
                f"Anthropic streaming error: {type(e).__name__}: {str(e)}\n"
                f"Model: {model}, User: {user}"
            )
            logger.debug(f"Streaming error traceback: {traceback.format_exc()}")
            raise
    
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
        Generate a text completion using Anthropic's Claude API.
        This is a wrapper around generate_chat_completion that accepts a single prompt.
        
        Args:
            prompt: The text prompt to complete
            model: The model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            n: Number of completions to generate
            stop: Sequence(s) at which to stop generation
            presence_penalty: Presence penalty (-2 to 2)
            frequency_penalty: Frequency penalty (-2 to 2)
            user: End-user identifier
            kwargs: Additional provider-specific parameters
            
        Returns:
            CompletionResponse object with generated completion(s)
        """
        try:
            # Create a chat message from the prompt
            chat_message = ChatMessage(role="user", content=prompt)
            
            # Call the chat completion method
            chat_response = await self.generate_chat_completion(
                messages=[chat_message],
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
            
            # Extract the content safely
            content = ""
            finish_reason = "stop"
            
            if chat_response.choices and len(chat_response.choices) > 0:
                choice = chat_response.choices[0]
                # Access content and finish_reason properly
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    content = choice.message.content or ""
                    finish_reason = choice.finish_reason or "stop"
                elif isinstance(choice, dict):
                    # If it's a dictionary
                    message = choice.get("message", {})
                    if isinstance(message, dict):
                        content = message.get("content", "")
                    else:
                        content = getattr(message, "content", "")
                    finish_reason = choice.get("finish_reason", "stop")
            
            # Extract usage data
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            
            if hasattr(chat_response, "usage"):
                usage = chat_response.usage
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                else:
                    prompt_tokens = getattr(usage, "prompt_tokens", 0)
                    completion_tokens = getattr(usage, "completion_tokens", 0)
                    total_tokens = getattr(usage, "total_tokens", 0)
            
            # Create response choices
            choices = [
                CompletionResponseChoice(
                    text=content,
                    index=0,
                    logprobs=None,
                    finish_reason=finish_reason
                )
            ]
            
            # Convert chat completion to completion format with proper usage dictionary
            return CompletionResponse(
                id=chat_response.id,
                object="text_completion",
                created=chat_response.created,
                model=chat_response.model,
                choices=choices,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            )
        except Exception as e:
            logger.error(f"Error in generate_completion: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise
    
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
        Stream a text completion from Anthropic's Claude API.
        This is a wrapper around stream_chat_completion that accepts a single prompt.
        
        Args:
            prompt: The text prompt to complete
            model: The model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            stop: Sequence(s) at which to stop generation
            presence_penalty: Presence penalty (-2 to 2)
            frequency_penalty: Frequency penalty (-2 to 2)
            user: End-user identifier
            kwargs: Additional provider-specific parameters
            
        Returns:
            Async iterator yielding completion text chunks
        """
        try:
            # Create a chat message from the prompt
            chat_message = ChatMessage(role="user", content=prompt)
            
            # Use the streaming chat completion method
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
        except Exception as e:
            logger.error(f"Error in stream_completion: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> List[MessageParam]:
        """
        Format messages for the Anthropic API.
        
        Args:
            messages: List of message objects with role and content
            
        Returns:
            Properly formatted messages for Anthropic API
        """
        formatted_messages = []
        
        for msg in messages:
            role = msg["role"].lower()
            # Skip system messages as they're handled separately
            if role == "system":
                continue
                
            # Map roles to their Anthropic equivalents
            api_role = {
                "user": "user",
                "assistant": "assistant",
                "function": "user"  # Map function messages to user messages with special formatting
            }.get(role, role)
            
            # For function messages, add a prefix to the content
            content = msg["content"]
            if role == "function" and "function_name" in msg:
                content = f"Function {msg['function_name']} returned: {content}"
            
            formatted_messages.append({
                "role": api_role,
                "content": content
            })
            
        return formatted_messages
    
    def _get_system_prompt(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Extract system prompt from messages if present.
        
        Args:
            messages: List of message objects
            
        Returns:
            System prompt string or None
        """
        for msg in messages:
            if msg["role"].lower() == "system":
                return msg["content"]
        return None