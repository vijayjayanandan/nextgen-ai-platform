from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncIterator, Union

from app.schemas.chat import ChatMessage, ChatCompletionResponse
from app.schemas.completion import CompletionResponse


class LLMService(ABC):
    """
    Base abstract class for LLM services.
    All LLM providers should implement this interface.
    """
    
    @abstractmethod
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
        Generate a text completion.
        
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
        pass
    
    @abstractmethod
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
        Generate a chat completion.
        
        Args:
            messages: List of chat messages in the conversation
            model: The model to use
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
            kwargs: Additional provider-specific parameters
            
        Returns:
            ChatCompletionResponse object with generated response(s)
        """
        pass
    
    @abstractmethod
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
        Stream a text completion.
        
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
        pass
    
    @abstractmethod
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
        Stream a chat completion.
        
        Args:
            messages: List of chat messages in the conversation
            model: The model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            stop: Sequence(s) at which to stop generation
            presence_penalty: Presence penalty (-2 to 2)
            frequency_penalty: Frequency penalty (-2 to 2)
            functions: List of function definitions
            function_call: Controls function calling behavior
            user: End-user identifier
            kwargs: Additional provider-specific parameters
            
        Returns:
            Async iterator yielding chat response text chunks
        """
        pass
    
    @abstractmethod
    async def count_tokens(self, text: str, model: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            text: The text to tokenize
            model: The model to use for tokenization
            
        Returns:
            Number of tokens
        """
        pass