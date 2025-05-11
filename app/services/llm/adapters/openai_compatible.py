from app.services.llm.adapters.base import ModelAdapter
from typing import Dict, List, Optional, Any

class OpenAICompatibleAdapter(ModelAdapter):
    """Adapter for models that use an OpenAI-compatible API."""
    
    async def prepare_completion_payload(self, **kwargs) -> Dict[str, Any]:
        """Prepare payload for the completion endpoint."""
        # Standard OpenAI format
        return {
            "model": kwargs.get("model"),
            "prompt": kwargs.get("prompt"),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 1.0),
            "n": kwargs.get("n", 1),
            "stream": kwargs.get("stream", False),
            "max_tokens": kwargs.get("max_tokens"),
            "presence_penalty": kwargs.get("presence_penalty", 0),
            "frequency_penalty": kwargs.get("frequency_penalty", 0),
            "stop": kwargs.get("stop"),
            "user": kwargs.get("user")
        }
    
    async def prepare_chat_completion_payload(self, **kwargs) -> Dict[str, Any]:
        """Prepare payload for the chat completion endpoint."""
        return {
            "model": kwargs.get("model"),
            "messages": kwargs.get("messages"),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 1.0),
            "n": kwargs.get("n", 1),
            "stream": kwargs.get("stream", False),
            "max_tokens": kwargs.get("max_tokens"),
            "presence_penalty": kwargs.get("presence_penalty", 0),
            "frequency_penalty": kwargs.get("frequency_penalty", 0),
            "functions": kwargs.get("functions"),
            "function_call": kwargs.get("function_call"),
            "stop": kwargs.get("stop"),
            "user": kwargs.get("user")
        }
    
    async def parse_completion_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the response from the completion endpoint."""
        # OpenAI format is our standard, so just return as is
        return response
    
    async def parse_chat_completion_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the response from the chat completion endpoint."""
        # OpenAI format is our standard, so just return as is
        return response
    
    async def parse_streaming_chunk(self, chunk: Dict[str, Any]) -> str:
        """Parse a streaming chunk from OpenAI format."""
        if "choices" in chunk and len(chunk["choices"]) > 0:
            if "text" in chunk["choices"][0]:
                # Completion API
                return chunk["choices"][0].get("text", "")
            elif "delta" in chunk["choices"][0]:
                # Chat Completion API
                return chunk["choices"][0]["delta"].get("content", "")
        return ""