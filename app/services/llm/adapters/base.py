from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncIterator, Union
import httpx

class ModelAdapter(ABC):
    """Abstract base class for model adapters."""
    
    @abstractmethod
    async def prepare_completion_payload(self, **kwargs) -> Dict[str, Any]:
        """Prepare payload for the completion endpoint."""
        pass
    
    @abstractmethod
    async def prepare_chat_completion_payload(self, **kwargs) -> Dict[str, Any]:
        """Prepare payload for the chat completion endpoint."""
        pass
    
    @abstractmethod
    async def parse_completion_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the response from the completion endpoint."""
        pass
    
    @abstractmethod
    async def parse_chat_completion_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the response from the chat completion endpoint."""
        pass
    
    @abstractmethod
    async def parse_streaming_chunk(self, chunk: Dict[str, Any]) -> str:
        """Parse a streaming chunk."""
        pass