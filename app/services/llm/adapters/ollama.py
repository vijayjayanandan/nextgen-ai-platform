"""Ollama adapter for on-premises LLM service."""
import json
from typing import Dict, List, Optional, Any, Union

from app.services.llm.adapters.base import ModelAdapter
from app.core.logging import get_logger

logger = get_logger(__name__)


class OllamaAdapter(ModelAdapter):
    """Adapter for Ollama API format."""
    
    async def prepare_completion_payload(
        self,
        prompt: str,
        model: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        n: int = 1,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare completion payload for Ollama format."""
        # Ollama uses 'num_predict' instead of 'max_tokens'
        options = {
            "temperature": temperature,
            "top_p": top_p,
        }
        
        if max_tokens:
            options["num_predict"] = max_tokens
            
        if stop:
            options["stop"] = stop if isinstance(stop, list) else [stop]
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": options
        }
        
        logger.debug(f"Prepared Ollama completion payload: {payload}")
        return payload
    
    async def parse_completion_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Ollama completion response to standard format."""
        # Ollama returns a simple format, we need to convert it
        text = response.get("response", "")
        
        # Use actual token counts from Ollama if available
        prompt_tokens = response.get("prompt_eval_count", 0)
        completion_tokens = response.get("eval_count", 0)
        
        # Convert created_at timestamp to Unix timestamp
        import time
        from datetime import datetime
        created_at = response.get("created_at", "")
        if created_at:
            try:
                # Parse ISO format timestamp
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                created_timestamp = int(dt.timestamp())
            except:
                created_timestamp = int(time.time())
        else:
            created_timestamp = int(time.time())
        
        # Map done_reason to finish_reason
        done_reason = response.get("done_reason", "stop")
        finish_reason = "length" if done_reason == "length" else "stop"
        
        return {
            "id": f"ollama-{response.get('model', 'unknown')}-{created_timestamp}",
            "object": "text_completion",
            "created": created_timestamp,
            "model": response.get("model", ""),
            "choices": [{
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
    
    async def prepare_chat_completion_payload(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        n: int = 1,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare chat completion payload for Ollama format."""
        # Convert messages to a single prompt
        prompt = self._messages_to_prompt(messages)
        
        options = {
            "temperature": temperature,
            "top_p": top_p,
        }
        
        if max_tokens:
            options["num_predict"] = max_tokens
            
        if stop:
            options["stop"] = stop if isinstance(stop, list) else [stop]
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": options
        }
        
        logger.debug(f"Prepared Ollama chat payload: {payload}")
        return payload
    
    async def parse_chat_completion_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Ollama chat completion response to standard format."""
        text = response.get("response", "")
        
        # Use actual token counts from Ollama if available
        prompt_tokens = response.get("prompt_eval_count", 0)
        completion_tokens = response.get("eval_count", 0)
        
        # Convert created_at timestamp to Unix timestamp
        import time
        from datetime import datetime
        created_at = response.get("created_at", "")
        if created_at:
            try:
                # Parse ISO format timestamp
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                created_timestamp = int(dt.timestamp())
            except:
                created_timestamp = int(time.time())
        else:
            created_timestamp = int(time.time())
        
        # Map done_reason to finish_reason
        done_reason = response.get("done_reason", "stop")
        finish_reason = "length" if done_reason == "length" else "stop"
        
        return {
            "id": f"ollama-{response.get('model', 'unknown')}-{created_timestamp}",
            "object": "chat.completion",
            "created": created_timestamp,
            "model": response.get("model", ""),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text
                },
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
    
    async def parse_streaming_chunk(self, chunk: Dict[str, Any]) -> Optional[str]:
        """Parse a streaming chunk from Ollama."""
        return chunk.get("response", "")
    
    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Convert chat messages to a single prompt for Ollama."""
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Add a final "Assistant:" to prompt the model to respond
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
