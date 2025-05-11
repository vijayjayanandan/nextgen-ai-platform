from app.services.llm.adapters.openai_compatible import OpenAICompatibleAdapter
from typing import Dict, List, Optional, Any

class LlamaAdapter(OpenAICompatibleAdapter):
    """Adapter for Llama models."""
    
    async def prepare_chat_completion_payload(self, **kwargs) -> Dict[str, Any]:
        """Prepare payload for the chat completion endpoint."""
        payload = await super().prepare_chat_completion_payload(**kwargs)
        
        # Add Llama-specific parameters
        if "frequency_penalty" in payload:
            # Llama 3 calls it "repetition_penalty" instead
            payload["repetition_penalty"] = payload.pop("frequency_penalty") + 1.0
            
        # Transform function calling format if needed
        if kwargs.get("functions") and "llama-3" in kwargs.get("model", ""):
            # Llama 3 uses a slightly different format for tool/function calling
            tools = []
            for func in kwargs.get("functions", []):
                tools.append({
                    "type": "function",
                    "function": func
                })
            payload["tools"] = tools
            payload.pop("functions", None)
            
            if "function_call" in payload:
                if isinstance(payload["function_call"], dict) and "name" in payload["function_call"]:
                    payload["tool_choice"] = {
                        "type": "function",
                        "function": {"name": payload["function_call"]["name"]}
                    }
                elif payload["function_call"] == "auto":
                    payload["tool_choice"] = "auto"
                payload.pop("function_call", None)
            
        return payload
    
    async def parse_chat_completion_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the response from Llama API."""
        # Convert Llama-specific formats to OpenAI format if needed
        if "tool_calls" in response:
            # Map tool_calls to function_call format
            for choice in response.get("choices", []):
                if "message" in choice and "tool_calls" in choice["message"]:
                    tool_call = choice["message"]["tool_calls"][0]
                    if tool_call["type"] == "function":
                        choice["message"]["function_call"] = {
                            "name": tool_call["function"]["name"],
                            "arguments": tool_call["function"]["arguments"]
                        }
                        choice["message"].pop("tool_calls")
                    
        return response
    
    async def parse_streaming_chunk(self, chunk: Dict[str, Any]) -> str:
        """Parse a streaming chunk from Llama."""
        # Handle tool calls in streaming mode if present
        if "choices" in chunk and len(chunk["choices"]) > 0:
            delta = chunk["choices"][0].get("delta", {})
            
            if "tool_calls" in delta:
                # Skip tool call chunks in streaming mode
                return ""
                
        # Otherwise use standard parsing
        return await super().parse_streaming_chunk(chunk)
    
    async def count_tokens(self, text: str) -> int:
        """Estimate token count for Llama models."""
        # Llama models typically use ~4.5 tokens per word
        return len(text.split()) * 9 // 2