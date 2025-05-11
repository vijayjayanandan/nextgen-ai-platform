from app.services.llm.adapters.openai_compatible import OpenAICompatibleAdapter
from typing import Dict, List, Optional, Any

class DeepseekAdapter(OpenAICompatibleAdapter):
    """Adapter for Deepseek models."""
    
    async def prepare_chat_completion_payload(self, **kwargs) -> Dict[str, Any]:
        """Prepare payload for the chat completion endpoint."""
        payload = await super().prepare_chat_completion_payload(**kwargs)
        
        # Deepseek-specific parameters
        # For Deepseek-Coder, we might need to modify the system prompt
        if "deepseek-coder" in kwargs.get("model", "") and "messages" in payload:
            messages = payload["messages"]
            
            # Look for system message and enhance it for code generation
            for i, msg in enumerate(messages):
                if msg.get("role") == "system":
                    # Add code-specific instructions to system prompt
                    if not msg["content"].startswith("You are DeepSeek Coder"):
                        messages[i]["content"] = (
                            "You are DeepSeek Coder, a helpful coding assistant. "
                            "Provide clear, efficient code solutions to problems. "
                            f"{msg['content']}"
                        )
                    break
                    
            # If no system message, add one
            if not any(msg.get("role") == "system" for msg in messages):
                messages.insert(0, {
                    "role": "system",
                    "content": "You are DeepSeek Coder, a helpful coding assistant. Provide clear, efficient code solutions to problems."
                })
                
        # Handle Deepseek's different repetition penalty (similar to Llama)
        if "frequency_penalty" in payload:
            payload["repetition_penalty"] = payload.pop("frequency_penalty") + 1.0
            
        # Function calling support if available
        if "deepseek-coder-instruct" in kwargs.get("model", ""):
            # Convert to correct format for Deepseek's newer instruction models
            if kwargs.get("functions"):
                tools = []
                for func in kwargs.get("functions", []):
                    tools.append({
                        "type": "function",
                        "function": func
                    })
                payload["tools"] = tools
                payload.pop("functions", None)
                
                # Convert function_call format
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
        """Parse the response from Deepseek API."""
        # Convert Deepseek-specific formats to OpenAI format if needed
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
    
    async def count_tokens(self, text: str) -> int:
        """Estimate token count for Deepseek models."""
        # Deepseek models average tokenization ratio
        return len(text.split()) * 5 // 3