with open('app/services/llm/adapters/ollama.py', 'r') as f:
    content = f.read()

# Fix the parse_completion_response method
old_parse_method = '''    async def parse_completion_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Ollama completion response to standard format."""
        # Ollama returns a simple format, we need to convert it
        text = response.get("response", "")
        
        # Calculate token counts (approximate)
        prompt_tokens = len(response.get("prompt", "").split()) * 4 // 3
        completion_tokens = len(text.split()) * 4 // 3
        
        return {
            "id": response.get("created_at", ""),
            "object": "text_completion",
            "created": int(response.get("created_at", 0)),
            "model": response.get("model", ""),
            "choices": [{
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop" if response.get("done", False) else "length"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }'''

new_parse_method = '''    async def parse_completion_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
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
        }'''

content = content.replace(old_parse_method, new_parse_method)

# Also fix the parse_chat_completion_response method
old_chat_parse = '''    async def parse_chat_completion_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Ollama chat completion response to standard format."""
        text = response.get("response", "")
        
        # Calculate token counts (approximate)
        prompt_tokens = len(response.get("prompt", "").split()) * 4 // 3
        completion_tokens = len(text.split()) * 4 // 3
        
        return {
            "id": response.get("created_at", ""),
            "object": "chat.completion",
            "created": int(response.get("created_at", 0)),
            "model": response.get("model", ""),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text
                },
                "finish_reason": "stop" if response.get("done", False) else "length"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }'''

new_chat_parse = '''    async def parse_chat_completion_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
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
        }'''

content = content.replace(old_chat_parse, new_chat_parse)

with open('app/services/llm/adapters/ollama.py', 'w') as f:
    f.write(content)

print("Fixed Ollama adapter parsing methods")
