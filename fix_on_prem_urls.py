# Fix the URL construction in on_prem_service.py
import re

with open('app/services/llm/on_prem_service.py', 'r') as f:
    content = f.read()

# Fix the generate_completion method
old_line = 'url = f"{self.endpoint_url}/completions"'
new_line = '''# For Ollama, the endpoint already includes the full path
        if self.model_type == "ollama" or self.endpoint_url.endswith("/api/generate"):
            url = self.endpoint_url
        else:
            url = f"{self.endpoint_url}/completions"'''

content = content.replace(old_line, new_line)

# Fix the generate_chat_completion method
old_chat_line = 'url = f"{self.endpoint_url}/chat/completions"'
new_chat_line = '''# For Ollama, use the same endpoint for chat
        if self.model_type == "ollama" or self.endpoint_url.endswith("/api/generate"):
            url = self.endpoint_url
        else:
            url = f"{self.endpoint_url}/chat/completions"'''

content = content.replace(old_chat_line, new_chat_line)

# Update model type detection
old_init = '''            self.model_type = self.model_config.get("model_type", "default")'''
new_init = '''            self.model_type = self.model_config.get("model_type", "default")
            # Auto-detect Ollama models
            if self.model_config.get("base_model") == "ollama" or "ollama" in self.model_name:
                self.model_type = "ollama"'''

content = content.replace(old_init, new_init)

with open('app/services/llm/on_prem_service.py', 'w') as f:
    f.write(content)

print("Fixed URL construction in on_prem_service.py")
