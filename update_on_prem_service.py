import fileinput
import sys

# Read the file and add Ollama import
with open('app/services/llm/on_prem_service.py', 'r') as f:
    content = f.read()

# Add the import after other adapter imports
import_line = "from app.services.llm.adapters.deepseek import DeepseekAdapter"
new_import = "from app.services.llm.adapters.ollama import OllamaAdapter"

if new_import not in content:
    content = content.replace(
        import_line,
        f"{import_line}\n{new_import}"
    )

# Update the adapters dictionary in _get_adapter method
old_adapters = '''        adapters = {
            "llama": LlamaAdapter(),
            "deepseek": DeepseekAdapter(),
            "default": OpenAICompatibleAdapter()
        }'''

new_adapters = '''        adapters = {
            "llama": LlamaAdapter(),
            "deepseek": DeepseekAdapter(),
            "ollama": OllamaAdapter(),
            "default": OpenAICompatibleAdapter()
        }'''

content = content.replace(old_adapters, new_adapters)

# Write back
with open('app/services/llm/on_prem_service.py', 'w') as f:
    f.write(content)

print("Updated on_prem_service.py to include Ollama adapter")
