with open('app/services/llm/on_prem_service.py', 'r') as f:
    content = f.read()

# Add logging before calling the adapter's parse method
import re

# Find where parse_completion_response is called
pattern = r'(parsed_result = await self\.adapter\.parse_completion_response\(result\))'
replacement = '''logger.info(f"Using adapter: {self.adapter.__class__.__name__} for model_type: {self.model_type}")
                logger.debug(f"Raw result from API: {result}")
                \1'''

content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

with open('app/services/llm/on_prem_service.py', 'w') as f:
    f.write(content)

print("Added detailed logging before parse_completion_response")
