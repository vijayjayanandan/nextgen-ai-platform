with open('app/services/llm/on_prem_service.py', 'r') as f:
    content = f.read()

# Find the section with the logging but missing parse line
import re

# Replace the incomplete section
pattern = r'(result = response\.json\(\)\s*\n\s*# Parse response using the adapter\s*\n\s*logger\.info.*\n\s*logger\.debug.*\n\s*# Map response to our schema)'

replacement = '''result = response.json()
                
                # Parse response using the adapter
                logger.info(f"Using adapter: {self.adapter.__class__.__name__} for model_type: {self.model_type}")
                logger.debug(f"Raw result from API: {result}")
                parsed_result = await self.adapter.parse_completion_response(result)
                
                # Map response to our schema'''

content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

with open('app/services/llm/on_prem_service.py', 'w') as f:
    f.write(content)

print("Fixed missing parse_completion_response line")
