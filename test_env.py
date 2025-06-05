from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Check critical environment variables
env_vars = [
    'POSTGRES_SERVER',
    'POSTGRES_USER', 
    'POSTGRES_PASSWORD',
    'POSTGRES_DB',
    'VECTOR_DB_TYPE',
    'VECTOR_DB_API_KEY',
    'SECRET_KEY'
]

print("Environment Variables Check:")
print("-" * 40)
for var in env_vars:
    value = os.getenv(var)
    if value:
        if 'KEY' in var or 'PASSWORD' in var:
            print(f"{var}: {'*' * 10} (hidden)")
        else:
            print(f"{var}: {value}")
    else:
        print(f"{var}: NOT SET ❌")
