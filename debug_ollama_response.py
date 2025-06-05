import asyncio
import httpx
import json

async def test_ollama():
    # Test what Ollama actually returns
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "deepseek-coder:1.3b",
                "prompt": "def fibonacci(n):",
                "stream": False,
                "options": {
                    "num_predict": 200
                }
            },
            timeout=30.0
        )
        
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

asyncio.run(test_ollama())
