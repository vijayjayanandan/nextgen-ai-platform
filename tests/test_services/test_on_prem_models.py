import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import Response

from app.services.llm.on_prem_service import OnPremLLMService
from app.services.llm.adapters.llama import LlamaAdapter
from app.services.llm.adapters.deepseek import DeepseekAdapter
from app.schemas.chat import ChatMessage
from app.models.chat import MessageRole


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for testing."""
    with patch("httpx.AsyncClient") as mock_client:
        # Configure the mock client
        mock_instance = MagicMock()
        mock_client.return_value.__aenter__.return_value = mock_instance
        
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1620000000,
            "model": "llama-7b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a test response from the mock Llama model."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 15,
                "total_tokens": 35
            }
        }
        mock_instance.post.return_value = mock_response
        
        yield mock_instance


@pytest.mark.asyncio
async def test_llama_adapter_params():
    """Test that Llama adapter properly formats parameters."""
    adapter = LlamaAdapter()
    
    # Test chat completion payload formatting
    payload = await adapter.prepare_chat_completion_payload(
        model="llama-3-8b",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
        frequency_penalty=0.5,
        functions=[{"name": "get_weather", "parameters": {"type": "object"}}],
        function_call={"name": "get_weather"}
    )
    
    # Check Llama-specific transformations
    assert "repetition_penalty" in payload
    assert payload["repetition_penalty"] == 1.5  # frequency_penalty + 1.0
    assert "tools" in payload
    assert payload["tools"][0]["type"] == "function"
    assert "tool_choice" in payload
    assert payload["tool_choice"]["function"]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_deepseek_adapter_params():
    """Test that Deepseek adapter properly formats parameters."""
    adapter = DeepseekAdapter()
    
    # Test chat completion payload for Deepseek Coder
    payload = await adapter.prepare_chat_completion_payload(
        model="deepseek-coder-instruct",
        messages=[{"role": "user", "content": "Write a Python function"}],
        temperature=0.7
    )
    
    # Check for system message enhancement for code
    found_system = False
    for msg in payload["messages"]:
        if msg.get("role") == "system" and "DeepSeek Coder" in msg.get("content", ""):
            found_system = True
            break
            
    assert found_system, "Should add or enhance system message for Deepseek Coder"


@pytest.mark.asyncio
async def test_on_prem_llm_service_with_llama(mock_httpx_client):
    """Test the OnPremLLMService with Llama model."""
    # Create OnPremLLMService with Llama model
    service = OnPremLLMService(model_name="llama-7b")
    
    # Verify adapter type
    assert isinstance(service.adapter, LlamaAdapter)
    
    # Test chat completion
    messages = [
        ChatMessage(role=MessageRole.USER, content="Hello, how are you?")
    ]
    
    result = await service.generate_chat_completion(
        messages=messages,
        model="llama-7b",
        temperature=0.7
    )
    
    # Verify the API was called correctly
    call_args = mock_httpx_client.post.call_args[1]
    assert "json" in call_args
    
    # Check that Llama-specific parameters were included
    assert "repetition_penalty" in call_args["json"]
    
    # Verify response processing
    assert result.choices[0].message.content == "This is a test response from the mock Llama model."


@pytest.mark.asyncio
async def test_on_prem_llm_service_with_deepseek(mock_httpx_client):
    """Test the OnPremLLMService with Deepseek model."""
    # Create OnPremLLMService with Deepseek model
    service = OnPremLLMService(model_name="deepseek-coder")
    
    # Verify adapter type
    assert isinstance(service.adapter, DeepseekAdapter)
    
    # Test chat completion for code generation
    messages = [
        ChatMessage(role=MessageRole.USER, content="Write a Python function to calculate Fibonacci")
    ]
    
    result = await service.generate_chat_completion(
        messages=messages,
        model="deepseek-coder",
        temperature=0.7
    )
    
    # Verify the API was called correctly
    call_args = mock_httpx_client.post.call_args[1]
    assert "json" in call_args
    
    # Check that messages include a system prompt for code
    messages = call_args["json"]["messages"]
    has_code_system = any(
        msg.get("role") == "system" and "Coder" in msg.get("content", "")
        for msg in messages
    )
    assert has_code_system, "Should include Deepseek Coder system prompt"
    
    # Verify response processing
    assert result.choices[0].message.content == "This is a test response from the mock Llama model."