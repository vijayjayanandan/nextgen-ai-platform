import pytest
import pytest_asyncio
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock
from typing import Optional
import httpx
from fastapi.testclient import TestClient

# Import your actual services
from app.services.llm.anthropic_service import AnthropicService
from app.core.config import settings
from app.main import app


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def real_anthropic_service():
    """Fixture for real Anthropic service - requires ANTHROPIC_API_KEY"""
    if not settings.ANTHROPIC_API_KEY:
        pytest.skip("ANTHROPIC_API_KEY not configured - skipping real API tests")
    
    return AnthropicService(
        api_key=settings.ANTHROPIC_API_KEY,
        api_base=settings.ANTHROPIC_API_BASE
    )


@pytest.fixture
def mock_ollama_service():
    """Mock Ollama service for testing without real API calls"""
    mock_service = AsyncMock()
    
    # Configure mock responses based on model type
    async def mock_generate(prompt: str, model: str, **kwargs):
        if "claude" in model.lower():
            # Simulate Claude responses
            if "analyze the user query" in prompt.lower():
                return '{"query_type": "simple", "intent": "Canadian citizenship inquiry", "entities": ["citizenship", "requirements"]}'
            elif "rank documents" in prompt.lower():
                return '[0, 1, 2, 3, 4]'
            else:
                return "Based on the provided context documents, here is a comprehensive response about Canadian citizenship requirements..."
        else:
            # Simulate local model responses
            if "analyze the user query" in prompt.lower():
                return '{"query_type": "simple", "intent": "test query", "entities": []}'
            elif "rank documents" in prompt.lower():
                return '[0, 1, 2]'
            else:
                return "This is a mock response for testing."
    
    mock_service.generate = mock_generate
    return mock_service


@pytest.fixture
def use_real_api():
    """Fixture to determine if tests should use real API or mocks"""
    return os.getenv("USE_REAL_API", "false").lower() == "true"


@pytest.fixture
def llm_service(use_real_api, real_anthropic_service, mock_ollama_service):
    """Fixture that returns either real or mock LLM service based on configuration"""
    if use_real_api:
        return real_anthropic_service
    else:
        return mock_ollama_service


# Test configuration markers
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "real_api: mark test to run with real API calls"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle real API tests"""
    use_real_api = os.getenv("USE_REAL_API", "false").lower() == "true"
    
    for item in items:
        # Skip real API tests if not configured
        if "real_api" in item.keywords and not use_real_api:
            item.add_marker(pytest.mark.skip(reason="USE_REAL_API not enabled"))
        
        # Add slow marker to integration tests
        if "integration" in item.keywords:
            item.add_marker(pytest.mark.slow)


# Fixtures for common test data
@pytest.fixture
def sample_rag_state():
    """Sample RAG state for testing"""
    from app.services.rag.workflow_state import RAGState
    
    return RAGState(
        user_query="What are the requirements for Canadian citizenship?",
        conversation_id="test-123",
        user_id="test-user"
    )


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        {
            "id": "doc1",
            "title": "Citizenship Guide",
            "content": "Requirements for Canadian citizenship include...",
            "source_type": "policy",
            "score": 0.9
        },
        {
            "id": "doc2",
            "title": "Immigration FAQ",
            "content": "Common questions about immigration...",
            "source_type": "faq",
            "score": 0.8
        },
        {
            "id": "doc3",
            "title": "Application Process",
            "content": "How to apply for citizenship...",
            "source_type": "guide",
            "score": 0.7
        }
    ]


@pytest.fixture
def api_response_tracker():
    """Fixture to track API responses for analysis"""
    class ResponseTracker:
        def __init__(self):
            self.responses = []
            self.errors = []
            self.timeouts = []
        
        def track_response(self, model: str, prompt: str, response: str, duration: float):
            self.responses.append({
                "model": model,
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "response": response,
                "duration": duration,
                "response_length": len(response)
            })
        
        def track_error(self, model: str, error: Exception):
            self.errors.append({
                "model": model,
                "error": str(error),
                "error_type": type(error).__name__
            })
        
        def track_timeout(self, model: str, duration: float):
            self.timeouts.append({
                "model": model,
                "duration": duration
            })
        
        def get_summary(self):
            return {
                "total_responses": len(self.responses),
                "total_errors": len(self.errors),
                "total_timeouts": len(self.timeouts),
                "avg_response_time": sum(r["duration"] for r in self.responses) / len(self.responses) if self.responses else 0,
                "avg_response_length": sum(r["response_length"] for r in self.responses) / len(self.responses) if self.responses else 0
            }
    
    return ResponseTracker()


@pytest_asyncio.fixture
async def test_client():
    """Async HTTP client for testing FastAPI endpoints"""
    async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=30.0) as client:
        yield client


@pytest.fixture
def sync_test_client():
    """Synchronous test client for FastAPI endpoints"""
    return TestClient(app)
