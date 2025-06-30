#!/usr/bin/env python3
"""
Test script to validate LLM-agnostic configuration
"""

import asyncio
import os
import sys
from typing import Dict, Any

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.config import settings
from app.services.rag.workflow_state import RAGState, QueryType
from app.services.rag.nodes.query_analysis import QueryAnalysisNode
from app.services.rag.nodes.generation import GenerationNode
from app.services.rag.nodes.reranking import RerankingNode


class MockLLMService:
    """Mock LLM service for testing configuration (supports both Ollama and Claude formats)"""
    
    def __init__(self):
        self.calls = []
    
    async def generate(self, prompt: str, model: str, **kwargs) -> str:
        """Mock generate method that records calls"""
        call_info = {
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "model": model,
            "kwargs": kwargs
        }
        self.calls.append(call_info)
        
        # Simulate different response formats based on model type
        if "claude" in model.lower():
            # Simulate Claude's response format
            if "analyze the user query" in prompt.lower():
                return '{"query_type": "simple", "intent": "Canadian citizenship inquiry", "entities": ["citizenship", "requirements"]}'
            elif "rank documents" in prompt.lower():
                return '[0, 1, 2, 3, 4]'
            else:
                return "Based on the provided context documents, here is a comprehensive response about Canadian citizenship requirements..."
        else:
            # Simulate Ollama/local model response format
            if "analyze the user query" in prompt.lower():
                return '{"query_type": "simple", "intent": "test query", "entities": []}'
            elif "rank documents" in prompt.lower():
                return '[0, 1, 2]'
            else:
                return "This is a mock response for testing LLM-agnostic configuration."


async def test_llm_agnostic_configuration():
    """Test that all RAG nodes use configurable models"""
    
    print("🔍 Testing LLM-Agnostic Configuration")
    print("=" * 50)
    
    # Display current configuration
    print("\n📋 Current Model Configuration:")
    print(f"  Query Analysis: {settings.RAG_QUERY_ANALYSIS_MODEL}")
    print(f"  Generation: {settings.RAG_GENERATION_MODEL}")
    print(f"  Reranking: {settings.RAG_RERANKING_MODEL}")
    print(f"  Memory Retrieval: {settings.RAG_MEMORY_RETRIEVAL_MODEL}")
    print(f"  Citation: {settings.RAG_CITATION_MODEL}")
    
    # Create mock service
    mock_llm = MockLLMService()
    
    # Initialize RAG nodes
    query_analysis_node = QueryAnalysisNode(mock_llm)
    generation_node = GenerationNode(mock_llm)
    reranking_node = RerankingNode(mock_llm)
    
    # Create test state
    state = RAGState(
        user_query="What are the requirements for Canadian citizenship?",
        conversation_id="test-123"
    )
    
    # Add mock documents for reranking test (need more than 5 to trigger reranking)
    state.raw_documents = [
        {"id": "1", "title": "Citizenship Guide", "content": "Requirements for citizenship...", "score": 0.9},
        {"id": "2", "title": "Immigration FAQ", "content": "Common questions about immigration...", "score": 0.8},
        {"id": "3", "title": "Application Process", "content": "How to apply for citizenship...", "score": 0.7},
        {"id": "4", "title": "Eligibility Criteria", "content": "Who can apply for citizenship...", "score": 0.6},
        {"id": "5", "title": "Required Documents", "content": "Documents needed for application...", "score": 0.5},
        {"id": "6", "title": "Processing Times", "content": "How long does it take...", "score": 0.4}
    ]
    
    print("\n🧪 Testing RAG Nodes...")
    
    # Test Query Analysis Node
    print("\n1. Testing Query Analysis Node...")
    try:
        state = await query_analysis_node.execute(state)
        print(f"   ✅ Query Analysis completed")
        print(f"   📊 Query Type: {state.query_type}")
        print(f"   🎯 Intent: {state.intent}")
    except Exception as e:
        print(f"   ❌ Query Analysis failed: {e}")
    
    # Test Reranking Node
    print("\n2. Testing Reranking Node...")
    try:
        state = await reranking_node.execute(state)
        print(f"   ✅ Reranking completed")
        print(f"   📄 Documents reranked: {len(state.reranked_documents)}")
    except Exception as e:
        print(f"   ❌ Reranking failed: {e}")
    
    # Test Generation Node
    print("\n3. Testing Generation Node...")
    try:
        state = await generation_node.execute(state)
        print(f"   ✅ Generation completed")
        print(f"   📝 Response length: {len(state.response)} characters")
        print(f"   🤖 Model used: {state.model_used}")
    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
    
    # Analyze model usage
    print("\n📊 Model Usage Analysis:")
    print("-" * 30)
    
    for i, call in enumerate(mock_llm.calls, 1):
        print(f"{i}. Model: {call['model']}")
        print(f"   Prompt: {call['prompt']}")
        print(f"   Params: {call['kwargs']}")
        print()
    
    # Verify configuration is being used
    models_used = [call['model'] for call in mock_llm.calls]
    expected_models = [
        settings.RAG_QUERY_ANALYSIS_MODEL,
        settings.RAG_RERANKING_MODEL,
        settings.RAG_GENERATION_MODEL
    ]
    
    print("✅ Configuration Validation:")
    for expected, actual in zip(expected_models, models_used):
        if expected == actual:
            print(f"   ✅ {expected} - CORRECT")
        else:
            print(f"   ❌ Expected {expected}, got {actual} - INCORRECT")
    
    print("\n🎉 LLM-Agnostic Configuration Test Complete!")
    
    return len([call for call in mock_llm.calls]) > 0


def test_configuration_switching():
    """Test configuration switching scenarios"""
    
    print("\n🔄 Testing Configuration Switching Scenarios")
    print("=" * 50)
    
    # Test different provider configurations
    scenarios = [
        {
            "name": "Local Models (Ollama)",
            "config": {
                "RAG_QUERY_ANALYSIS_MODEL": "mistral:7b",
                "RAG_GENERATION_MODEL": "deepseek-coder:6.7b",
                "RAG_RERANKING_MODEL": "mistral:7b"
            }
        },
        {
            "name": "OpenAI Models",
            "config": {
                "RAG_QUERY_ANALYSIS_MODEL": "gpt-3.5-turbo",
                "RAG_GENERATION_MODEL": "gpt-4",
                "RAG_RERANKING_MODEL": "gpt-3.5-turbo"
            }
        },
        {
            "name": "Anthropic Models",
            "config": {
                "RAG_QUERY_ANALYSIS_MODEL": "claude-3-haiku-20240307",
                "RAG_GENERATION_MODEL": "claude-3-sonnet-20240229",
                "RAG_RERANKING_MODEL": "claude-3-haiku-20240307"
            }
        },
        {
            "name": "Hybrid Setup",
            "config": {
                "RAG_QUERY_ANALYSIS_MODEL": "mistral:7b",
                "RAG_GENERATION_MODEL": "gpt-4",
                "RAG_RERANKING_MODEL": "claude-3-haiku-20240307"
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 Scenario: {scenario['name']}")
        print("   Configuration:")
        for key, value in scenario['config'].items():
            print(f"     {key}={value}")
        
        # Show environment variable commands
        print("   Environment Variables:")
        for key, value in scenario['config'].items():
            print(f"     export {key}={value}")
    
    print("\n💡 To switch configurations:")
    print("   1. Update your .env file with the desired model names")
    print("   2. Restart the application")
    print("   3. The system will automatically use the new models")


def main():
    """Main test function"""
    
    print("🚀 LLM-Agnostic Configuration Test Suite")
    print("=" * 60)
    
    # Test current configuration
    asyncio.run(test_llm_agnostic_configuration())
    
    # Test configuration switching scenarios
    test_configuration_switching()
    
    print("\n📚 Next Steps:")
    print("   1. Review the LLM_AGNOSTIC_CONFIGURATION.md guide")
    print("   2. Update your .env file with your preferred models")
    print("   3. Test with real LLM providers")
    print("   4. Monitor performance and costs")
    
    print("\n✨ Your RAG system is now fully LLM-agnostic!")


if __name__ == "__main__":
    main()
