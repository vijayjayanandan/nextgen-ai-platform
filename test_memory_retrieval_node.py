#!/usr/bin/env python3
"""
Test script for the enhanced MemoryRetrievalNode.

This demonstrates the node's functionality and expected input/output schemas.
"""

import asyncio
import sys
import os
from typing import Dict, Any

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.rag.nodes.memory_retrieval import (
    MemoryRetrievalNode, 
    MemoryConfig, 
    MemoryTurn,
    create_memory_retrieval_node
)
from app.services.rag.workflow_state import RAGState
from app.services.retrieval.qdrant_service import SearchResult


class MockQdrantService:
    """Mock Qdrant service for testing"""
    
    def __init__(self):
        # Mock conversation memory data
        self.mock_memory = [
            SearchResult(
                id="turn_1",
                score=0.9,
                content="User: What are the visa requirements for Canada?\nAssistant: For Canadian visas, you need to meet eligibility criteria including financial support, clean criminal record, and medical exams.",
                metadata={
                    "conversation_id": "conv_123",
                    "turn_number": 1,
                    "user_message": "What are the visa requirements for Canada?",
                    "assistant_message": "For Canadian visas, you need to meet eligibility criteria including financial support, clean criminal record, and medical exams.",
                    "timestamp": 1640995200.0
                }
            ),
            SearchResult(
                id="turn_2", 
                score=0.7,
                content="User: How long does processing take?\nAssistant: Processing times vary by visa type, typically 2-12 weeks for visitor visas and 6-12 months for permanent residence applications.",
                metadata={
                    "conversation_id": "conv_123",
                    "turn_number": 2,
                    "user_message": "How long does processing take?",
                    "assistant_message": "Processing times vary by visa type, typically 2-12 weeks for visitor visas and 6-12 months for permanent residence applications.",
                    "timestamp": 1640995800.0
                }
            ),
            SearchResult(
                id="turn_5",
                score=0.85,
                content="User: Can I work while my application is being processed?\nAssistant: Work authorization depends on your current status and visa type. Visitor visa holders generally cannot work, but some permit holders may be eligible.",
                metadata={
                    "conversation_id": "conv_123", 
                    "turn_number": 5,
                    "user_message": "Can I work while my application is being processed?",
                    "assistant_message": "Work authorization depends on your current status and visa type. Visitor visa holders generally cannot work, but some permit holders may be eligible.",
                    "timestamp": 1641000000.0
                }
            )
        ]
    
    async def search_memory(
        self,
        conversation_id: str,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.5
    ) -> list[SearchResult]:
        """Mock memory search that returns relevant turns"""
        
        # Filter by conversation_id and score_threshold
        results = [
            result for result in self.mock_memory
            if (result.metadata.get("conversation_id") == conversation_id and 
                result.score >= score_threshold)
        ]
        
        # Sort by relevance score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:limit]


async def test_memory_retrieval_node():
    """Test the MemoryRetrievalNode functionality"""
    
    print("🧪 Testing Enhanced MemoryRetrievalNode")
    print("=" * 50)
    
    # Create mock services
    mock_qdrant = MockQdrantService()
    
    # Create node with custom configuration
    config = MemoryConfig(
        max_turns=3,
        score_threshold=0.6,
        include_recent_turns=1,
        max_context_length=1500
    )
    
    memory_node = MemoryRetrievalNode(mock_qdrant, config)
    
    # Test Case 1: Normal conversation with memory
    print("\n📝 Test Case 1: Conversation with relevant memory")
    
    state = RAGState(
        user_query="What documents do I need for work authorization?",
        conversation_id="conv_123",
        user_id="user_456"
    )
    
    # Execute the node
    result_state = await memory_node.execute(state)
    
    # Print results
    print(f"✅ Memory turns retrieved: {len(result_state.relevant_history)}")
    print(f"✅ Memory context length: {len(result_state.memory_context)} characters")
    print(f"✅ Average relevance score: {result_state.memory_metadata.get('avg_relevance_score', 0):.3f}")
    
    print("\n📋 Formatted Memory Context:")
    print("-" * 30)
    print(result_state.memory_context)
    
    # Test Case 2: No conversation ID
    print("\n📝 Test Case 2: No conversation ID provided")
    
    state_no_conv = RAGState(
        user_query="What are the requirements?",
        conversation_id=None,
        user_id="user_456"
    )
    
    result_state_no_conv = await memory_node.execute(state_no_conv)
    
    print(f"✅ Memory enabled: {result_state_no_conv.memory_metadata.get('memory_enabled', False)}")
    print(f"✅ Turns retrieved: {result_state_no_conv.memory_metadata.get('turns_retrieved', 0)}")
    
    # Test Case 3: Factory function
    print("\n📝 Test Case 3: Using factory function")
    
    factory_node = create_memory_retrieval_node(
        qdrant_service=mock_qdrant,
        max_turns=2,
        score_threshold=0.8
    )
    
    state_factory = RAGState(
        user_query="Tell me about work permits",
        conversation_id="conv_123",
        user_id="user_789"
    )
    
    result_state_factory = await factory_node.execute(state_factory)
    
    print(f"✅ Factory node turns: {len(result_state_factory.relevant_history)}")
    print(f"✅ High threshold filtering: {result_state_factory.memory_metadata.get('avg_relevance_score', 0):.3f}")
    
    # Test Case 4: Memory summary
    print("\n📝 Test Case 4: Memory summary statistics")
    
    summary = memory_node.get_memory_summary(result_state)
    print(f"✅ Memory summary: {summary}")
    
    print("\n🎉 All tests completed successfully!")
    
    return result_state


def demonstrate_input_output_schema():
    """Demonstrate the expected input and output schemas"""
    
    print("\n📊 Input/Output Schema Documentation")
    print("=" * 50)
    
    print("\n📥 INPUT SCHEMA (RAGState):")
    print("""
    Required fields:
    - user_query: str                    # Current user query for semantic matching
    - conversation_id: Optional[str]     # Conversation identifier (None = skip memory)
    - user_id: str                       # User identifier
    
    Optional fields:
    - relevant_history: List[Dict]       # Will be populated by the node
    - memory_context: str                # Will be populated by the node
    - memory_metadata: Dict              # Will be populated by the node
    """)
    
    print("\n📤 OUTPUT SCHEMA (RAGState):")
    print("""
    Populated fields:
    - relevant_history: List[MemoryTurn] # Structured memory turns
    - memory_context: str                # Formatted context for LLM prompt
    - memory_metadata: Dict              # Statistics and metadata
        ├── turns_retrieved: int         # Number of turns found
        ├── avg_relevance_score: float   # Average relevance score
        ├── memory_enabled: bool         # Whether memory was used
        └── retrieval_error: Optional[str] # Any error that occurred
    """)
    
    print("\n🏗️ MemoryTurn Structure:")
    print("""
    @dataclass MemoryTurn:
    - id: str                           # Unique turn identifier
    - turn_number: int                  # Sequential turn number
    - user_message: str                 # User's message
    - assistant_message: str            # Assistant's response
    - timestamp: float                  # Unix timestamp
    - relevance_score: float            # Semantic relevance score (0-1)
    - context_snippet: str              # Combined context text
    
    Methods:
    - to_context_string() -> str        # Formatted string for LLM context
    """)
    
    print("\n⚙️ Configuration Options:")
    print("""
    MemoryConfig:
    - max_turns: int = 5                # Maximum memory turns to retrieve
    - score_threshold: float = 0.5      # Minimum relevance score
    - include_recent_turns: int = 2     # Always include N recent turns
    - merge_consecutive_turns: bool = True  # Merge adjacent turns
    - max_context_length: int = 2000    # Max characters for context
    """)


if __name__ == "__main__":
    # Run the demonstration
    demonstrate_input_output_schema()
    
    # Run the tests
    asyncio.run(test_memory_retrieval_node())
