# tests/test_memory_config_edge_cases.py
"""
Tests for MemoryConfig edge cases and boundary conditions.

Tests various memory configuration scenarios:
- Zero limits (max_turns=0, score_threshold=1.0)
- Overflow handling (include_recent_turns > available)
- Context truncation (max_context_length forcing ellipsis)
- Invalid configurations
"""

import pytest
from typing import Dict, Any, Optional
from unittest.mock import AsyncMock

from app.services.rag.nodes.memory_retrieval import MemoryRetrievalNode, MemoryConfig
from app.services.rag.workflow_state import RAGState
from tests.fixtures import (
    ProductionMockQdrantService,
    MockScenarioConfig,
    create_test_state,
    assert_memory_metadata_valid,
    MEMORY_CONFIG_SCENARIOS
)


class TestMemoryConfigEdgeCases:
    """Test MemoryConfig boundary conditions and edge cases"""
    
    @pytest.fixture
    def mock_qdrant_service(self):
        """Create mock Qdrant service with full memory data"""
        config = MockScenarioConfig(
            memory_enabled=True,
            documents_available=False,  # Focus on memory only
            response_latency=0.01,
            error_rate=0.0,
            max_memory_turns=10  # Plenty of memory available
        )
        return ProductionMockQdrantService(config)
    
    @pytest.mark.asyncio
    async def test_memory_config_max_turns_zero(self, mock_qdrant_service):
        """Test MemoryConfig with max_turns=0 (no memory retrieval)"""
        
        # Create config with zero max turns
        config = MemoryConfig(
            max_turns=0,
            score_threshold=0.5,
            include_recent_turns=2,
            max_context_length=2000
        )
        
        memory_node = MemoryRetrievalNode(
            qdrant_service=mock_qdrant_service,
            config=config
        )
        
        # Create state with conversation ID
        state = create_test_state(
            user_query="What documents do I need?",
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # Process the state
        result_state = await memory_node.process(state)
        
        # Should return no memory context due to max_turns=0
        assert result_state.memory_context == ""
        assert result_state.memory_metadata["turns_retrieved"] == 0
        assert result_state.memory_metadata["memory_enabled"] == True
        assert result_state.memory_metadata["avg_relevance_score"] == 0.0
    
    @pytest.mark.asyncio
    async def test_memory_config_score_threshold_maximum(self, mock_qdrant_service):
        """Test MemoryConfig with score_threshold=1.0 (nothing retrieved)"""
        
        # Create config with maximum score threshold
        config = MemoryConfig(
            max_turns=5,
            score_threshold=1.0,  # Perfect match required
            include_recent_turns=2,
            max_context_length=2000
        )
        
        memory_node = MemoryRetrievalNode(
            qdrant_service=mock_qdrant_service,
            config=config
        )
        
        # Create state with conversation ID
        state = create_test_state(
            user_query="What documents do I need?",
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # Process the state
        result_state = await memory_node.process(state)
        
        # Should return no memory context due to high threshold
        assert result_state.memory_context == ""
        assert result_state.memory_metadata["turns_retrieved"] == 0
        assert result_state.memory_metadata["memory_enabled"] == True
        assert result_state.memory_metadata["avg_relevance_score"] == 0.0
    
    @pytest.mark.asyncio
    async def test_memory_config_include_recent_turns_overflow(self, mock_qdrant_service):
        """Test MemoryConfig with include_recent_turns > available turns"""
        
        # Create config requesting more recent turns than available
        config = MemoryConfig(
            max_turns=5,
            score_threshold=0.5,
            include_recent_turns=10,  # More than available
            max_context_length=2000
        )
        
        memory_node = MemoryRetrievalNode(
            qdrant_service=mock_qdrant_service,
            config=config
        )
        
        # Create state with conversation ID
        state = create_test_state(
            user_query="What documents do I need?",
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # Process the state
        result_state = await memory_node.process(state)
        
        # Should handle gracefully and return available turns
        assert result_state.memory_metadata["memory_enabled"] == True
        assert result_state.memory_metadata["turns_retrieved"] <= 5  # Capped by max_turns
        
        # Should still have valid memory context if any turns were retrieved
        if result_state.memory_metadata["turns_retrieved"] > 0:
            assert len(result_state.memory_context) > 0
            assert "## Conversation History:" in result_state.memory_context
    
    @pytest.mark.asyncio
    async def test_memory_config_max_context_length_truncation(self, mock_qdrant_service):
        """Test MemoryConfig with very small max_context_length forcing truncation"""
        
        # Create config with very small context length
        config = MemoryConfig(
            max_turns=5,
            score_threshold=0.3,  # Low threshold to get more content
            include_recent_turns=2,
            max_context_length=100  # Very small limit
        )
        
        memory_node = MemoryRetrievalNode(
            qdrant_service=mock_qdrant_service,
            config=config
        )
        
        # Create state with conversation ID
        state = create_test_state(
            user_query="What documents do I need?",
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # Process the state
        result_state = await memory_node.process(state)
        
        # Should truncate context to fit within limit
        if result_state.memory_metadata["turns_retrieved"] > 0:
            assert len(result_state.memory_context) <= config.max_context_length
            
            # Should include ellipsis if truncated
            if len(result_state.memory_context) == config.max_context_length:
                assert result_state.memory_context.endswith("...")
    
    @pytest.mark.asyncio
    async def test_memory_config_include_recent_turns_zero(self, mock_qdrant_service):
        """Test MemoryConfig with include_recent_turns=0 (relevance-only)"""
        
        # Create config with no guaranteed recent turns
        config = MemoryConfig(
            max_turns=3,
            score_threshold=0.7,  # Higher threshold
            include_recent_turns=0,  # No guaranteed recent turns
            max_context_length=2000
        )
        
        memory_node = MemoryRetrievalNode(
            qdrant_service=mock_qdrant_service,
            config=config
        )
        
        # Create state with conversation ID
        state = create_test_state(
            user_query="What documents do I need?",
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # Process the state
        result_state = await memory_node.process(state)
        
        # Should only include turns that meet relevance threshold
        assert result_state.memory_metadata["memory_enabled"] == True
        
        if result_state.memory_metadata["turns_retrieved"] > 0:
            # All retrieved turns should meet the threshold
            assert result_state.memory_metadata["avg_relevance_score"] >= config.score_threshold
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("config_name,expected_behavior", [
        ("default", "normal_retrieval"),
        ("restrictive", "limited_retrieval"),
        ("permissive", "expanded_retrieval"),
        ("edge_cases", "minimal_or_no_retrieval"),
    ])
    async def test_memory_config_scenarios(
        self, 
        mock_qdrant_service, 
        config_name, 
        expected_behavior
    ):
        """Test predefined memory configuration scenarios"""
        
        # Get config from test data
        config_data = MEMORY_CONFIG_SCENARIOS[config_name]
        config = MemoryConfig(**config_data)
        
        memory_node = MemoryRetrievalNode(
            qdrant_service=mock_qdrant_service,
            config=config
        )
        
        # Create state with conversation ID
        state = create_test_state(
            user_query="What documents do I need?",
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # Process the state
        result_state = await memory_node.process(state)
        
        # Validate based on expected behavior
        assert_memory_metadata_valid(result_state.memory_metadata)
        
        if expected_behavior == "normal_retrieval":
            # Should retrieve some memory turns
            assert result_state.memory_metadata["turns_retrieved"] > 0
            assert len(result_state.memory_context) > 0
            
        elif expected_behavior == "limited_retrieval":
            # Should retrieve fewer turns due to restrictions
            assert result_state.memory_metadata["turns_retrieved"] <= 2
            if result_state.memory_metadata["turns_retrieved"] > 0:
                assert result_state.memory_metadata["avg_relevance_score"] >= 0.8
                
        elif expected_behavior == "expanded_retrieval":
            # Should retrieve more turns with lower threshold
            # (Limited by available test data)
            assert result_state.memory_metadata["memory_enabled"] == True
            
        elif expected_behavior == "minimal_or_no_retrieval":
            # Edge case config should retrieve very little or nothing
            assert result_state.memory_metadata["turns_retrieved"] == 0
            assert result_state.memory_context == ""
    
    @pytest.mark.asyncio
    async def test_memory_config_merge_consecutive_turns(self, mock_qdrant_service):
        """Test MemoryConfig with merge_consecutive_turns enabled"""
        
        # Create config with consecutive turn merging
        config = MemoryConfig(
            max_turns=5,
            score_threshold=0.5,
            include_recent_turns=2,
            merge_consecutive_turns=True,
            max_context_length=2000
        )
        
        memory_node = MemoryRetrievalNode(
            qdrant_service=mock_qdrant_service,
            config=config
        )
        
        # Create state with conversation ID
        state = create_test_state(
            user_query="What documents do I need?",
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # Process the state
        result_state = await memory_node.process(state)
        
        # Should handle consecutive turn merging
        assert result_state.memory_metadata["memory_enabled"] == True
        
        if result_state.memory_metadata["turns_retrieved"] > 0:
            # Memory context should be properly formatted
            assert "## Conversation History:" in result_state.memory_context
            assert len(result_state.memory_context) > 0


class TestMemoryConfigValidation:
    """Test MemoryConfig validation and error handling"""
    
    def test_memory_config_invalid_max_turns(self):
        """Test MemoryConfig with invalid max_turns values"""
        
        # Negative max_turns should be handled gracefully
        config = MemoryConfig(max_turns=-1)
        assert config.max_turns >= 0  # Should be corrected or default
    
    def test_memory_config_invalid_score_threshold(self):
        """Test MemoryConfig with invalid score_threshold values"""
        
        # Score threshold outside 0-1 range
        config_high = MemoryConfig(score_threshold=1.5)
        assert 0.0 <= config_high.score_threshold <= 1.0
        
        config_low = MemoryConfig(score_threshold=-0.5)
        assert 0.0 <= config_low.score_threshold <= 1.0
    
    def test_memory_config_invalid_context_length(self):
        """Test MemoryConfig with invalid max_context_length values"""
        
        # Negative context length should be handled
        config = MemoryConfig(max_context_length=-100)
        assert config.max_context_length > 0  # Should be corrected or default
    
    def test_memory_config_defaults(self):
        """Test MemoryConfig default values are reasonable"""
        
        config = MemoryConfig()
        
        # Check that defaults are sensible
        assert config.max_turns > 0
        assert 0.0 <= config.score_threshold <= 1.0
        assert config.include_recent_turns >= 0
        assert config.max_context_length > 0
        assert isinstance(config.merge_consecutive_turns, bool)
    
    @pytest.mark.asyncio
    async def test_memory_config_extreme_values(self):
        """Test MemoryConfig with extreme but valid values"""
        
        # Very large values
        config_large = MemoryConfig(
            max_turns=1000,
            score_threshold=0.0,
            include_recent_turns=100,
            max_context_length=100000
        )
        
        # Very small values
        config_small = MemoryConfig(
            max_turns=1,
            score_threshold=0.99,
            include_recent_turns=0,
            max_context_length=50
        )
        
        # Both should be valid configurations
        assert config_large.max_turns == 1000
        assert config_small.max_turns == 1
        assert config_large.score_threshold == 0.0
        assert config_small.score_threshold == 0.99


class TestMemoryRetrievalErrorHandling:
    """Test error handling in memory retrieval with various configs"""
    
    @pytest.fixture
    def error_prone_qdrant_service(self):
        """Create mock Qdrant service that sometimes fails"""
        config = MockScenarioConfig(
            memory_enabled=True,
            documents_available=False,
            response_latency=0.01,
            error_rate=0.5  # 50% error rate
        )
        return ProductionMockQdrantService(config)
    
    @pytest.mark.asyncio
    async def test_memory_retrieval_with_errors_and_edge_config(self, error_prone_qdrant_service):
        """Test memory retrieval error handling with edge case configs"""
        
        # Create edge case config
        config = MemoryConfig(
            max_turns=0,  # Edge case: no turns allowed
            score_threshold=1.0,  # Edge case: perfect match required
            include_recent_turns=10,  # Edge case: more than available
            max_context_length=10  # Edge case: very small context
        )
        
        memory_node = MemoryRetrievalNode(
            qdrant_service=error_prone_qdrant_service,
            config=config
        )
        
        # Create state
        state = create_test_state(
            user_query="What documents do I need?",
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # Process the state - should handle errors gracefully
        result_state = await memory_node.process(state)
        
        # Should return valid metadata even with errors and edge config
        assert_memory_metadata_valid(result_state.memory_metadata)
        assert result_state.memory_metadata["memory_enabled"] == True
        
        # With max_turns=0, should return no memory regardless of errors
        assert result_state.memory_metadata["turns_retrieved"] == 0
        assert result_state.memory_context == ""
        
        # Should have error information if retrieval was attempted and failed
        if result_state.memory_metadata.get("retrieval_error"):
            assert isinstance(result_state.memory_metadata["retrieval_error"], str)
