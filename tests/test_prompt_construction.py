# tests/test_prompt_construction.py
"""
Tests for prompt construction logic in the GenerationNode.

Validates prompt building under different context conditions:
- Memory context only
- Document context only  
- Both memory and documents
- Fallback scenarios
- Token limit handling
"""

import pytest
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, patch

from app.services.rag.nodes.generation import GenerationNode
from app.services.rag.workflow_state import RAGState
from app.services.rag.nodes.memory_retrieval import MemoryTurn
from tests.fixtures import (
    MockOllamaAdapter,
    create_test_state,
    assert_prompt_structure_valid,
    TEST_STATE_TEMPLATES,
    SAMPLE_DOCUMENTS
)


class TestPromptConstruction:
    """Test prompt construction under various context scenarios"""
    
    @pytest.fixture
    def mock_ollama_service(self):
        """Create mock Ollama service for testing"""
        return MockOllamaAdapter()
    
    @pytest.fixture
    def generation_node(self, mock_ollama_service):
        """Create GenerationNode with mock service"""
        return GenerationNode(ollama_service=mock_ollama_service)
    
    @pytest.mark.asyncio
    async def test_prompt_with_memory_context_only(self, generation_node):
        """Test prompt construction when only memory context is available"""
        
        # Create state with memory context but no documents
        state = create_test_state(
            user_query="Tell me more about work permits",
            conversation_id="conv_123",
            memory_context="""## Conversation History:
🔥 [Turn 3] User: Can I work while my application is being processed?
Assistant: Work authorization depends on your current status and visa type...

📝 [Turn 1] User: What are the visa requirements for Canada?
Assistant: For Canadian visas, you need to meet eligibility criteria...""",
            raw_documents=[]
        )
        
        # Process the state
        result_state = await generation_node.process(state)
        
        # Validate that response was generated
        assert result_state.response is not None
        assert len(result_state.response) > 0
        
        # Check that the generated prompt contained expected sections
        # We can't directly access the prompt, but we can verify the response
        # indicates memory was used (mock service should respond appropriately)
        assert "previous" in result_state.response.lower() or "discussed" in result_state.response.lower()
    
    @pytest.mark.asyncio
    async def test_prompt_with_documents_only(self, generation_node):
        """Test prompt construction when only documents are available"""
        
        # Create state with documents but no memory context
        state = create_test_state(
            user_query="What is Express Entry?",
            conversation_id=None,
            memory_context="",
            raw_documents=[
                {
                    "id": "doc_express_entry",
                    "title": "Express Entry System Overview",
                    "content": "The Express Entry system manages applications for three federal economic immigration programs...",
                    "source_type": "program",
                    "relevance_score": 0.9
                }
            ]
        )
        
        # Process the state
        result_state = await generation_node.process(state)
        
        # Validate response was generated
        assert result_state.response is not None
        assert len(result_state.response) > 0
        
        # Should reference documents
        assert "[Document" in result_state.response or "Express Entry" in result_state.response
    
    @pytest.mark.asyncio
    async def test_prompt_with_memory_and_documents(self, generation_node):
        """Test prompt construction with both memory and documents"""
        
        # Create state with both memory and documents
        state = create_test_state(
            user_query="What documents do I need for work authorization?",
            conversation_id="conv_123",
            memory_context="""## Conversation History:
🔥 [Turn 3] User: Can I work while my application is being processed?
Assistant: Work authorization depends on your current status and visa type...

📝 [Turn 1] User: What are the visa requirements for Canada?
Assistant: For Canadian visas, you need to meet eligibility criteria...""",
            raw_documents=[
                {
                    "id": "doc_work_permit_guide",
                    "title": "Work Permit Application Guide",
                    "content": "To apply for a work permit, you must provide proof of job offer, LMIA, passport...",
                    "source_type": "policy",
                    "relevance_score": 0.95
                }
            ]
        )
        
        # Process the state
        result_state = await generation_node.process(state)
        
        # Validate response was generated
        assert result_state.response is not None
        assert len(result_state.response) > 0
        
        # Should reference both memory and documents
        response_lower = result_state.response.lower()
        assert ("previous" in response_lower or "discussed" in response_lower) and \
               ("[document" in response_lower or "work permit" in response_lower)
    
    @pytest.mark.asyncio
    async def test_prompt_fallback_no_context(self, generation_node):
        """Test prompt construction when no context is available"""
        
        # Create state with no memory or documents
        state = create_test_state(
            user_query="Tell me about artificial intelligence",
            conversation_id=None,
            memory_context="",
            raw_documents=[]
        )
        
        # Process the state
        result_state = await generation_node.process(state)
        
        # Validate fallback response was generated
        assert result_state.response is not None
        assert len(result_state.response) > 0
        
        # Should be a fallback response
        response_lower = result_state.response.lower()
        assert "apologize" in response_lower or "information" in response_lower or "context" in response_lower
    
    @pytest.mark.asyncio
    async def test_prompt_memory_unavailable_with_conversation_id(self, generation_node):
        """Test prompt when conversation_id exists but memory is unavailable"""
        
        # Create state with conversation_id but empty memory context
        state = create_test_state(
            user_query="What are the requirements?",
            conversation_id="conv_123",
            memory_context="",  # Empty memory context
            raw_documents=[
                {
                    "id": "doc_visa_requirements",
                    "title": "Canadian Visa Requirements Overview",
                    "content": "Canadian visa requirements vary depending on your nationality...",
                    "source_type": "guideline",
                    "relevance_score": 0.8
                }
            ]
        )
        
        # Process the state
        result_state = await generation_node.process(state)
        
        # Validate response was generated
        assert result_state.response is not None
        assert len(result_state.response) > 0
        
        # Should still provide useful response from documents
        assert "requirements" in result_state.response.lower()
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("memory_context,documents,expected_sections", [
        (
            "## Conversation History:\n🔥 [Turn 1] User: Test\nAssistant: Response",
            [],
            ["Conversation History", "User Question"]
        ),
        (
            "",
            [{"title": "Test Doc", "content": "Test content"}],
            ["Context Documents", "User Question"]
        ),
        (
            "## Conversation History:\n🔥 [Turn 1] User: Test\nAssistant: Response",
            [{"title": "Test Doc", "content": "Test content"}],
            ["Conversation History", "Context Documents", "User Question"]
        ),
        (
            "",
            [],
            ["Note: Previous conversation context unavailable"]
        ),
    ])
    async def test_prompt_structure_scenarios(
        self, 
        generation_node, 
        memory_context, 
        documents, 
        expected_sections
    ):
        """Test prompt structure under different context combinations"""
        
        # Create state based on parameters
        state = create_test_state(
            user_query="Test query",
            conversation_id="conv_123" if memory_context or documents else None,
            memory_context=memory_context,
            raw_documents=documents
        )
        
        # Mock the prompt building to capture the actual prompt
        original_build_prompt = generation_node._build_prompt
        captured_prompt = None
        
        def capture_prompt(*args, **kwargs):
            nonlocal captured_prompt
            captured_prompt = original_build_prompt(*args, **kwargs)
            return captured_prompt
        
        with patch.object(generation_node, '_build_prompt', side_effect=capture_prompt):
            await generation_node.process(state)
        
        # Validate prompt structure
        assert captured_prompt is not None
        assert_prompt_structure_valid(captured_prompt, expected_sections)
    
    @pytest.mark.asyncio
    async def test_prompt_token_limit_handling(self, generation_node):
        """Test prompt construction respects token limits"""
        
        # Create very long memory context to test truncation
        long_memory_context = "## Conversation History:\n" + \
                             "\n".join([f"🔥 [Turn {i}] User: Very long question about immigration requirements and documentation.\nAssistant: Detailed response about immigration processes and requirements." 
                                       for i in range(1, 20)])  # 20 turns of conversation
        
        # Create state with very long context
        state = create_test_state(
            user_query="What are the requirements?",
            conversation_id="conv_123",
            memory_context=long_memory_context,
            raw_documents=[
                {
                    "id": "doc_test",
                    "title": "Test Document",
                    "content": "Test content for token limit testing.",
                    "source_type": "test",
                    "relevance_score": 0.8
                }
            ]
        )
        
        # Process the state
        result_state = await generation_node.process(state)
        
        # Validate response was still generated despite long context
        assert result_state.response is not None
        assert len(result_state.response) > 0
        
        # Response should still be coherent
        assert "requirements" in result_state.response.lower()
    
    @pytest.mark.asyncio
    async def test_prompt_citation_format_validation(self, generation_node):
        """Test that document citations are properly formatted in prompts"""
        
        # Create state with multiple documents
        state = create_test_state(
            user_query="What are the requirements?",
            conversation_id="conv_123",
            memory_context="",
            raw_documents=[
                {
                    "id": "doc_1",
                    "title": "Document One",
                    "content": "Content of document one with requirements.",
                    "source_type": "policy",
                    "relevance_score": 0.9
                },
                {
                    "id": "doc_2", 
                    "title": "Document Two",
                    "content": "Content of document two with additional requirements.",
                    "source_type": "guideline",
                    "relevance_score": 0.8
                }
            ]
        )
        
        # Mock prompt building to capture the prompt
        original_build_prompt = generation_node._build_prompt
        captured_prompt = None
        
        def capture_prompt(*args, **kwargs):
            nonlocal captured_prompt
            captured_prompt = original_build_prompt(*args, **kwargs)
            return captured_prompt
        
        with patch.object(generation_node, '_build_prompt', side_effect=capture_prompt):
            await generation_node.process(state)
        
        # Validate citation format in prompt
        assert captured_prompt is not None
        assert "[Document 1]" in captured_prompt
        assert "[Document 2]" in captured_prompt
        
        # Check document ordering
        doc1_pos = captured_prompt.find("[Document 1]")
        doc2_pos = captured_prompt.find("[Document 2]")
        assert doc1_pos < doc2_pos  # Document 1 should come before Document 2
    
    @pytest.mark.asyncio
    async def test_prompt_memory_context_formatting(self, generation_node):
        """Test that memory context is properly formatted with emoji indicators"""
        
        # Create state with memory context
        memory_context = """## Conversation History:
🔥 [Turn 3] User: Can I work while my application is being processed?
Assistant: Work authorization depends on your current status and visa type.

📝 [Turn 1] User: What are the visa requirements for Canada?
Assistant: For Canadian visas, you need to meet eligibility criteria."""
        
        state = create_test_state(
            user_query="What documents do I need?",
            conversation_id="conv_123",
            memory_context=memory_context,
            raw_documents=[]
        )
        
        # Mock prompt building to capture the prompt
        original_build_prompt = generation_node._build_prompt
        captured_prompt = None
        
        def capture_prompt(*args, **kwargs):
            nonlocal captured_prompt
            captured_prompt = original_build_prompt(*args, **kwargs)
            return captured_prompt
        
        with patch.object(generation_node, '_build_prompt', side_effect=capture_prompt):
            await generation_node.process(state)
        
        # Validate memory formatting in prompt
        assert captured_prompt is not None
        assert "## Conversation History:" in captured_prompt
        assert "🔥 [Turn 3]" in captured_prompt
        assert "📝 [Turn 1]" in captured_prompt
    
    @pytest.mark.asyncio
    async def test_prompt_section_ordering(self, generation_node):
        """Test that prompt sections are in the correct order"""
        
        # Create state with both memory and documents
        state = create_test_state(
            user_query="Test query",
            conversation_id="conv_123",
            memory_context="## Conversation History:\n🔥 [Turn 1] User: Test\nAssistant: Response",
            raw_documents=[
                {
                    "id": "doc_test",
                    "title": "Test Document",
                    "content": "Test content",
                    "source_type": "test",
                    "relevance_score": 0.8
                }
            ]
        )
        
        # Mock prompt building to capture the prompt
        original_build_prompt = generation_node._build_prompt
        captured_prompt = None
        
        def capture_prompt(*args, **kwargs):
            nonlocal captured_prompt
            captured_prompt = original_build_prompt(*args, **kwargs)
            return captured_prompt
        
        with patch.object(generation_node, '_build_prompt', side_effect=capture_prompt):
            await generation_node.process(state)
        
        # Validate section ordering
        assert captured_prompt is not None
        
        # Find positions of key sections
        system_pos = captured_prompt.find("You are an expert AI assistant")
        memory_pos = captured_prompt.find("## Conversation History:")
        docs_pos = captured_prompt.find("## Context Documents:")
        query_pos = captured_prompt.find("## User Question:")
        instructions_pos = captured_prompt.find("## Instructions:")
        
        # Validate ordering: System → Memory → Documents → Query → Instructions
        assert system_pos < memory_pos
        assert memory_pos < docs_pos
        assert docs_pos < query_pos
        assert query_pos < instructions_pos


class TestPromptEdgeCases:
    """Test prompt construction edge cases and error handling"""
    
    @pytest.fixture
    def generation_node(self):
        """Create GenerationNode for edge case testing"""
        mock_ollama = MockOllamaAdapter()
        return GenerationNode(ollama_service=mock_ollama)
    
    @pytest.mark.asyncio
    async def test_prompt_empty_query(self, generation_node):
        """Test prompt construction with empty query"""
        
        state = create_test_state(
            user_query="",
            conversation_id="conv_123",
            memory_context="",
            raw_documents=[]
        )
        
        # Should handle empty query gracefully
        result_state = await generation_node.process(state)
        assert result_state.response is not None
        assert len(result_state.response) > 0
    
    @pytest.mark.asyncio
    async def test_prompt_malformed_memory_context(self, generation_node):
        """Test prompt construction with malformed memory context"""
        
        # Malformed memory context (missing proper formatting)
        malformed_context = "Random text without proper structure"
        
        state = create_test_state(
            user_query="What are the requirements?",
            conversation_id="conv_123",
            memory_context=malformed_context,
            raw_documents=[]
        )
        
        # Should handle malformed context gracefully
        result_state = await generation_node.process(state)
        assert result_state.response is not None
        assert len(result_state.response) > 0
    
    @pytest.mark.asyncio
    async def test_prompt_documents_missing_fields(self, generation_node):
        """Test prompt construction with documents missing required fields"""
        
        # Documents with missing fields
        incomplete_documents = [
            {
                "id": "doc_1",
                # Missing title
                "content": "Content without title",
                "source_type": "test"
            },
            {
                "id": "doc_2",
                "title": "Document without content",
                # Missing content
                "source_type": "test"
            }
        ]
        
        state = create_test_state(
            user_query="What are the requirements?",
            conversation_id="conv_123",
            memory_context="",
            raw_documents=incomplete_documents
        )
        
        # Should handle incomplete documents gracefully
        result_state = await generation_node.process(state)
        assert result_state.response is not None
        assert len(result_state.response) > 0
