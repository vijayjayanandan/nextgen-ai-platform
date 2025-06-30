# tests/test_nodes/test_generation.py
"""
Unit tests for GenerationNode - validates prompt construction, LLM integration,
context handling, and response generation with various scenarios.
"""

import pytest
from unittest.mock import AsyncMock, patch
from app.services.rag.nodes.generation import GenerationNode
from app.services.rag.workflow_state import RAGState
from tests.fixtures import (
    MockOllamaAdapter,
    create_test_state,
    assert_prompt_structure_valid
)


class TestGenerationNode:
    """Test suite for GenerationNode functionality"""
    
    @pytest.fixture
    def mock_ollama(self):
        """Create mock Ollama service with deterministic responses"""
        return MockOllamaAdapter({
            "work_authorization": """Based on the provided documentation, here are the key documents you'll need for work authorization [Document 1]:

**Primary Documents:**
- Valid passport
- Job offer letter from Canadian employer
- Labour Market Impact Assessment (LMIA) if required
- Educational credential assessment

**Supporting Documents [Document 2]:**
- Medical examination results
- Police certificates
- Proof of funds
- Passport-style photographs

The processing time is typically 2-12 weeks depending on your country of residence.""",
            
            "visa_requirements": """For Canadian visas, you need to meet several eligibility criteria [Document 1]:

**Basic Requirements:**
- Valid passport
- Completed application forms
- Application fees
- Biometric information

**Supporting Documentation [Document 2]:**
- Proof of financial support
- Medical examination (if required)
- Police certificates
- Purpose of visit documentation

Processing times vary by visa type, typically 2-4 weeks for visitor visas.""",
            
            "fallback": """I apologize, but I don't have sufficient information in the provided context to answer your specific question. Please provide more details about what you'd like to know about Canadian immigration, or rephrase your question to be more specific."""
        })
    
    @pytest.fixture
    def generation_node(self, mock_ollama):
        """Create GenerationNode with mock service"""
        return GenerationNode(mock_ollama)
    
    @pytest.mark.asyncio
    async def test_generation_with_documents_and_memory(self, generation_node, mock_ollama):
        """Test generation with both documents and memory context"""
        
        # Setup state with both memory and documents
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
                    "content": "To apply for a work permit, you must provide several key documents...",
                    "source_type": "policy",
                    "relevance_score": 0.9
                }
            ]
        )
        
        mock_ollama.response_templates["default"] = mock_ollama.response_templates["work_authorization"]
        
        result_state = await generation_node.execute(state)
        
        # Validate response was generated
        assert result_state.response is not None
        assert len(result_state.response) > 100
        assert "documents" in result_state.response.lower()
        assert "work authorization" in result_state.response.lower()
        
        # Validate prompt was constructed
        assert result_state.context_prompt is not None
        assert len(result_state.context_prompt) > 0
        
        # Validate model was recorded
        assert result_state.model_used is not None
        
        # Verify LLM was called
        assert mock_ollama.get_generation_count() == 1
    
    @pytest.mark.asyncio
    async def test_generation_documents_only(self, generation_node, mock_ollama):
        """Test generation with documents but no memory context"""
        
        state = create_test_state(
            user_query="What are visa requirements?",
            conversation_id=None,  # No memory
            raw_documents=[
                {
                    "id": "doc_visa_requirements",
                    "title": "Canadian Visa Requirements",
                    "content": "Canadian visa requirements vary depending on nationality...",
                    "source_type": "guideline",
                    "relevance_score": 0.85
                }
            ]
        )
        
        mock_ollama.response_templates["default"] = mock_ollama.response_templates["visa_requirements"]
        
        result_state = await generation_node.execute(state)
        
        # Should generate response from documents only
        assert result_state.response is not None
        assert "visa" in result_state.response.lower()
        assert "requirements" in result_state.response.lower()
        
        # Prompt should contain documents but no memory section
        prompt = result_state.context_prompt
        assert "Context Documents" in prompt
        assert "Conversation History" not in prompt
    
    @pytest.mark.asyncio
    async def test_generation_memory_only(self, generation_node, mock_ollama):
        """Test generation with memory context but no documents"""
        
        state = create_test_state(
            user_query="Tell me more about what we discussed",
            conversation_id="conv_123",
            memory_context="""## Conversation History:
📝 [Turn 2] User: What documents do I need for work permit?
Assistant: For work permits, you need a job offer, LMIA...""",
            raw_documents=[]  # No documents
        )
        
        mock_ollama.response_templates["default"] = "Based on our previous discussion about work permits, I can provide more details..."
        
        result_state = await generation_node.execute(state)
        
        # Should generate response from memory only
        assert result_state.response is not None
        assert len(result_state.response) > 50
        
        # Prompt should contain memory but no documents section
        prompt = result_state.context_prompt
        assert "Conversation History" in prompt
        assert "Context Documents" not in prompt
    
    @pytest.mark.asyncio
    async def test_generation_no_context_fallback(self, generation_node, mock_ollama):
        """Test generation fallback when no context is available"""
        
        state = create_test_state(
            user_query="Tell me about artificial intelligence",
            conversation_id=None,
            memory_context="",
            raw_documents=[]
        )
        
        mock_ollama.response_templates["default"] = mock_ollama.response_templates["fallback"]
        
        result_state = await generation_node.execute(state)
        
        # Should generate fallback response
        assert result_state.response is not None
        assert "apologize" in result_state.response.lower()
        assert "information" in result_state.response.lower()
        
        # Prompt should indicate no context available
        prompt = result_state.context_prompt
        assert "no relevant" in prompt.lower() or "insufficient" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_prompt_construction_structure(self, generation_node, mock_ollama):
        """Test that prompts are constructed with proper structure"""
        
        state = create_test_state(
            user_query="What are work permit requirements?",
            memory_context="Previous conversation about immigration...",
            raw_documents=[
                {
                    "id": "doc1",
                    "title": "Work Permit Guide",
                    "content": "Work permit requirements include...",
                    "source_type": "policy"
                }
            ]
        )
        
        # Capture the prompt
        captured_prompt = None
        
        async def capture_prompt(prompt, **kwargs):
            nonlocal captured_prompt
            captured_prompt = prompt
            return "Test response"
        
        mock_ollama.generate = capture_prompt
        
        await generation_node.execute(state)
        
        # Validate prompt structure
        assert captured_prompt is not None
        
        expected_sections = [
            "Conversation History",
            "Context Documents", 
            "User Question",
            "Instructions"
        ]
        
        assert_prompt_structure_valid(captured_prompt, expected_sections)
        
        # Check specific content
        assert state.user_query in captured_prompt
        assert "Work Permit Guide" in captured_prompt
        assert "[Document 1]" in captured_prompt
    
    @pytest.mark.asyncio
    async def test_document_numbering_in_prompt(self, generation_node, mock_ollama):
        """Test that documents are properly numbered in prompts"""
        
        documents = [
            {"id": "doc1", "title": "Guide 1", "content": "Content 1", "source_type": "policy"},
            {"id": "doc2", "title": "Guide 2", "content": "Content 2", "source_type": "guideline"},
            {"id": "doc3", "title": "Guide 3", "content": "Content 3", "source_type": "reference"}
        ]
        
        state = create_test_state(
            user_query="Test query",
            raw_documents=documents
        )
        
        captured_prompt = None
        
        async def capture_prompt(prompt, **kwargs):
            nonlocal captured_prompt
            captured_prompt = prompt
            return "Test response"
        
        mock_ollama.generate = capture_prompt
        
        await generation_node.execute(state)
        
        # Check document numbering
        assert "[Document 1]" in captured_prompt
        assert "[Document 2]" in captured_prompt
        assert "[Document 3]" in captured_prompt
        
        # Check titles are included
        assert "Guide 1" in captured_prompt
        assert "Guide 2" in captured_prompt
        assert "Guide 3" in captured_prompt
    
    @pytest.mark.asyncio
    async def test_llm_service_error_handling(self, generation_node, mock_ollama):
        """Test handling of LLM service errors"""
        
        state = create_test_state(
            user_query="Test query",
            raw_documents=[{"id": "doc1", "title": "Test", "content": "Test content", "source_type": "test"}]
        )
        
        # Mock LLM to raise exception
        mock_ollama.generate = AsyncMock(side_effect=Exception("LLM service unavailable"))
        
        result_state = await generation_node.execute(state)
        
        # Should handle error gracefully
        assert result_state.error_message is not None
        assert "generation failed" in result_state.error_message.lower()
        assert result_state.response is None or len(result_state.response) == 0
    
    @pytest.mark.asyncio
    async def test_empty_llm_response_handling(self, generation_node, mock_ollama):
        """Test handling of empty LLM responses"""
        
        state = create_test_state(user_query="Test query")
        
        # Mock empty response
        mock_ollama.response_templates["default"] = ""
        
        result_state = await generation_node.execute(state)
        
        # Should handle empty response
        assert result_state.error_message is not None
        assert "empty response" in result_state.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_model_parameters(self, generation_node, mock_ollama):
        """Test that correct model parameters are used"""
        
        state = create_test_state(user_query="Test query")
        
        captured_params = None
        
        async def capture_params(**kwargs):
            nonlocal captured_params
            captured_params = kwargs
            return "Test response"
        
        mock_ollama.generate = capture_params
        
        await generation_node.execute(state)
        
        # Validate model parameters
        assert captured_params is not None
        assert captured_params.get("model") == "deepseek-coder:6.7b"
        assert captured_params.get("temperature") == 0.3
        assert captured_params.get("max_tokens") == 1000
    
    @pytest.mark.asyncio
    async def test_context_length_management(self, generation_node, mock_ollama):
        """Test handling of very long context that might exceed limits"""
        
        # Create very long documents
        long_documents = []
        for i in range(10):
            long_content = "Very long document content. " * 200  # ~5000 chars each
            long_documents.append({
                "id": f"doc_{i}",
                "title": f"Long Document {i}",
                "content": long_content,
                "source_type": "policy"
            })
        
        long_memory = "Previous conversation context. " * 100  # ~3000 chars
        
        state = create_test_state(
            user_query="Test query with very long context",
            memory_context=long_memory,
            raw_documents=long_documents
        )
        
        captured_prompt = None
        
        async def capture_prompt(prompt, **kwargs):
            nonlocal captured_prompt
            captured_prompt = prompt
            return "Test response"
        
        mock_ollama.generate = capture_prompt
        
        await generation_node.execute(state)
        
        # Should handle long context (may truncate or summarize)
        assert captured_prompt is not None
        assert len(captured_prompt) > 0
        
        # Should still contain essential elements
        assert state.user_query in captured_prompt
    
    @pytest.mark.asyncio
    async def test_special_characters_in_content(self, generation_node, mock_ollama):
        """Test handling of special characters and formatting in content"""
        
        special_content = """
        Content with special characters: àáâãäåæçèéêë
        Unicode symbols: ★☆♠♣♥♦
        Code snippets: `function test() { return "hello"; }`
        Markdown: **bold** *italic* [link](url)
        JSON: {"key": "value", "number": 123}
        """
        
        state = create_test_state(
            user_query="Test query with special content",
            raw_documents=[{
                "id": "special_doc",
                "title": "Special Characters Document",
                "content": special_content,
                "source_type": "technical"
            }]
        )
        
        mock_ollama.response_templates["default"] = "Response with special content handled properly."
        
        result_state = await generation_node.execute(state)
        
        # Should handle special characters without errors
        assert result_state.response is not None
        assert result_state.error_message is None
        assert result_state.context_prompt is not None
    
    @pytest.mark.asyncio
    async def test_state_preservation(self, generation_node, mock_ollama):
        """Test that original state fields are preserved during generation"""
        
        original_state = create_test_state(
            user_query="test query",
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # Add existing state data
        original_state.query_type = "simple"
        original_state.intent = "test intent"
        original_state.entities = ["test", "entity"]
        original_state.retrieval_strategy = "hybrid"
        original_state.raw_documents = [{"id": "doc1", "title": "Test", "content": "Test", "source_type": "test"}]
        
        mock_ollama.response_templates["default"] = "Test response"
        
        result_state = await generation_node.execute(original_state)
        
        # Original fields should be preserved
        assert result_state.user_query == original_state.user_query
        assert result_state.conversation_id == original_state.conversation_id
        assert result_state.user_id == original_state.user_id
        assert result_state.query_type == original_state.query_type
        assert result_state.intent == original_state.intent
        assert result_state.entities == original_state.entities
        
        # New fields should be populated
        assert result_state.response is not None
        assert result_state.context_prompt is not None
        assert result_state.model_used is not None


@pytest.mark.integration
class TestGenerationIntegration:
    """Integration tests for GenerationNode with realistic scenarios"""
    
    @pytest.fixture
    def integration_node(self):
        """Create node for integration testing"""
        mock_ollama = MockOllamaAdapter()
        return GenerationNode(mock_ollama)
    
    @pytest.mark.asyncio
    async def test_immigration_domain_generation(self, integration_node):
        """Test generation for various immigration domain scenarios"""
        
        immigration_scenarios = [
            {
                "query": "How do I apply for Canadian citizenship?",
                "documents": [
                    {
                        "id": "citizenship_guide",
                        "title": "Canadian Citizenship Application Guide",
                        "content": "To apply for Canadian citizenship, you must meet residency requirements, pass language tests, and complete the citizenship test...",
                        "source_type": "official_guide"
                    }
                ],
                "expected_keywords": ["citizenship", "application", "requirements"]
            },
            {
                "query": "What documents are needed for work permit?",
                "documents": [
                    {
                        "id": "work_permit_docs",
                        "title": "Work Permit Documentation Requirements",
                        "content": "Required documents include job offer letter, LMIA, passport, educational credentials...",
                        "source_type": "requirements_list"
                    }
                ],
                "expected_keywords": ["documents", "work permit", "job offer"]
            }
        ]
        
        for scenario in immigration_scenarios:
            state = create_test_state(
                user_query=scenario["query"],
                raw_documents=scenario["documents"]
            )
            
            # Configure appropriate response
            integration_node.ollama.response_templates["default"] = f"Based on the documentation, here's what you need to know about {scenario['query'].lower()}..."
            
            result = await integration_node.execute(state)
            
            # Validate response quality
            assert result.response is not None
            assert len(result.response) > 100
            
            # Check for expected keywords
            response_lower = result.response.lower()
            for keyword in scenario["expected_keywords"]:
                assert keyword.lower() in response_lower
    
    @pytest.mark.asyncio
    async def test_conversational_context_integration(self, integration_node):
        """Test generation with conversational context"""
        
        conversation_state = create_test_state(
            user_query="Can you tell me more about the processing times?",
            conversation_id="conv_123",
            memory_context="""## Conversation History:
📝 [Turn 1] User: What are the requirements for work permits?
Assistant: Work permits require a job offer, LMIA, and supporting documents...

📝 [Turn 2] User: How long does the application take?
Assistant: Processing times vary by country, typically 2-12 weeks...""",
            raw_documents=[
                {
                    "id": "processing_times",
                    "title": "Immigration Processing Times",
                    "content": "Current processing times for various immigration applications...",
                    "source_type": "current_data"
                }
            ]
        )
        
        integration_node.ollama.response_templates["default"] = "Based on our previous discussion about work permits and the current processing information..."
        
        result = await integration_node.execute(conversation_state)
        
        # Should reference previous conversation
        assert result.response is not None
        assert "previous" in result.response.lower() or "discussed" in result.response.lower()
        assert "processing" in result.response.lower()
        
        # Prompt should include both memory and documents
        assert "Conversation History" in result.context_prompt
        assert "Context Documents" in result.context_prompt
    
    @pytest.mark.asyncio
    async def test_response_quality_metrics(self, integration_node):
        """Test response quality across different scenarios"""
        
        quality_test_cases = [
            {
                "scenario": "detailed_query",
                "query": "What are the specific eligibility requirements for Express Entry?",
                "min_length": 200,
                "should_contain": ["eligibility", "Express Entry", "requirements"]
            },
            {
                "scenario": "simple_query", 
                "query": "Do I need a visa?",
                "min_length": 100,
                "should_contain": ["visa", "need", "depend"]
            },
            {
                "scenario": "complex_query",
                "query": "Compare the advantages and disadvantages of different immigration pathways",
                "min_length": 300,
                "should_contain": ["advantages", "disadvantages", "pathways"]
            }
        ]
        
        for test_case in quality_test_cases:
            state = create_test_state(
                user_query=test_case["query"],
                raw_documents=[
                    {
                        "id": f"doc_{test_case['scenario']}",
                        "title": f"Guide for {test_case['scenario']}",
                        "content": f"Comprehensive information about {test_case['query']}...",
                        "source_type": "guide"
                    }
                ]
            )
            
            # Configure response based on scenario
            integration_node.ollama.response_templates["default"] = f"Detailed response for {test_case['query']} with comprehensive information..."
            
            result = await integration_node.execute(state)
            
            # Validate response quality
            assert result.response is not None
            assert len(result.response) >= test_case["min_length"]
            
            response_lower = result.response.lower()
            for keyword in test_case["should_contain"]:
                assert keyword.lower() in response_lower
