# tests/test_nodes/test_citation.py
"""
Unit tests for CitationNode - validates citation extraction, document linking,
and source attribution functionality.
"""

import pytest
import re
from unittest.mock import patch
from app.services.rag.nodes.citation import CitationNode
from app.services.rag.workflow_state import RAGState
from tests.fixtures import (
    create_test_state,
    assert_citation_format_valid,
    assert_document_metadata_valid
)


class TestCitationNode:
    """Test suite for CitationNode functionality"""
    
    @pytest.fixture
    def citation_node(self):
        """Create CitationNode instance"""
        return CitationNode()
    
    @pytest.mark.asyncio
    async def test_citation_extraction_with_valid_citations(self, citation_node):
        """Test extraction of properly formatted citations from response"""
        
        response_with_citations = """Based on the provided documentation, here are the key requirements [Document 1]:

**Primary Documents:**
- Valid passport
- Job offer letter from Canadian employer [Document 2]
- Labour Market Impact Assessment (LMIA) if required

**Supporting Documents [Document 3]:**
- Medical examination results
- Police certificates [Document 1]
- Proof of funds

The processing time varies by country [Document 2]."""
        
        documents = [
            {
                "id": "doc_work_permit_guide",
                "title": "Work Permit Application Guide",
                "content": "To apply for a work permit, you must provide several key documents...",
                "source_type": "policy"
            },
            {
                "id": "doc_processing_times",
                "title": "Processing Times Guide",
                "content": "Processing times vary by country and application type...",
                "source_type": "guideline"
            },
            {
                "id": "doc_supporting_docs",
                "title": "Supporting Documentation Requirements",
                "content": "Additional supporting documents may be required...",
                "source_type": "requirements"
            }
        ]
        
        state = create_test_state(
            user_query="What documents do I need for work authorization?",
            raw_documents=documents
        )
        state.response = response_with_citations
        
        result_state = await citation_node.execute(state)
        
        # Validate citations were extracted
        assert len(result_state.citations) > 0
        
        # Check citation format
        assert_citation_format_valid(result_state.citations)
        
        # Validate source documents were created
        assert len(result_state.source_documents) > 0
        assert_document_metadata_valid(result_state.source_documents)
        
        # Check that all cited documents are included in source documents
        cited_doc_numbers = {citation["document_number"] for citation in result_state.citations}
        source_doc_numbers = {doc["document_number"] for doc in result_state.source_documents}
        
        assert cited_doc_numbers.issubset(source_doc_numbers)
    
    @pytest.mark.asyncio
    async def test_citation_extraction_no_citations(self, citation_node):
        """Test handling of response with no citations"""
        
        response_without_citations = """Based on general knowledge, work permits typically require:

- Valid passport
- Job offer letter
- Educational credentials
- Medical examination

Processing times vary depending on your situation."""
        
        state = create_test_state(
            user_query="What documents do I need?",
            raw_documents=[
                {
                    "id": "doc1",
                    "title": "Test Document",
                    "content": "Test content",
                    "source_type": "test"
                }
            ]
        )
        state.response = response_without_citations
        
        result_state = await citation_node.execute(state)
        
        # Should handle gracefully with no citations
        assert len(result_state.citations) == 0
        assert len(result_state.source_documents) == 0
    
    @pytest.mark.asyncio
    async def test_citation_extraction_malformed_citations(self, citation_node):
        """Test handling of malformed citation patterns"""
        
        response_with_malformed = """Here's some information [Document] about work permits.
        
Also see [Doc 1] and [Document ABC] for more details.
        
The requirements include [Document 999] various documents."""
        
        state = create_test_state(
            user_query="Test query",
            raw_documents=[
                {"id": "doc1", "title": "Test", "content": "Test", "source_type": "test"}
            ]
        )
        state.response = response_with_malformed
        
        result_state = await citation_node.execute(state)
        
        # Should only extract valid citations (if any)
        for citation in result_state.citations:
            assert re.match(r"\[Document \d+\]", citation["citation_text"])
    
    @pytest.mark.asyncio
    async def test_citation_document_mapping(self, citation_node):
        """Test correct mapping between citations and source documents"""
        
        response = """Requirements include passport [Document 1] and job offer [Document 2].
        
Medical exams may be required [Document 3] depending on your country."""
        
        documents = [
            {"id": "passport_doc", "title": "Passport Requirements", "content": "...", "source_type": "policy"},
            {"id": "job_offer_doc", "title": "Job Offer Guidelines", "content": "...", "source_type": "guideline"},
            {"id": "medical_doc", "title": "Medical Examination Guide", "content": "...", "source_type": "medical"}
        ]
        
        state = create_test_state(user_query="Test", raw_documents=documents)
        state.response = response
        
        result_state = await citation_node.execute(state)
        
        # Validate document mapping
        assert len(result_state.citations) == 3
        assert len(result_state.source_documents) == 3
        
        # Check that document numbers are sequential and correct
        citation_numbers = sorted([c["document_number"] for c in result_state.citations])
        assert citation_numbers == [1, 2, 3]
        
        source_numbers = sorted([d["document_number"] for d in result_state.source_documents])
        assert source_numbers == [1, 2, 3]
        
        # Verify titles match expected documents
        source_by_number = {d["document_number"]: d for d in result_state.source_documents}
        assert source_by_number[1]["title"] == "Passport Requirements"
        assert source_by_number[2]["title"] == "Job Offer Guidelines"
        assert source_by_number[3]["title"] == "Medical Examination Guide"
    
    @pytest.mark.asyncio
    async def test_citation_position_tracking(self, citation_node):
        """Test tracking of citation positions in response text"""
        
        response = """Work permits require a passport [Document 1] and job offer letter [Document 2].
        
Processing times [Document 1] vary by country."""
        
        state = create_test_state(
            user_query="Test",
            raw_documents=[
                {"id": "doc1", "title": "Requirements", "content": "...", "source_type": "policy"},
                {"id": "doc2", "title": "Job Offers", "content": "...", "source_type": "guideline"}
            ]
        )
        state.response = response
        
        result_state = await citation_node.execute(state)
        
        # Check that positions are tracked
        for citation in result_state.citations:
            if "start_position" in citation:
                assert isinstance(citation["start_position"], int)
                assert citation["start_position"] >= 0
            
            if "end_position" in citation:
                assert isinstance(citation["end_position"], int)
                assert citation["end_position"] > citation.get("start_position", 0)
    
    @pytest.mark.asyncio
    async def test_duplicate_citation_handling(self, citation_node):
        """Test handling of duplicate citations to the same document"""
        
        response = """Passport requirements [Document 1] are important.
        
The passport [Document 1] must be valid for at least 6 months.
        
Additional passport information [Document 1] can be found in the guide."""
        
        state = create_test_state(
            user_query="Test",
            raw_documents=[
                {"id": "passport_doc", "title": "Passport Guide", "content": "...", "source_type": "policy"}
            ]
        )
        state.response = response
        
        result_state = await citation_node.execute(state)
        
        # Should have multiple citations but only one source document
        assert len(result_state.citations) == 3  # Three citations
        assert len(result_state.source_documents) == 1  # One unique document
        
        # All citations should reference document 1
        for citation in result_state.citations:
            assert citation["document_number"] == 1
            assert citation["citation_text"] == "[Document 1]"
    
    @pytest.mark.asyncio
    async def test_citation_with_no_documents(self, citation_node):
        """Test citation processing when no documents are available"""
        
        response = """This is a response with citations [Document 1] but no source documents."""
        
        state = create_test_state(
            user_query="Test",
            raw_documents=[]  # No documents
        )
        state.response = response
        
        result_state = await citation_node.execute(state)
        
        # Should handle gracefully
        assert len(result_state.citations) == 0
        assert len(result_state.source_documents) == 0
    
    @pytest.mark.asyncio
    async def test_citation_out_of_range_document_numbers(self, citation_node):
        """Test handling of citations referencing non-existent document numbers"""
        
        response = """Here's info from [Document 1] and [Document 5] and [Document 10]."""
        
        state = create_test_state(
            user_query="Test",
            raw_documents=[
                {"id": "doc1", "title": "Doc 1", "content": "...", "source_type": "policy"},
                {"id": "doc2", "title": "Doc 2", "content": "...", "source_type": "guideline"}
            ]  # Only 2 documents available
        )
        state.response = response
        
        result_state = await citation_node.execute(state)
        
        # Should only include valid citations
        valid_citations = [c for c in result_state.citations if c["document_number"] <= 2]
        assert len(valid_citations) == len(result_state.citations)
        
        # Should only include available documents
        assert len(result_state.source_documents) <= 2
    
    @pytest.mark.asyncio
    async def test_empty_response_handling(self, citation_node):
        """Test handling of empty or None response"""
        
        test_cases = [None, "", "   ", "\n\t"]
        
        for empty_response in test_cases:
            state = create_test_state(
                user_query="Test",
                raw_documents=[{"id": "doc1", "title": "Test", "content": "...", "source_type": "test"}]
            )
            state.response = empty_response
            
            result_state = await citation_node.execute(state)
            
            # Should handle gracefully
            assert len(result_state.citations) == 0
            assert len(result_state.source_documents) == 0
    
    @pytest.mark.asyncio
    async def test_citation_regex_pattern_validation(self, citation_node):
        """Test that citation regex pattern correctly identifies valid citations"""
        
        test_cases = [
            ("[Document 1]", True),
            ("[Document 123]", True),
            ("[Document 0]", True),
            ("[document 1]", False),  # lowercase
            ("[Document]", False),   # no number
            ("[Doc 1]", False),      # wrong format
            ("Document 1", False),   # no brackets
            ("[Document 1", False),  # missing closing bracket
            ("Document 1]", False),  # missing opening bracket
            ("[Document 1.5]", False), # decimal number
            ("[Document -1]", False),  # negative number
        ]
        
        for citation_text, should_match in test_cases:
            response = f"Test response with {citation_text} citation."
            
            state = create_test_state(
                user_query="Test",
                raw_documents=[{"id": "doc1", "title": "Test", "content": "...", "source_type": "test"}]
            )
            state.response = response
            
            result_state = await citation_node.execute(state)
            
            if should_match:
                assert len(result_state.citations) > 0, f"Should match: {citation_text}"
            else:
                assert len(result_state.citations) == 0, f"Should not match: {citation_text}"
    
    @pytest.mark.asyncio
    async def test_source_document_metadata_completeness(self, citation_node):
        """Test that source documents contain all required metadata"""
        
        response = "Information from [Document 1] and [Document 2]."
        
        documents = [
            {
                "id": "doc1",
                "title": "Document One",
                "content": "Content of document one...",
                "source_type": "policy",
                "relevance_score": 0.95,
                "metadata": {
                    "page_number": 1,
                    "section": "requirements",
                    "last_updated": "2024-01-15"
                }
            },
            {
                "id": "doc2", 
                "title": "Document Two",
                "content": "Content of document two...",
                "source_type": "guideline",
                "relevance_score": 0.87
            }
        ]
        
        state = create_test_state(user_query="Test", raw_documents=documents)
        state.response = response
        
        result_state = await citation_node.execute(state)
        
        # Validate source document structure
        assert len(result_state.source_documents) == 2
        
        for source_doc in result_state.source_documents:
            # Required fields
            assert "document_number" in source_doc
            assert "title" in source_doc
            assert "source_type" in source_doc
            
            # Optional fields should be preserved if present
            if "relevance_score" in source_doc:
                assert isinstance(source_doc["relevance_score"], (int, float))
            
            if "content_preview" in source_doc:
                assert isinstance(source_doc["content_preview"], str)
    
    @pytest.mark.asyncio
    async def test_state_preservation(self, citation_node):
        """Test that original state fields are preserved during citation processing"""
        
        original_state = create_test_state(
            user_query="test query",
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # Add existing state data
        original_state.query_type = "simple"
        original_state.intent = "test intent"
        original_state.entities = ["test", "entity"]
        original_state.response = "Test response with [Document 1] citation."
        original_state.raw_documents = [
            {"id": "doc1", "title": "Test Doc", "content": "Test content", "source_type": "test"}
        ]
        
        result_state = await citation_node.execute(original_state)
        
        # Original fields should be preserved
        assert result_state.user_query == original_state.user_query
        assert result_state.conversation_id == original_state.conversation_id
        assert result_state.user_id == original_state.user_id
        assert result_state.query_type == original_state.query_type
        assert result_state.intent == original_state.intent
        assert result_state.entities == original_state.entities
        assert result_state.response == original_state.response
        
        # New fields should be populated
        assert len(result_state.citations) > 0
        assert len(result_state.source_documents) > 0


@pytest.mark.integration
class TestCitationIntegration:
    """Integration tests for CitationNode with realistic scenarios"""
    
    @pytest.fixture
    def integration_node(self):
        """Create node for integration testing"""
        return CitationNode()
    
    @pytest.mark.asyncio
    async def test_immigration_response_citation_extraction(self, integration_node):
        """Test citation extraction from realistic immigration responses"""
        
        immigration_response = """Based on the official documentation, here are the requirements for Canadian work permits [Document 1]:

**Required Documents:**
- Valid passport with at least 6 months validity [Document 2]
- Job offer letter from a Canadian employer [Document 1]
- Labour Market Impact Assessment (LMIA) if required [Document 3]
- Educational credential assessment [Document 4]

**Processing Information:**
Processing times vary by country of residence [Document 2], typically ranging from 2-12 weeks. Medical examinations may be required [Document 5] depending on your country of origin and the duration of your intended stay.

**Additional Requirements:**
Proof of funds to support yourself and any family members [Document 1] is also required. Police certificates from countries where you've lived for 6+ months [Document 5] may be necessary."""
        
        documents = [
            {"id": "work_permit_guide", "title": "Work Permit Application Guide", "content": "...", "source_type": "official_guide"},
            {"id": "processing_times", "title": "Current Processing Times", "content": "...", "source_type": "current_data"},
            {"id": "lmia_guide", "title": "LMIA Requirements", "content": "...", "source_type": "policy"},
            {"id": "credential_assessment", "title": "Educational Credential Assessment", "content": "...", "source_type": "requirements"},
            {"id": "medical_police", "title": "Medical and Police Certificate Requirements", "content": "...", "source_type": "requirements"}
        ]
        
        state = create_test_state(
            user_query="What documents do I need for a work permit?",
            raw_documents=documents
        )
        state.response = immigration_response
        
        result = await integration_node.execute(state)
        
        # Should extract multiple citations
        assert len(result.citations) > 5
        
        # Should create source documents for all cited documents
        assert len(result.source_documents) == 5
        
        # Validate citation and source document consistency
        cited_numbers = {c["document_number"] for c in result.citations}
        source_numbers = {d["document_number"] for d in result.source_documents}
        assert cited_numbers == source_numbers
        
        # Check that all document numbers are within valid range
        for citation in result.citations:
            assert 1 <= citation["document_number"] <= 5
    
    @pytest.mark.asyncio
    async def test_complex_citation_patterns(self, integration_node):
        """Test extraction of citations in complex text patterns"""
        
        complex_response = """The requirements [Document 1] include several components:

1. Primary documents [Document 2]:
   - Passport [Document 1]
   - Application forms [Document 3]

2. Supporting evidence [Document 4]:
   - Financial proof [Document 2]
   - Medical results [Document 5]

Note: Some requirements [Document 1] may vary by country [Document 6]."""
        
        documents = [
            {"id": f"doc_{i}", "title": f"Document {i}", "content": "...", "source_type": "test"}
            for i in range(1, 7)
        ]
        
        state = create_test_state(user_query="Test", raw_documents=documents)
        state.response = complex_response
        
        result = await integration_node.execute(state)
        
        # Should handle complex nested citations
        assert len(result.citations) > 0
        assert len(result.source_documents) > 0
        
        # All citations should be valid
        for citation in result.citations:
            assert re.match(r"\[Document \d+\]", citation["citation_text"])
            assert 1 <= citation["document_number"] <= 6
    
    @pytest.mark.asyncio
    async def test_citation_performance_with_large_response(self, integration_node):
        """Test citation extraction performance with large responses"""
        
        # Create a large response with many citations
        large_response_parts = []
        for i in range(1, 51):  # 50 citations
            large_response_parts.append(f"Information from document {i} [Document {i}] is important.")
        
        large_response = " ".join(large_response_parts)
        
        # Create corresponding documents
        documents = [
            {"id": f"doc_{i}", "title": f"Document {i}", "content": f"Content {i}", "source_type": "test"}
            for i in range(1, 51)
        ]
        
        state = create_test_state(user_query="Test", raw_documents=documents)
        state.response = large_response
        
        import time
        start_time = time.time()
        
        result = await integration_node.execute(state)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time
        assert processing_time < 1.0  # Should be fast for citation extraction
        
        # Should extract all citations
        assert len(result.citations) == 50
        assert len(result.source_documents) == 50
    
    @pytest.mark.asyncio
    async def test_citation_edge_cases_integration(self, integration_node):
        """Test citation extraction with various edge cases"""
        
        edge_case_responses = [
            # Multiple citations in same sentence
            "Documents [Document 1] and forms [Document 2] and certificates [Document 3] are required.",
            
            # Citations at beginning and end
            "[Document 1] Requirements include various documents and forms [Document 2].",
            
            # Citations in parentheses and quotes
            "The guide (see [Document 1]) mentions that \"requirements [Document 2]\" vary.",
            
            # Citations with punctuation
            "Requirements: [Document 1], [Document 2]; and [Document 3]!",
            
            # Mixed valid and invalid citations
            "Valid [Document 1] and invalid [Document] and [Doc 2] citations."
        ]
        
        for i, response in enumerate(edge_case_responses):
            documents = [
                {"id": f"doc_{j}", "title": f"Doc {j}", "content": "...", "source_type": "test"}
                for j in range(1, 4)
            ]
            
            state = create_test_state(user_query=f"Test {i}", raw_documents=documents)
            state.response = response
            
            result = await integration_node.execute(state)
            
            # Should handle all edge cases without errors
            assert isinstance(result.citations, list)
            assert isinstance(result.source_documents, list)
            
            # All extracted citations should be valid
            for citation in result.citations:
                assert re.match(r"\[Document \d+\]", citation["citation_text"])
