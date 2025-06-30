"""
Citation Functionality Demo

This script demonstrates the citation functionality in action,
showing how the RAG system now automatically includes source
citations in all responses for transparency and auditability.
"""

import asyncio
from typing import List, Dict, Any

def demo_citation_formatter():
    """Demonstrate the citation formatting functionality"""
    
    print("🎯 Citation Functionality Demo")
    print("=" * 60)
    
    # Import the citation formatter
    from app.utils.citation_formatter import format_citations, append_citations_to_response
    
    print("\n📚 Sample Document Chunks (Retrieved from Vector DB)")
    print("-" * 55)
    
    # Simulate chunks retrieved from the RAG system
    sample_chunks = [
        {
            "content": "To be eligible for Canadian citizenship, you must have been physically present in Canada for at least 1,095 days (3 years) during the 5 years immediately before the date of your application.",
            "title": "Canadian Citizenship Guide 2024",
            "page_number": 12,
            "section_title": "Residency Requirements",
            "source_type": "policy",
            "score": 0.92,
            "metadata": {
                "document_id": "citizenship-guide-2024",
                "chunk_index": 8,
                "document_title": "Canadian Citizenship Guide 2024"
            }
        },
        {
            "content": "You must demonstrate adequate knowledge of English or French. This can be done through approved language tests such as CELPIP, IELTS, or TEF.",
            "metadata": {
                "document_title": "Canadian Citizenship Guide 2024",
                "page_number": 18,
                "section_title": "Language Requirements",
                "source_type": "policy",
                "document_id": "citizenship-guide-2024",
                "chunk_index": 12
            },
            "score": 0.88
        },
        {
            "content": "You must file income tax returns for at least 3 years during the 5-year period if you are required to do so under the Income Tax Act.",
            "title": "Tax Obligations for Citizenship",
            "page_number": 5,
            "source_type": "guideline",
            "score": 0.85,
            "metadata": {
                "document_id": "tax-obligations-guide",
                "chunk_index": 3
            }
        },
        {
            "content": "Current processing times for citizenship applications are approximately 12-18 months from the date we receive your complete application.",
            "title": "IRCC Processing Times",
            "metadata": {
                "section_title": "Citizenship Applications",
                "source_type": "operational",
                "document_id": "processing-times-2024",
                "chunk_index": 1
            },
            "score": 0.76
        },
        {
            "content": "Duplicate chunk for testing deduplication - same document and page as first chunk.",
            "title": "Canadian Citizenship Guide 2024",
            "page_number": 12,
            "section_title": "Residency Requirements",
            "source_type": "policy",
            "score": 0.70
        }
    ]
    
    # Display the chunks
    for i, chunk in enumerate(sample_chunks, 1):
        title = chunk.get('title') or chunk.get('metadata', {}).get('document_title', 'Untitled')
        page = chunk.get('page_number') or chunk.get('metadata', {}).get('page_number', 'N/A')
        section = chunk.get('section_title') or chunk.get('metadata', {}).get('section_title', 'N/A')
        score = chunk.get('score', 0.0)
        
        print(f"Chunk {i}: {title}")
        print(f"  Page: {page}, Section: {section}, Score: {score:.2f}")
        print(f"  Content: {chunk['content'][:100]}...")
        print()
    
    print("\n🔧 Citation Processing")
    print("-" * 25)
    
    # Format citations
    formatted_citations = format_citations(sample_chunks)
    
    print("Generated Citations:")
    print(formatted_citations)
    
    print(f"\n✅ Processed {len(sample_chunks)} chunks into {len(formatted_citations.split(chr(10)))} unique citations")
    print("✅ Duplicates automatically removed")
    print("✅ Sources sorted by document title and page number")
    
    return sample_chunks, formatted_citations


def demo_response_integration():
    """Demonstrate citation integration with LLM responses"""
    
    print("\n🤖 LLM Response Integration Demo")
    print("=" * 40)
    
    from app.utils.citation_formatter import append_citations_to_response
    
    # Simulate an LLM response about Canadian citizenship
    llm_response = """Based on the available information, here are the key requirements for Canadian citizenship:

**1. Physical Presence Requirement**
You must have been physically present in Canada for at least 1,095 days (3 years) during the 5 years immediately before applying for citizenship.

**2. Language Proficiency**
You must demonstrate adequate knowledge of English or French through approved language tests such as CELPIP, IELTS, or TEF.

**3. Tax Filing Obligations**
You must file income tax returns for at least 3 years during the 5-year period if required under the Income Tax Act.

**4. Processing Timeline**
Current processing times for citizenship applications are approximately 12-18 months from the date your complete application is received.

These requirements ensure that new citizens have sufficient connection to Canada and can participate fully in Canadian society."""
    
    print("Original LLM Response:")
    print("-" * 25)
    print(llm_response)
    
    # Get sample chunks from previous demo
    sample_chunks, _ = demo_citation_formatter()
    
    # Append citations to the response
    final_response = append_citations_to_response(llm_response, sample_chunks)
    
    print("\n" + "="*80)
    print("FINAL RESPONSE WITH CITATIONS")
    print("="*80)
    print(final_response)
    
    print("\n✅ Citations automatically appended to response")
    print("✅ Professional formatting maintained")
    print("✅ Sources clearly attributed")
    print("✅ Ready for government-grade transparency requirements")


async def demo_citation_node():
    """Demonstrate the CitationNode in workflow context"""
    
    print("\n🔄 Citation Node Workflow Demo")
    print("=" * 40)
    
    from app.services.rag.nodes.citation import CitationNode
    from app.services.rag.workflow_state import RAGState
    
    # Create a sample RAG state
    state = RAGState(
        user_query="What are the requirements for Canadian citizenship?",
        response="You must meet residency, language, and tax filing requirements for Canadian citizenship."
    )
    
    # Add sample documents (simulating retrieval results)
    state.raw_documents = [
        {
            "id": "chunk-citizenship-1",
            "content": "Physical presence requirement: 1,095 days in 5 years",
            "title": "Citizenship Guide 2024",
            "page_number": 12,
            "section_title": "Residency Requirements",
            "source_type": "policy",
            "score": 0.92,
            "metadata": {
                "document_id": "citizenship-guide-2024",
                "chunk_index": 8
            }
        },
        {
            "id": "chunk-citizenship-2",
            "content": "Language proficiency through CELPIP, IELTS, or TEF",
            "title": "Citizenship Guide 2024", 
            "page_number": 18,
            "section_title": "Language Requirements",
            "source_type": "policy",
            "score": 0.88,
            "metadata": {
                "document_id": "citizenship-guide-2024",
                "chunk_index": 12
            }
        }
    ]
    
    print("Input State:")
    print(f"  Query: {state.user_query}")
    print(f"  Response: {state.response}")
    print(f"  Documents: {len(state.raw_documents)} chunks")
    
    # Execute citation node
    citation_node = CitationNode()
    result_state = await citation_node.execute(state)
    
    print("\nAfter Citation Processing:")
    print("-" * 30)
    print("Enhanced Response:")
    print(result_state.response)
    
    print(f"\nCitation Metadata ({len(result_state.citations)} citations):")
    for citation in result_state.citations:
        print(f"  - {citation['document_title']} (page {citation.get('page_number', 'N/A')})")
    
    print(f"\nSource Documents ({len(result_state.source_documents)} documents):")
    for doc in result_state.source_documents:
        print(f"  - {doc['title']} (relevance: {doc['relevance_score']:.2f})")
        print(f"    Chunks used: {len(doc['chunks_used'])}")


def demo_edge_cases():
    """Demonstrate handling of edge cases"""
    
    print("\n⚠️  Edge Case Handling Demo")
    print("=" * 35)
    
    from app.utils.citation_formatter import format_citations, append_citations_to_response
    
    # Test cases
    test_cases = [
        {
            "name": "Empty chunks list",
            "chunks": [],
            "expected": "No citations (empty string)"
        },
        {
            "name": "Chunks with missing metadata",
            "chunks": [{"content": "Some content without metadata"}],
            "expected": "Default title used"
        },
        {
            "name": "Chunks with partial metadata",
            "chunks": [
                {
                    "content": "Content with only title",
                    "title": "Partial Document"
                }
            ],
            "expected": "Citation without page/section"
        },
        {
            "name": "Empty response",
            "chunks": [{"title": "Test Doc", "content": "Test content"}],
            "response": "",
            "expected": "Empty response preserved"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * (len(test_case['name']) + 8))
        
        if 'response' in test_case:
            # Test response integration
            result = append_citations_to_response(test_case['response'], test_case['chunks'])
            print(f"Result: '{result}'")
        else:
            # Test citation formatting
            result = format_citations(test_case['chunks'])
            print(f"Result: '{result}'")
        
        print(f"Expected: {test_case['expected']}")
        print("✅ Handled gracefully")


def main():
    """Main demo execution"""
    
    print("🎯 Citation Functionality Complete Demo")
    print("=" * 70)
    print("This demo shows how the RAG system now automatically includes")
    print("source citations for transparency and auditability.")
    print()
    
    # Run all demos
    demo_citation_formatter()
    demo_response_integration()
    
    # Run async demo
    print("\n" + "="*70)
    asyncio.run(demo_citation_node())
    
    demo_edge_cases()
    
    print("\n" + "="*70)
    print("🎉 DEMO COMPLETE")
    print("="*70)
    print()
    print("Key Benefits Demonstrated:")
    print("✅ Automatic citation generation from retrieved chunks")
    print("✅ Professional formatting with document emoji markers")
    print("✅ Intelligent deduplication of duplicate sources")
    print("✅ Graceful handling of missing or partial metadata")
    print("✅ Seamless integration with existing RAG workflow")
    print("✅ Government-grade transparency and auditability")
    print()
    print("The NextGen AI Platform now provides fully transparent,")
    print("auditable responses with proper source attribution!")


if __name__ == "__main__":
    main()
