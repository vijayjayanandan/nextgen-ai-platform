# Claude Real API Integration Test Plan

## Overview
This document outlines a comprehensive test plan for validating real Claude API integration in our LangGraph RAG platform. The tests focus on end-to-end system integration using actual Claude API endpoints rather than mocks.

## System Architecture Summary

### Key Components Identified:
1. **FastAPI Backend** (`app/main.py`)
2. **Claude Integration** (`app/services/llm/anthropic_service.py`)
3. **Document Processing** (`app/api/v1/endpoints/documents.py`)
4. **Chat/RAG Endpoints** (`app/api/v1/endpoints/chat.py`)
5. **Retrieval System** (`app/api/v1/endpoints/retrieval.py`)
6. **LangGraph Workflow** (`app/services/rag/workflow.py`)

### Core API Endpoints:
- `POST /api/v1/documents/` - Document upload and processing
- `POST /api/v1/chat/completions` - Chat completions (with RAG)
- `POST /api/v1/chat/stream` - Streaming chat completions
- `POST /api/v1/retrieval/semantic-search` - Document search
- `GET /api/v1/documents/{id}` - Document retrieval

## Test Plan Structure

### Phase 1: Document Upload and Processing Tests

#### Test 1.1: Document Upload Validation
**Objective**: Verify document upload and processing pipeline works end-to-end

**Test Steps**:
1. **Setup**: Ensure Claude API key is configured in environment
2. **Upload Test Documents**:
   ```bash
   # Upload PDF document
   curl -X POST "http://localhost:8000/api/v1/documents/" \
     -H "Authorization: Bearer $AUTH_TOKEN" \
     -F "file=@test_documents/immigration_policy.pdf" \
     -F "title=Immigration Policy Guide" \
     -F "is_public=true"
   
   # Upload text document
   curl -X POST "http://localhost:8000/api/v1/documents/" \
     -H "Authorization: Bearer $AUTH_TOKEN" \
     -F "file=@test_documents/citizenship_requirements.txt" \
     -F "title=Citizenship Requirements" \
     -F "is_public=true"
   ```

3. **Validation Criteria**:
   - Document upload returns 200 status
   - Document ID is generated
   - Background processing starts successfully
   - Document status transitions: PENDING → PROCESSING → PROCESSED
   - Document chunks are created and embedded
   - No errors in application logs

#### Test 1.2: Document Processing Status Monitoring
**Objective**: Verify document processing status tracking

**Test Steps**:
1. Monitor document processing status:
   ```bash
   curl -X GET "http://localhost:8000/api/v1/documents/{document_id}/status" \
     -H "Authorization: Bearer $AUTH_TOKEN"
   ```

2. **Validation Criteria**:
   - Status updates correctly (PENDING → PROCESSING → PROCESSED)
   - Chunk count is populated when processed
   - Error messages are captured if processing fails
   - Processing completes within reasonable time (< 5 minutes for typical documents)

### Phase 2: Semantic Search and Retrieval Tests

#### Test 2.1: Basic Semantic Search
**Objective**: Verify document retrieval using semantic search

**Test Steps**:
1. **Perform Semantic Search**:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/retrieval/semantic-search" \
     -H "Authorization: Bearer $AUTH_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What are the requirements for citizenship?",
       "top_k": 5,
       "include_content": true
     }'
   ```

2. **Validation Criteria**:
   - Returns relevant document chunks
   - Similarity scores are reasonable (> 0.3 for relevant content)
   - Document metadata is included
   - Content is properly extracted and readable
   - Response time < 2 seconds

#### Test 2.2: Contextual Relevance Testing
**Objective**: Verify retrieval quality and contextual grounding

**Test Steps**:
1. **Test Multiple Query Types**:
   ```json
   // Factual query
   {"query": "What documents are needed for immigration?", "top_k": 3}
   
   // Procedural query  
   {"query": "How do I apply for citizenship?", "top_k": 3}
   
   // Specific detail query
   {"query": "What is the minimum residency requirement?", "top_k": 3}
   ```

2. **Validation Criteria**:
   - Retrieved chunks contain relevant information for each query type
   - No irrelevant or off-topic chunks in top 3 results
   - Chunks from appropriate documents are prioritized
   - Semantic similarity reflects actual content relevance

### Phase 3: Claude API Integration Tests

#### Test 3.1: Basic Chat Completion with Claude
**Objective**: Verify direct Claude API integration without RAG

**Test Steps**:
1. **Simple Chat Completion**:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/chat/completions" \
     -H "Authorization: Bearer $AUTH_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "claude-3-5-sonnet-20241022",
       "messages": [
         {"role": "user", "content": "Hello, can you help me understand immigration policies?"}
       ],
       "max_tokens": 500,
       "temperature": 0.7
     }'
   ```

2. **Validation Criteria**:
   - Returns valid Claude API response
   - Response contains coherent, relevant content
   - Token usage is tracked correctly
   - Response time < 10 seconds
   - No API errors or rate limiting issues

#### Test 3.2: RAG-Enhanced Chat Completion
**Objective**: Verify Claude integration with document retrieval (core RAG functionality)

**Test Steps**:
1. **RAG-Enhanced Query**:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/chat/completions" \
     -H "Authorization: Bearer $AUTH_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "claude-3-5-sonnet-20241022",
       "messages": [
         {"role": "user", "content": "What are the specific requirements for obtaining citizenship?"}
       ],
       "retrieve": true,
       "retrieval_options": {
         "top_k": 5,
         "filters": null
       },
       "max_tokens": 1000,
       "temperature": 0.3
     }'
   ```

2. **Validation Criteria**:
   - Response demonstrates knowledge from uploaded documents
   - Answer includes specific details that could only come from document content
   - Response is more accurate/detailed than without RAG
   - Retrieved context is properly integrated into Claude's response
   - No hallucination of facts not present in documents

#### Test 3.3: Streaming Chat with RAG
**Objective**: Verify streaming responses work with Claude and RAG

**Test Steps**:
1. **Streaming RAG Query**:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/chat/stream" \
     -H "Authorization: Bearer $AUTH_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "claude-3-5-sonnet-20241022",
       "messages": [
         {"role": "user", "content": "Explain the immigration process step by step"}
       ],
       "retrieve": true,
       "stream": true,
       "max_tokens": 1500
     }'
   ```

2. **Validation Criteria**:
   - Receives Server-Sent Events (SSE) stream
   - Chunks arrive progressively and coherently
   - Final response demonstrates document knowledge
   - Stream completes without errors
   - Metadata includes retrieval information

### Phase 4: Error Handling and Edge Cases

#### Test 4.1: Claude API Error Handling
**Objective**: Verify graceful handling of Claude API errors

**Test Steps**:
1. **Test Invalid Model**:
   ```json
   {"model": "invalid-model-name", "messages": [...]}
   ```

2. **Test Rate Limiting** (if applicable):
   - Send multiple rapid requests to trigger rate limits

3. **Test Malformed Requests**:
   ```json
   {"model": "claude-3-5-sonnet-20241022", "messages": "invalid-format"}
   ```

4. **Validation Criteria**:
   - Appropriate HTTP error codes returned
   - Error messages are informative but don't expose sensitive details
   - System remains stable after errors
   - Retry logic works for transient failures

#### Test 4.2: RAG Fallback Behavior
**Objective**: Verify system behavior when retrieval fails or returns no results

**Test Steps**:
1. **Query with No Relevant Documents**:
   ```json
   {"query": "What is the weather like on Mars?", "retrieve": true}
   ```

2. **Query When Vector DB is Unavailable** (simulate by stopping vector DB)

3. **Validation Criteria**:
   - System gracefully falls back to Claude without context
   - User receives helpful response indicating limited context
   - No system crashes or unhandled exceptions
   - Appropriate logging of fallback scenarios

### Phase 5: Performance and Load Testing

#### Test 5.1: Concurrent Request Handling
**Objective**: Verify system handles multiple simultaneous Claude API calls

**Test Steps**:
1. **Concurrent Chat Requests**:
   ```bash
   # Run 10 concurrent requests
   for i in {1..10}; do
     curl -X POST "http://localhost:8000/api/v1/chat/completions" \
       -H "Authorization: Bearer $AUTH_TOKEN" \
       -H "Content-Type: application/json" \
       -d '{"model": "claude-3-5-sonnet-20241022", "messages": [{"role": "user", "content": "Test query '$i'"}]}' &
   done
   wait
   ```

2. **Validation Criteria**:
   - All requests complete successfully
   - Response times remain reasonable (< 15 seconds)
   - No request failures due to concurrency issues
   - Claude API rate limits are respected
   - System resources remain stable

#### Test 5.2: Large Document Processing
**Objective**: Verify handling of large documents with Claude integration

**Test Steps**:
1. Upload large document (>10MB PDF)
2. Verify processing completes successfully
3. Test RAG queries against large document content

**Validation Criteria**:
- Large documents are processed without timeout
- Chunking strategy handles large documents appropriately
- RAG queries work effectively across large document corpus
- Memory usage remains reasonable

### Phase 6: End-to-End User Scenarios

#### Test 6.1: Complete Immigration Consultation Workflow
**Objective**: Simulate realistic user interaction with immigration documents

**Test Scenario**:
1. **Setup**: Upload comprehensive immigration document set
2. **User Query Sequence**:
   ```
   Query 1: "What documents do I need to apply for permanent residency?"
   Query 2: "How long does the application process typically take?"
   Query 3: "What are the language requirements?"
   Query 4: "Can I work while my application is being processed?"
   ```

3. **Validation Criteria**:
   - Each response demonstrates specific knowledge from uploaded documents
   - Responses are contextually appropriate and helpful
   - Follow-up questions build on previous context appropriately
   - No contradictory information between responses

#### Test 6.2: Multi-Document Cross-Reference
**Objective**: Verify Claude can synthesize information across multiple documents

**Test Steps**:
1. Upload related but separate documents (e.g., policy document + FAQ + application guide)
2. Ask questions requiring information from multiple sources
3. **Example Query**: "Compare the requirements mentioned in the policy document with the FAQ answers"

**Validation Criteria**:
- Claude successfully references multiple documents
- Information is synthesized coherently
- Source attribution is maintained
- No conflicting information is presented

## Test Environment Setup

### Prerequisites:
1. **Environment Variables**:
   ```bash
   export ANTHROPIC_API_KEY="your-claude-api-key"
   export ANTHROPIC_API_BASE="https://api.anthropic.com"
   export ENABLE_FUNCTION_CALLING=true
   ```

2. **Test Documents**:
   - Immigration policy PDF (5-10 pages)
   - Citizenship requirements text file
   - FAQ document (HTML/Markdown)
   - Application guide (DOCX)

3. **Authentication**:
   - Valid user account with appropriate roles
   - API authentication tokens

### Test Data Validation:
- Ensure test documents contain factual, verifiable information
- Include edge cases (special characters, multiple languages if supported)
- Prepare "ground truth" answers for validation

## Success Criteria

### Functional Requirements:
- ✅ All document uploads process successfully
- ✅ Semantic search returns relevant results
- ✅ Claude API integration works without errors
- ✅ RAG responses demonstrate document knowledge
- ✅ Streaming functionality works correctly

### Performance Requirements:
- ✅ Document processing: < 5 minutes for typical documents
- ✅ Semantic search: < 2 seconds response time
- ✅ Chat completions: < 10 seconds response time
- ✅ Concurrent requests: Handle 10+ simultaneous users

### Quality Requirements:
- ✅ RAG responses are factually accurate based on source documents
- ✅ No hallucination of information not present in documents
- ✅ Appropriate fallback behavior when retrieval fails
- ✅ Error handling is graceful and informative

## Test Execution Schedule

### Phase 1-2: Foundation (Day 1)
- Document upload and processing tests
- Basic retrieval functionality

### Phase 3: Core Integration (Day 2)
- Claude API integration tests
- RAG functionality validation

### Phase 4-5: Robustness (Day 3)
- Error handling and edge cases
- Performance and load testing

### Phase 6: User Scenarios (Day 4)
- End-to-end workflow testing
- User acceptance validation

## Monitoring and Logging

### Key Metrics to Track:
- Claude API response times and error rates
- Document processing success/failure rates
- Retrieval accuracy and relevance scores
- System resource utilization
- User query patterns and satisfaction

### Log Analysis:
- Monitor application logs for Claude API errors
- Track document processing pipeline issues
- Analyze retrieval quality metrics
- Review user interaction patterns

## Risk Mitigation

### Potential Issues:
1. **Claude API Rate Limits**: Implement proper rate limiting and retry logic
2. **Document Processing Failures**: Ensure robust error handling and user feedback
3. **Retrieval Quality**: Validate embedding model performance and chunking strategy
4. **Performance Degradation**: Monitor system resources and optimize as needed

### Contingency Plans:
- Fallback to cached responses for common queries
- Manual document processing for critical failures
- Alternative embedding models if primary fails
- Graceful degradation when Claude API is unavailable

This test plan ensures comprehensive validation of the Claude API integration within the RAG platform, focusing on real-world usage scenarios and system reliability.
