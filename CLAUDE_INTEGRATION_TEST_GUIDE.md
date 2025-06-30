# Claude API Integration Test Guide

## Overview

This comprehensive test suite validates the production-grade integration of Claude API with our LangGraph-based RAG platform. It performs end-to-end testing with real Claude API calls to ensure the system is ready for production deployment.

## Test Suite Components

### 1. Main Test File: `tests/test_real_claude_api_integration.py`

Production-grade pytest test suite that validates:
- **End-to-end RAG behavior** with real Claude API calls
- **Intermediate RAG node validation** through metadata analysis
- **Document grounding accuracy** with unique content verification
- **Performance and quality metrics** under various conditions

### 2. Test Runner: `run_claude_integration_tests.py`

Automated test execution script that:
- Checks prerequisites and configuration
- Runs the complete test suite
- Analyzes results and generates reports
- Provides actionable recommendations

## Test Phases

### Phase 1: Document Upload and Processing
**Objective**: Validate document ingestion pipeline

**Tests**:
- Upload test document with unique markers
- Verify processing status transitions (PENDING → PROCESSING → PROCESSED)
- Validate chunk creation and embedding generation
- Confirm grounding keywords are preserved in chunks

**Success Criteria**:
- Document processes successfully within 120 seconds
- At least 5 grounding keywords found in chunks
- Document status reaches "processed"

### Phase 2: RAG Query Processing

#### Simple RAG with Grounding Validation
**Objective**: Test basic RAG functionality with document grounding

**Tests**:
- Query: "What are the specific requirements for Canadian citizenship application?"
- Validate response contains unique markers from uploaded document
- Verify intermediate node execution through metadata
- Confirm citations reference test document

**Success Criteria**:
- Response contains ≥2 grounding keywords
- No hallucination indicators present
- All RAG nodes execute successfully
- Citations reference uploaded document

#### Complex Multi-Document RAG
**Objective**: Test information synthesis across multiple sources

**Tests**:
- Upload additional comparison document
- Query: "Compare processing times and language requirements between Express Entry and citizenship"
- Validate reranking node execution
- Confirm multi-source information synthesis

**Success Criteria**:
- Documents retrieved ≥ documents used (reranking occurred)
- Information from both documents present in response
- Complex query type detected

#### Conversational RAG with Memory
**Objective**: Test memory-enabled contextual understanding

**Tests**:
- Create conversation about work permits
- Follow-up query: "How long does that application process typically take?"
- Validate memory retrieval and contextual understanding

**Success Criteria**:
- Memory turns used > 0
- Response demonstrates contextual understanding
- Conversation history maintained

### Phase 3: Streaming RAG Response
**Objective**: Validate real-time streaming with Claude API

**Tests**:
- Complex query requiring detailed explanation
- Validate SSE stream format and progression
- Confirm grounding in streaming content

**Success Criteria**:
- Chunks received > 0
- Response length > 200 characters
- ≥2 grounding keywords in streamed content

### Phase 4: Error Handling and Fallbacks
**Objective**: Test graceful degradation

**Tests**:
- Invalid/malformed queries
- Queries with no relevant documents
- System resilience validation

**Success Criteria**:
- Graceful handling of invalid queries
- Appropriate limitation acknowledgment
- No system crashes or errors

### Phase 5: Performance and Quality Metrics
**Objective**: Benchmark production readiness

**Tests**:
- Sequential query performance
- Concurrent request handling
- Quality assessment across multiple queries

**Success Criteria**:
- Average response time < 10 seconds
- Concurrent success rate > 80%
- Consistent quality scores

## Grounding Validation Strategy

### Unique Content Approach
The test suite uses documents with unique, verifiable phrases:

```
UNIQUE_MARKER_12345
UNIQUE_TERM_FLUENCY
UNIQUE_NUMBER_1095
UNIQUE_TIMEFRAME_18_MONTHS
UNIQUE_CONDITION_HUMANITARIAN_GROUNDS
```

### Validation Logic
- **Grounding Check**: Response must contain ≥2 unique keywords
- **Hallucination Detection**: Response must not contain contradictory information
- **Citation Validation**: Citations must reference uploaded documents

## Node Validation Through Metadata

### Query Analysis Node
- Validates: `query_type`, `intent`, `entities` in metadata
- Confirms: Proper query classification and intent detection

### Retrieval Node
- Validates: `documents_retrieved` > 0
- Confirms: Successful document retrieval from vector store

### Reranking Node
- Validates: `documents_retrieved` ≥ `documents_used`
- Confirms: Document filtering and relevance ranking

### Generation Node
- Validates: `model_used`, `processing_time` > 0
- Confirms: Claude API integration and response generation

### Memory Node
- Validates: `memory_turns_used` > 0 for conversational queries
- Confirms: Conversation history retrieval and context maintenance

## Prerequisites

### Environment Setup
1. **Claude API Key**: Configure `ANTHROPIC_API_KEY` in `.env` file
2. **Database**: Ensure PostgreSQL and vector store are running
3. **Dependencies**: Install required packages via `pip install -r requirements.txt`

### Configuration Validation
```bash
# Check configuration
python -c "from app.core.config import settings; print(f'Claude API: {bool(settings.ANTHROPIC_API_KEY)}')"
```

## Execution Methods

### Method 1: Automated Test Runner (Recommended)
```bash
python run_claude_integration_tests.py
```

**Features**:
- Prerequisite checking
- Automated execution
- Comprehensive reporting
- Actionable recommendations

### Method 2: Direct Pytest Execution
```bash
pytest tests/test_real_claude_api_integration.py -v --tb=short
```

**Features**:
- Detailed test output
- Individual test control
- Debug-friendly execution

### Method 3: Individual Test Execution
```bash
pytest tests/test_real_claude_api_integration.py::TestRealClaudeAPIIntegration::test_02_simple_rag_query_with_grounding -v
```

## Success Criteria

### Functional Requirements
- **100%** of RAG queries return responses with ≥2 grounding keywords
- **0%** hallucination rate on verifiable facts
- **All** intermediate node outputs present in metadata
- **All** configured Claude models respond successfully

### Performance Requirements
- **Average latency** < 10 seconds for complex queries
- **Concurrent success rate** > 80%
- **Document processing** < 120 seconds

### Quality Requirements
- **Grounding accuracy** ≥ 2 keywords per response
- **Citation accuracy** references uploaded documents
- **Contextual understanding** in conversational scenarios

## Generated Reports

### 1. Pytest JSON Report (`claude_test_report.json`)
- Test execution details
- Pass/fail status for each test
- Execution times and error details

### 2. Detailed Results (`claude_real_api_integration_results.json`)
- Grounding validation results
- Node execution analysis
- Performance metrics
- Quality assessments

### 3. Final Summary (`claude_integration_final_report.json`)
- Overall assessment (Excellent/Good/Fair/Poor)
- Success rate and key metrics
- Actionable recommendations

## Troubleshooting

### Common Issues

#### 1. Claude API Key Issues
```
❌ ANTHROPIC_API_KEY not configured in .env file
```
**Solution**: Add `ANTHROPIC_API_KEY=your_key_here` to `.env` file

#### 2. Document Processing Timeout
```
❌ Document processing timed out
```
**Solution**: Check vector store connectivity and embedding service

#### 3. Grounding Validation Failures
```
❌ Grounding validation failed: insufficient keywords
```
**Solution**: Review document chunking and retrieval configuration

#### 4. Node Validation Failures
```
❌ Node validation failed: missing metadata
```
**Solution**: Check RAG workflow configuration and logging

### Debug Mode
```bash
pytest tests/test_real_claude_api_integration.py -v -s --log-cli-level=DEBUG
```

## Production Deployment Checklist

### Before Deployment
- [ ] All tests pass with ≥90% success rate
- [ ] Average response time < 10 seconds
- [ ] Grounding validation shows ≥2 keywords per response
- [ ] No hallucination indicators detected
- [ ] Concurrent request handling validated

### Monitoring Setup
- [ ] Claude API usage tracking
- [ ] Response time monitoring
- [ ] Grounding accuracy metrics
- [ ] Error rate alerting

### Continuous Testing
- [ ] Automated test pipeline configured
- [ ] Regular grounding validation
- [ ] Performance regression testing

## Advanced Configuration

### Custom Grounding Keywords
Modify `GROUNDING_KEYWORDS` in test file for domain-specific validation:

```python
GROUNDING_KEYWORDS = [
    "YOUR_UNIQUE_MARKER_1",
    "YOUR_UNIQUE_MARKER_2",
    # Add domain-specific markers
]
```

### Performance Tuning
Adjust test parameters for your environment:

```python
# Document processing timeout
timeout=120  # Increase for slower systems

# Retrieval configuration
"retrieval_options": {"top_k": 5}  # Adjust based on needs

# Claude API parameters
"max_tokens": 1000,  # Adjust for response length
"temperature": 0.3   # Adjust for creativity vs consistency
```

## Support and Maintenance

### Regular Updates
- Update grounding keywords for new content
- Adjust performance thresholds based on production metrics
- Enhance validation logic for new RAG features

### Monitoring Integration
- Integrate with existing monitoring systems
- Set up alerts for test failures
- Track trends in grounding accuracy and performance

This test suite provides comprehensive validation of Claude API integration with your LangGraph RAG platform, ensuring production readiness through rigorous testing of functionality, performance, and quality metrics.
