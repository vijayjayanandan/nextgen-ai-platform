# Claude Real API Integration Test Usage Guide

## Overview

This guide explains how to use the comprehensive Claude API integration test suite to validate your RAG platform's real-world functionality.

## Files Created

1. **`CLAUDE_REAL_API_TEST_PLAN.md`** - Detailed test plan with 5 phases of testing
2. **`run_claude_real_api_tests.py`** - Automated test runner implementing Phase 1
3. **`CLAUDE_REAL_API_TEST_USAGE_GUIDE.md`** - This usage guide

## Prerequisites

### 1. Environment Setup
```bash
# Ensure you have a valid Claude API key
export ANTHROPIC_API_KEY=your_claude_api_key_here

# Make sure your FastAPI server is running
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2. Test Documents
Ensure you have test documents in the `test_documents/` directory:
- `citizenship_guide.txt`
- `eligibility_criteria.md` 
- `refund_policy.html`

If these don't exist, create simple test files:
```bash
mkdir -p test_documents

echo "Canadian citizenship requires meeting residency requirements and passing tests." > test_documents/citizenship_guide.txt

echo "# Eligibility Criteria\n\nApplicants must meet age and residency requirements." > test_documents/eligibility_criteria.md

echo "<html><body><h1>Refund Policy</h1><p>Full refunds available within 30 days.</p></body></html>" > test_documents/refund_policy.html
```

## Running the Tests

### Basic Usage
```bash
# Run with environment variable API key
python run_claude_real_api_tests.py

# Run with explicit API key
python run_claude_real_api_tests.py --api-key your_claude_api_key_here

# Run against different server
python run_claude_real_api_tests.py --base-url http://localhost:3000
```

### Test Phases

The automated test runner implements **Phase 1** of the comprehensive test plan:

#### Phase 1: Document Processing & RAG Testing
1. **Server Health Check** - Verifies FastAPI server is responding
2. **Document Upload** - Uploads test documents via API
3. **Processing Verification** - Waits for document processing completion
4. **Chunk Validation** - Verifies documents were properly chunked
5. **RAG Query Testing** - Tests Claude API integration with document context
6. **Streaming Response** - Tests real-time streaming functionality
7. **Edge Case Testing** - Tests fallback behavior
8. **Cleanup** - Removes test documents

## Understanding Test Results

### Success Indicators
- ✅ **Server Health**: FastAPI server responding
- ✅ **Document Upload**: Files uploaded successfully
- ✅ **Processing Complete**: Documents processed without errors
- ✅ **Chunks Generated**: Content properly extracted and chunked
- ✅ **RAG Responses**: Claude provides contextually relevant answers
- ✅ **Streaming Works**: Real-time response streaming functional

### Common Failure Points

#### Document Processing Issues
```
❌ Document Processing: PDF
Error: Processing timeout or failure
```
**Solution**: Check the document processing bug identified in the test plan. The path resolution issue between API storage and DocumentProcessor needs to be fixed.

#### Claude API Issues
```
❌ RAG Query: What are the citizenship requirements?
Error: RAG query failed with status 401
```
**Solution**: Verify your `ANTHROPIC_API_KEY` is valid and has sufficient credits.

#### Server Connection Issues
```
❌ Cannot connect to server: Connection refused
```
**Solution**: Ensure FastAPI server is running on the specified port.

## Test Report

After completion, the test generates:
- **Console Output**: Real-time test progress and results
- **JSON Report**: Detailed test report saved as `claude_real_api_test_report_YYYYMMDD_HHMMSS.json`

### Sample Report Structure
```json
{
  "summary": {
    "total_tests": 12,
    "passed": 10,
    "failed": 2,
    "success_rate": "83.3%"
  },
  "test_results": [
    {
      "test_name": "Upload TXT Document",
      "success": true,
      "timestamp": "2024-01-15T10:30:00",
      "details": {
        "document_id": "uuid-here",
        "file_name": "citizenship_guide.txt"
      }
    }
  ]
}
```

## Next Steps: Advanced Testing

The current script implements Phase 1. For comprehensive testing, implement additional phases:

### Phase 2: Performance Testing
- Load testing with multiple concurrent requests
- Large document handling
- Memory usage monitoring

### Phase 3: Error Recovery Testing
- Network interruption simulation
- Claude API rate limiting scenarios
- Database connection failures

### Phase 4: Multi-Document Context Testing
- Cross-document reasoning
- Citation accuracy
- Context window management

### Phase 5: Production Readiness
- Security testing
- Monitoring and alerting validation
- Backup and recovery procedures

## Troubleshooting

### Document Processing Bug
**Issue**: Documents upload but fail to process due to path resolution issues.

**Temporary Workaround**: The test plan identifies this as a critical bug where the API stores absolute paths but DocumentProcessor expects relative paths.

**Permanent Fix**: Update the document storage logic to use consistent path formats.

### Claude API Rate Limits
**Issue**: Tests fail due to rate limiting.

**Solution**: 
- Implement exponential backoff in test script
- Use Claude 3 Haiku for faster, cheaper testing
- Space out test requests

### Memory Issues
**Issue**: Tests fail with out-of-memory errors.

**Solution**:
- Test with smaller documents initially
- Monitor memory usage during tests
- Implement proper cleanup between tests

## Integration with CI/CD

To integrate these tests into your CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
name: Claude API Integration Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Start services
        run: |
          docker-compose up -d
          sleep 30  # Wait for services to start
      - name: Run Claude API tests
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: python run_claude_real_api_tests.py
```

## Best Practices

1. **Test Environment**: Always test against a dedicated test environment, not production
2. **API Key Management**: Use environment variables or secure secret management
3. **Test Data**: Use consistent, known test documents for reproducible results
4. **Monitoring**: Monitor Claude API usage and costs during testing
5. **Documentation**: Keep test results and update test cases as the system evolves

## Support

If you encounter issues:
1. Check the detailed test plan in `CLAUDE_REAL_API_TEST_PLAN.md`
2. Review the test report JSON for specific error details
3. Verify all prerequisites are met
4. Check server logs for additional context

The test suite provides a solid foundation for validating your Claude API integration and ensuring production readiness.
