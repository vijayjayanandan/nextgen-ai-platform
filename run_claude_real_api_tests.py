#!/usr/bin/env python3
"""
Claude Real API Integration Test Runner
Implements Phase 1 of the comprehensive test plan for document processing and RAG functionality.
"""

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
import aiohttp
from datetime import datetime

# Add the app directory to the Python path
sys.path.append('.')

class ClaudeRealAPITester:
    """Test runner for Claude API integration with the RAG platform."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.test_results = []
        self.uploaded_documents = []
        
        # Test document configurations
        self.test_documents = [
            {
                'name': 'citizenship_guide.txt',
                'path': 'test_documents/citizenship_guide.txt',
                'type': 'TXT',
                'expected_content': 'citizenship',
                'test_query': 'What are the citizenship requirements?'
            },
            {
                'name': 'eligibility_criteria.md',
                'path': 'test_documents/eligibility_criteria.md',
                'type': 'Markdown',
                'expected_content': 'eligibility',
                'test_query': 'What are the eligibility criteria?'
            },
            {
                'name': 'refund_policy.html',
                'path': 'test_documents/refund_policy.html',
                'type': 'HTML',
                'expected_content': 'refund',
                'test_query': 'What is the refund policy?'
            }
        ]
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def record_result(self, test_name: str, success: bool, details: Dict[str, Any]):
        """Record a test result."""
        result = {
            'test_name': test_name,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        self.log(f"{status}: {test_name}")
        
        if not success and 'error' in details:
            self.log(f"Error: {details['error']}", "ERROR")
    
    async def check_server_health(self) -> bool:
        """Check if the FastAPI server is running."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                self.log("✅ Server is healthy and responding")
                return True
            else:
                self.log(f"❌ Server health check failed: {response.status_code}")
                return False
        except Exception as e:
            self.log(f"❌ Cannot connect to server: {str(e)}")
            return False
    
    async def upload_document(self, doc_config: Dict[str, str]) -> Optional[str]:
        """Upload a document and return its ID if successful."""
        try:
            file_path = doc_config['path']
            if not os.path.exists(file_path):
                self.log(f"❌ Test document not found: {file_path}")
                return None
            
            # Prepare the file upload
            with open(file_path, 'rb') as f:
                files = {'file': (doc_config['name'], f, 'application/octet-stream')}
                data = {
                    'title': doc_config['name'],
                    'description': f"Test document for Claude API integration - {doc_config['type']}",
                    'is_public': True
                }
                
                response = requests.post(
                    f"{self.base_url}/api/v1/documents/",
                    files=files,
                    data=data,
                    timeout=30
                )
            
            if response.status_code == 200:
                result = response.json()
                doc_id = result.get('id')
                self.log(f"✅ Document uploaded successfully: {doc_id}")
                
                # Store document info for cleanup
                self.uploaded_documents.append({
                    'id': doc_id,
                    'name': doc_config['name'],
                    'type': doc_config['type']
                })
                
                self.record_result(
                    f"Upload {doc_config['type']} Document",
                    True,
                    {
                        'document_id': doc_id,
                        'file_name': doc_config['name'],
                        'response': result
                    }
                )
                return doc_id
            else:
                error_msg = f"Upload failed with status {response.status_code}: {response.text}"
                self.record_result(
                    f"Upload {doc_config['type']} Document",
                    False,
                    {'error': error_msg}
                )
                return None
                
        except Exception as e:
            self.record_result(
                f"Upload {doc_config['type']} Document",
                False,
                {'error': str(e)}
            )
            return None
    
    async def wait_for_processing(self, doc_id: str, timeout: int = 60) -> bool:
        """Wait for document processing to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/api/v1/documents/{doc_id}/status")
                if response.status_code == 200:
                    status_data = response.json()
                    status = status_data.get('status')
                    
                    if status == 'PROCESSED':
                        self.log(f"✅ Document {doc_id} processing completed")
                        return True
                    elif status == 'FAILED':
                        error_msg = status_data.get('error_message', 'Unknown error')
                        self.log(f"❌ Document {doc_id} processing failed: {error_msg}")
                        return False
                    else:
                        self.log(f"⏳ Document {doc_id} status: {status}")
                        await asyncio.sleep(2)
                else:
                    self.log(f"❌ Failed to check status for {doc_id}: {response.status_code}")
                    return False
            except Exception as e:
                self.log(f"❌ Error checking status for {doc_id}: {str(e)}")
                return False
        
        self.log(f"❌ Timeout waiting for document {doc_id} to process")
        return False
    
    async def verify_document_chunks(self, doc_id: str, expected_content: str) -> bool:
        """Verify that document was properly chunked and contains expected content."""
        try:
            response = requests.get(f"{self.base_url}/api/v1/documents/{doc_id}?include_chunks=true")
            if response.status_code == 200:
                doc_data = response.json()
                chunks = doc_data.get('chunks', [])
                
                if not chunks:
                    self.log(f"❌ No chunks found for document {doc_id}")
                    return False
                
                # Check if expected content appears in chunks
                content_found = False
                for chunk in chunks:
                    if expected_content.lower() in chunk.get('content', '').lower():
                        content_found = True
                        break
                
                if content_found:
                    self.log(f"✅ Document {doc_id} properly chunked with {len(chunks)} chunks")
                    return True
                else:
                    self.log(f"❌ Expected content '{expected_content}' not found in chunks")
                    return False
            else:
                self.log(f"❌ Failed to retrieve document {doc_id}: {response.status_code}")
                return False
        except Exception as e:
            self.log(f"❌ Error verifying chunks for {doc_id}: {str(e)}")
            return False
    
    async def test_rag_query(self, query: str, expected_content: str) -> bool:
        """Test a RAG query using Claude API."""
        try:
            payload = {
                "messages": [{"role": "user", "content": query}],
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1000
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/chat/completions",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract the response content
                choices = result.get('choices', [])
                if choices:
                    message = choices[0].get('message', {})
                    content = message.get('content', '')
                    
                    # Check if response contains expected content or references documents
                    has_relevant_content = (
                        expected_content.lower() in content.lower() or
                        'document' in content.lower() or
                        'based on' in content.lower()
                    )
                    
                    if has_relevant_content:
                        self.log(f"✅ RAG query successful with relevant response")
                        self.record_result(
                            f"RAG Query: {query[:50]}...",
                            True,
                            {
                                'query': query,
                                'response_length': len(content),
                                'response_preview': content[:200] + "..." if len(content) > 200 else content
                            }
                        )
                        return True
                    else:
                        self.log(f"❌ RAG response doesn't seem relevant to uploaded documents")
                        self.record_result(
                            f"RAG Query: {query[:50]}...",
                            False,
                            {
                                'error': 'Response not relevant to documents',
                                'response': content[:200] + "..." if len(content) > 200 else content
                            }
                        )
                        return False
                else:
                    self.log(f"❌ No choices in RAG response")
                    return False
            else:
                error_msg = f"RAG query failed with status {response.status_code}: {response.text}"
                self.log(f"❌ {error_msg}")
                self.record_result(
                    f"RAG Query: {query[:50]}...",
                    False,
                    {'error': error_msg}
                )
                return False
                
        except Exception as e:
            self.record_result(
                f"RAG Query: {query[:50]}...",
                False,
                {'error': str(e)}
            )
            return False
    
    async def test_streaming_response(self, query: str) -> bool:
        """Test streaming RAG response."""
        try:
            payload = {
                "messages": [{"role": "user", "content": query}],
                "model": "claude-3-5-sonnet-20241022",
                "stream": True,
                "max_tokens": 500
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/v1/chat/stream",
                    json=payload,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        chunks_received = 0
                        total_content = ""
                        
                        async for line in response.content:
                            if line:
                                line_str = line.decode('utf-8').strip()
                                if line_str.startswith('data: '):
                                    data_str = line_str[6:]  # Remove 'data: ' prefix
                                    if data_str != '[DONE]':
                                        try:
                                            chunk_data = json.loads(data_str)
                                            choices = chunk_data.get('choices', [])
                                            if choices:
                                                delta = choices[0].get('delta', {})
                                                content = delta.get('content', '')
                                                if content:
                                                    total_content += content
                                                    chunks_received += 1
                                        except json.JSONDecodeError:
                                            continue
                        
                        if chunks_received > 0:
                            self.log(f"✅ Streaming response successful: {chunks_received} chunks received")
                            self.record_result(
                                "Streaming RAG Query",
                                True,
                                {
                                    'chunks_received': chunks_received,
                                    'total_length': len(total_content),
                                    'preview': total_content[:200] + "..." if len(total_content) > 200 else total_content
                                }
                            )
                            return True
                        else:
                            self.log(f"❌ No streaming chunks received")
                            return False
                    else:
                        self.log(f"❌ Streaming request failed: {response.status}")
                        return False
        except Exception as e:
            self.log(f"❌ Streaming test failed: {str(e)}")
            self.record_result(
                "Streaming RAG Query",
                False,
                {'error': str(e)}
            )
            return False
    
    async def cleanup_documents(self):
        """Clean up uploaded test documents."""
        self.log("🧹 Cleaning up uploaded documents...")
        for doc in self.uploaded_documents:
            try:
                response = requests.delete(f"{self.base_url}/api/v1/documents/{doc['id']}")
                if response.status_code == 200:
                    self.log(f"✅ Deleted document: {doc['name']}")
                else:
                    self.log(f"❌ Failed to delete document {doc['name']}: {response.status_code}")
            except Exception as e:
                self.log(f"❌ Error deleting document {doc['name']}: {str(e)}")
    
    def generate_report(self):
        """Generate a test report."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'success_rate': f"{(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%"
            },
            'test_results': self.test_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save report to file
        report_file = f"claude_real_api_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.log(f"📊 Test report saved to: {report_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 CLAUDE REAL API INTEGRATION TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {report['summary']['success_rate']}")
        print("="*60)
        
        return report
    
    async def run_full_test_suite(self):
        """Run the complete test suite."""
        self.log("🚀 Starting Claude Real API Integration Tests")
        self.log("="*60)
        
        try:
            # Phase 1: Server Health Check
            self.log("📋 Phase 1: Server Health Check")
            if not await self.check_server_health():
                self.log("❌ Server health check failed. Aborting tests.")
                return
            
            # Phase 2: Document Upload and Processing
            self.log("\n📋 Phase 2: Document Upload and Processing")
            processed_docs = []
            
            for doc_config in self.test_documents:
                self.log(f"\n📄 Processing {doc_config['type']} document: {doc_config['name']}")
                
                # Upload document
                doc_id = await self.upload_document(doc_config)
                if not doc_id:
                    continue
                
                # Wait for processing
                if await self.wait_for_processing(doc_id):
                    # Verify chunks
                    if await self.verify_document_chunks(doc_id, doc_config['expected_content']):
                        processed_docs.append((doc_id, doc_config))
                        self.record_result(
                            f"Document Processing: {doc_config['type']}",
                            True,
                            {'document_id': doc_id, 'chunks_verified': True}
                        )
                    else:
                        self.record_result(
                            f"Document Processing: {doc_config['type']}",
                            False,
                            {'document_id': doc_id, 'error': 'Chunk verification failed'}
                        )
                else:
                    self.record_result(
                        f"Document Processing: {doc_config['type']}",
                        False,
                        {'document_id': doc_id, 'error': 'Processing timeout or failure'}
                    )
            
            # Phase 3: RAG Query Testing
            self.log("\n📋 Phase 3: RAG Query Testing")
            if processed_docs:
                for doc_id, doc_config in processed_docs:
                    self.log(f"\n🔍 Testing RAG query for {doc_config['type']} document")
                    await self.test_rag_query(doc_config['test_query'], doc_config['expected_content'])
                
                # Test a general query that should use multiple documents
                self.log("\n🔍 Testing cross-document RAG query")
                await self.test_rag_query(
                    "What information is available in the uploaded documents?",
                    "document"
                )
            else:
                self.log("❌ No documents processed successfully. Skipping RAG tests.")
            
            # Phase 4: Streaming Response Test
            self.log("\n📋 Phase 4: Streaming Response Test")
            if processed_docs:
                await self.test_streaming_response("Provide a summary of the uploaded documents")
            else:
                self.log("❌ No documents available for streaming test")
            
            # Phase 5: Edge Case Testing
            self.log("\n📋 Phase 5: Edge Case Testing")
            
            # Test query with no relevant context
            await self.test_rag_query(
                "What is the weather on Mars?",
                "weather"  # This should fail relevance check
            )
            
        except Exception as e:
            self.log(f"❌ Test suite failed with exception: {str(e)}", "ERROR")
            self.record_result(
                "Test Suite Execution",
                False,
                {'error': str(e)}
            )
        
        finally:
            # Cleanup
            await self.cleanup_documents()
            
            # Generate report
            report = self.generate_report()
            
            self.log("\n🎉 Claude Real API Integration Tests Complete!")
            return report


async def main():
    """Main entry point for the test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Claude Real API Integration Test Runner')
    parser.add_argument('--base-url', default='http://localhost:8000', 
                       help='Base URL for the FastAPI server')
    parser.add_argument('--api-key', help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Check for API key
    api_key = args.api_key or os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set and --api-key not provided")
        print("Please set your Claude API key:")
        print("  export ANTHROPIC_API_KEY=your_api_key_here")
        print("  or use --api-key your_api_key_here")
        sys.exit(1)
    
    # Create and run tester
    tester = ClaudeRealAPITester(base_url=args.base_url, api_key=api_key)
    
    try:
        report = await tester.run_full_test_suite()
        
        # Exit with appropriate code
        if report['summary']['failed'] == 0:
            print("\n✅ All tests passed!")
            sys.exit(0)
        else:
            print(f"\n❌ {report['summary']['failed']} test(s) failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        await tester.cleanup_documents()
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
