"""
Document Format Validation Harness

This script validates the complete document ingestion and citation pipeline
across all supported formats (.pdf, .docx, .html, .md, .txt) for the
NextGen AI Platform's government-grade RAG system.

USAGE:
    python test_document_format_validation.py

REQUIREMENTS:
    - FastAPI server running on localhost:8000
    - Test documents in ./test_documents/ directory
    - Valid API endpoints for document upload and chat completions
"""

import os
import time
import json
import requests
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


class DocumentFormatValidator:
    """Validates document format support across the RAG pipeline"""
    
    def __init__(self, base_url: str = "http://localhost:8000", username: str = "admin@example.com", password: str = "adminpassword"):
        self.base_url = base_url.rstrip('/')
        self.upload_endpoint = f"{self.base_url}/api/v1/documents/"
        self.chat_endpoint = f"{self.base_url}/api/v1/chat/completions/"
        self.documents_endpoint = f"{self.base_url}/api/v1/documents/"
        self.token_endpoint = f"{self.base_url}/api/v1/token"
        
        # Authentication credentials
        self.username = username
        self.password = password
        self.access_token = None
        self.headers = {}
        
        # Test configuration for each supported format
        self.test_configs = {
            "PDF": {
                "file_path": "test_documents/citizenship_requirements.pdf",
                "expected_filename": "citizenship_requirements.pdf",
                "query": "What is the physical presence requirement for Canadian citizenship?",
                "expected_keywords": ["1,095 days", "3 years", "physically present", "citizenship"]
            },
            "DOCX": {
                "file_path": "test_documents/eligibility_criteria.docx", 
                "expected_filename": "eligibility_criteria.docx",
                "query": "What are the eligibility criteria for government services?",
                "expected_keywords": ["legal resident", "18 years", "identification", "income"]
            },
            "HTML": {
                "file_path": "test_documents/refund_policy.html",
                "expected_filename": "refund_policy.html", 
                "query": "What is the refund policy for government services?",
                "expected_keywords": ["refund", "90 days", "processing fee", "duplicate"]
            },
            "Markdown": {
                "file_path": "test_documents/eligibility_criteria.md",
                "expected_filename": "eligibility_criteria.md",
                "query": "What is the installation command for the platform?",
                "expected_keywords": ["pip install", "nextgen-ai-platform"]
            },
            "TXT": {
                "file_path": "test_documents/citizenship_guide.txt",
                "expected_filename": "citizenship_guide.txt", 
                "query": "What is the definition of AI?",
                "expected_keywords": ["Artificial Intelligence", "computer systems", "human intelligence"]
            }
        }
        
        self.results = {}
    
    def authenticate(self) -> bool:
        """
        Authenticate with the API and get access token
        
        Returns:
            True if authentication successful, False otherwise
        """
        
        try:
            print("🔐 Authenticating with API...")
            
            # Prepare authentication data
            auth_data = {
                "username": self.username,
                "password": self.password
            }
            
            # Send authentication request
            response = requests.post(
                self.token_endpoint,
                data=auth_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10
            )
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data.get("access_token")
                
                if self.access_token:
                    # Set up authorization header for future requests
                    self.headers["Authorization"] = f"Bearer {self.access_token}"
                    print("✅ Authentication successful")
                    return True
                else:
                    print("❌ No access token in response")
                    return False
            else:
                error_msg = f"Authentication failed: {response.status_code} - {response.text}"
                print(f"❌ {error_msg}")
                return False
                
        except Exception as e:
            print(f"❌ Authentication error: {str(e)}")
            return False
    
    def check_server_health(self) -> bool:
        """Check if the FastAPI server is running and accessible"""
        
        try:
            # Try the root health check endpoint
            health_url = f"{self.base_url}/"
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("status") == "healthy":
                    print("✅ Server is running and accessible")
                    print(f"  Service: {response_data.get('service', 'Unknown')}")
                    print(f"  Environment: {response_data.get('environment', 'Unknown')}")
                    return True
        except requests.exceptions.RequestException:
            pass
        
        # Try alternative health check via docs
        try:
            response = requests.get(f"{self.base_url}/api/v1/docs", timeout=5)
            if response.status_code == 200:
                print("✅ Server is running (docs accessible)")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print("❌ Server is not accessible. Please ensure FastAPI server is running on localhost:8000")
        return False
    
    def upload_document(self, file_path: str, format_name: str) -> Tuple[bool, Optional[str], str]:
        """
        Upload a document to the platform
        
        Returns:
            (success, document_id, message)
        """
        
        if not os.path.exists(file_path):
            return False, None, f"File not found: {file_path}"
        
        try:
            filename = os.path.basename(file_path)
            
            with open(file_path, 'rb') as f:
                files = {'file': (filename, f, self._get_content_type(file_path))}
                
                print(f"  📤 Uploading {filename}...")
                response = requests.post(
                    self.upload_endpoint,
                    files=files,
                    headers=self.headers,
                    timeout=30
                )
            
            if response.status_code in [200, 201]:
                response_data = response.json()
                document_id = response_data.get('id') or response_data.get('document_id')
                print(f"  ✅ Upload successful (ID: {document_id})")
                return True, document_id, "Upload successful"
            else:
                error_msg = f"Upload failed: {response.status_code} - {response.text}"
                print(f"  ❌ {error_msg}")
                return False, None, error_msg
                
        except Exception as e:
            error_msg = f"Upload error: {str(e)}"
            print(f"  ❌ {error_msg}")
            return False, None, error_msg
    
    def _get_content_type(self, file_path: str) -> str:
        """Get appropriate content type for file"""
        
        ext = Path(file_path).suffix.lower()
        content_types = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.html': 'text/html',
            '.md': 'text/markdown',
            '.txt': 'text/plain'
        }
        return content_types.get(ext, 'application/octet-stream')
    
    def wait_for_processing(self, document_id: str, max_wait: int = 120) -> Tuple[bool, str]:
        """
        Wait for document processing to complete
        
        Returns:
            (success, status_message)
        """
        
        if not document_id:
            return False, "No document ID provided"
        
        print(f"  ⏳ Waiting for processing (max {max_wait}s)...")
        
        start_time = time.time()
        last_status = None
        stable_count = 0
        
        while time.time() - start_time < max_wait:
            try:
                # Check document status
                status_url = f"{self.documents_endpoint}{document_id}"
                response = requests.get(status_url, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    doc_data = response.json()
                    status = doc_data.get('status', 'UNKNOWN')
                    
                    # Normalize status to handle case variations
                    status_upper = status.upper()
                    
                    if status_upper == 'PROCESSED':
                        print(f"  ✅ Processing complete (status: {status})")
                        return True, "Processing complete"
                    elif status_upper in ['FAILED', 'ERROR']:
                        error_msg = f"Processing failed with status: {status}"
                        print(f"  ❌ {error_msg}")
                        return False, error_msg
                    elif status_upper == 'PROCESSING':
                        print(f"  ⏳ Status: {status}")
                        last_status = status
                        stable_count = 0
                    else:
                        # Handle other statuses - if status is stable for multiple checks, consider it processed
                        if status == last_status:
                            stable_count += 1
                            if stable_count >= 3:  # Status stable for 3 checks (6 seconds)
                                # Check if we can find evidence of successful processing
                                if self._check_processing_evidence(doc_data):
                                    print(f"  ✅ Processing appears complete (stable status: {status})")
                                    return True, f"Processing complete (status: {status})"
                        else:
                            stable_count = 0
                        
                        print(f"  ⏳ Status: {status} (stable: {stable_count})")
                        last_status = status
                
            except Exception as e:
                print(f"  ⚠️  Status check error: {e}")
            
            time.sleep(2)
        
        error_msg = f"Processing timeout after {max_wait}s"
        print(f"  ❌ {error_msg}")
        return False, error_msg
    
    def _check_processing_evidence(self, doc_data: Dict[str, Any]) -> bool:
        """
        Check for evidence that document processing completed successfully
        
        Returns:
            True if evidence suggests processing is complete
        """
        
        # Look for indicators that processing completed
        indicators = [
            # Document has chunks
            doc_data.get('chunks_count', 0) > 0,
            # Document has embeddings
            doc_data.get('embeddings_count', 0) > 0,
            # Document has content
            bool(doc_data.get('content')),
            # No error message
            not doc_data.get('error_message'),
            # Status is not explicitly failed
            doc_data.get('status', '').upper() not in ['FAILED', 'ERROR', 'PENDING', 'UPLOADING']
        ]
        
        # If at least 2 indicators suggest success, consider it processed
        return sum(indicators) >= 2
    
    def query_document(self, query: str, format_name: str) -> Tuple[bool, Optional[str], str]:
        """
        Send a query and get response with citations
        
        Returns:
            (success, response_text, message)
        """
        
        try:
            print(f"  🔍 Querying: '{query}'")
            
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "model": "claude-3-5-sonnet-20241022",  # Default model
                "retrieve": True,  # Enable RAG retrieval - CRITICAL for document access
                "max_tokens": 1000,
                "temperature": 0.1
            }
            
            response = requests.post(
                self.chat_endpoint,
                json=payload,
                headers=self.headers,
                timeout=60
            )
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Extract response text from different possible formats
                response_text = None
                if 'choices' in response_data and response_data['choices']:
                    # OpenAI-style format
                    choice = response_data['choices'][0]
                    if 'message' in choice:
                        response_text = choice['message'].get('content', '')
                    elif 'text' in choice:
                        response_text = choice['text']
                elif 'response' in response_data:
                    # Direct response format
                    response_text = response_data['response']
                elif 'content' in response_data:
                    # Alternative format
                    response_text = response_data['content']
                
                if response_text:
                    print(f"  ✅ Response received ({len(response_text)} chars)")
                    return True, response_text, "Query successful"
                else:
                    error_msg = "No response content found"
                    print(f"  ❌ {error_msg}")
                    return False, None, error_msg
            else:
                error_msg = f"Query failed: {response.status_code} - {response.text}"
                print(f"  ❌ {error_msg}")
                return False, None, error_msg
                
        except Exception as e:
            error_msg = f"Query error: {str(e)}"
            print(f"  ❌ {error_msg}")
            return False, None, error_msg
    
    def parse_citations(self, response_text: str) -> List[str]:
        """
        Parse the Sources section from response text to extract citations
        
        Returns:
            List of citation strings
        """
        
        citations = []
        
        if not response_text:
            return citations
        
        # Look for Sources section
        sources_markers = ["Sources:", "sources:", "SOURCES:", "References:", "Citations:"]
        
        for marker in sources_markers:
            if marker in response_text:
                # Split on the marker and take everything after it
                parts = response_text.split(marker, 1)
                if len(parts) > 1:
                    sources_section = parts[1].strip()
                    
                    # Split into lines and extract citations
                    lines = sources_section.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and ('📄' in line or line.startswith('-') or line.startswith('•')):
                            citations.append(line)
                    break
        
        return citations
    
    def validate_citations(self, citations: List[str], expected_filename: str) -> Tuple[bool, str]:
        """
        Validate that citations contain the expected filename
        
        Returns:
            (success, message)
        """
        
        if not citations:
            return False, "No citations found in response"
        
        # Check if any citation contains the expected filename
        filename_base = Path(expected_filename).stem  # Remove extension for flexible matching
        
        for citation in citations:
            if expected_filename in citation or filename_base in citation:
                print(f"  ✅ Citation found: {citation}")
                return True, f"Citation contains expected filename: {expected_filename}"
        
        # If exact match not found, show what we got
        citations_text = "; ".join(citations)
        error_msg = f"Expected filename '{expected_filename}' not found in citations: {citations_text}"
        print(f"  ❌ {error_msg}")
        return False, error_msg
    
    def validate_content(self, response_text: str, expected_keywords: List[str]) -> Tuple[bool, str]:
        """
        Validate that response contains expected keywords from the document
        
        Returns:
            (success, message)
        """
        
        if not response_text:
            return False, "No response text to validate"
        
        response_lower = response_text.lower()
        found_keywords = []
        missing_keywords = []
        
        for keyword in expected_keywords:
            if keyword.lower() in response_lower:
                found_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        if found_keywords:
            print(f"  ✅ Found keywords: {', '.join(found_keywords)}")
            if missing_keywords:
                print(f"  ⚠️  Missing keywords: {', '.join(missing_keywords)}")
            return True, f"Found {len(found_keywords)}/{len(expected_keywords)} expected keywords"
        else:
            error_msg = f"No expected keywords found. Missing: {', '.join(missing_keywords)}"
            print(f"  ❌ {error_msg}")
            return False, error_msg
    
    def test_format(self, format_name: str) -> Dict[str, Any]:
        """
        Test a single document format through the complete pipeline
        
        Returns:
            Test result dictionary
        """
        
        config = self.test_configs[format_name]
        result = {
            "format": format_name,
            "file_path": config["file_path"],
            "upload_success": False,
            "processing_success": False,
            "query_success": False,
            "citation_success": False,
            "content_success": False,
            "overall_success": False,
            "messages": []
        }
        
        print(f"\n🧪 Testing {format_name} Format")
        print("=" * 50)
        
        # Step 1: Upload document
        upload_success, document_id, upload_msg = self.upload_document(
            config["file_path"], format_name
        )
        result["upload_success"] = upload_success
        result["messages"].append(f"Upload: {upload_msg}")
        
        if not upload_success:
            result["messages"].append("❌ Upload failed - skipping remaining tests")
            return result
        
        # Step 2: Wait for processing
        processing_success, processing_msg = self.wait_for_processing(document_id)
        result["processing_success"] = processing_success
        result["messages"].append(f"Processing: {processing_msg}")
        
        if not processing_success:
            result["messages"].append("❌ Processing failed - skipping remaining tests")
            return result
        
        # Step 3: Query document
        query_success, response_text, query_msg = self.query_document(
            config["query"], format_name
        )
        result["query_success"] = query_success
        result["messages"].append(f"Query: {query_msg}")
        
        if not query_success:
            result["messages"].append("❌ Query failed - skipping validation tests")
            return result
        
        # Step 4: Validate citations
        citations = self.parse_citations(response_text)
        citation_success, citation_msg = self.validate_citations(
            citations, config["expected_filename"]
        )
        result["citation_success"] = citation_success
        result["messages"].append(f"Citations: {citation_msg}")
        
        # Step 5: Validate content
        content_success, content_msg = self.validate_content(
            response_text, config["expected_keywords"]
        )
        result["content_success"] = content_success
        result["messages"].append(f"Content: {content_msg}")
        
        # Overall success requires all steps to pass
        result["overall_success"] = (
            upload_success and 
            processing_success and 
            query_success and 
            citation_success and 
            content_success
        )
        
        if result["overall_success"]:
            print(f"  🎉 {format_name} format: PASSED")
        else:
            print(f"  ❌ {format_name} format: FAILED")
        
        return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run validation tests for all supported formats
        
        Returns:
            Complete test results
        """
        
        print("🎯 Document Format Validation Harness")
        print("=" * 70)
        print("Testing complete document ingestion and citation pipeline")
        print("across all supported formats for government-grade RAG system")
        print()
        
        # Check server health first
        if not self.check_server_health():
            return {
                "server_accessible": False,
                "formats_tested": 0,
                "formats_passed": 0,
                "overall_success": False,
                "results": {}
            }
        
        # Authenticate with the API
        if not self.authenticate():
            return {
                "server_accessible": True,
                "authentication_failed": True,
                "formats_tested": 0,
                "formats_passed": 0,
                "overall_success": False,
                "results": {}
            }
        
        # Test each format
        for format_name in self.test_configs.keys():
            try:
                result = self.test_format(format_name)
                self.results[format_name] = result
            except Exception as e:
                print(f"\n❌ Unexpected error testing {format_name}: {e}")
                self.results[format_name] = {
                    "format": format_name,
                    "overall_success": False,
                    "error": str(e)
                }
        
        # Add delay after all documents are processed to ensure indexing completes
        if self.results:
            print("\n⏳ Waiting for document indexing to complete...")
            time.sleep(10)  # Allow time for all documents to be fully indexed
        
        # Generate summary
        return self.generate_summary()
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate test summary and results"""
        
        print("\n" + "=" * 70)
        print("📊 VALIDATION SUMMARY")
        print("=" * 70)
        
        formats_tested = len(self.results)
        formats_passed = sum(1 for r in self.results.values() if r.get("overall_success", False))
        
        # Print results for each format
        for format_name, result in self.results.items():
            status = "✅ PASSED" if result.get("overall_success", False) else "❌ FAILED"
            print(f"{status} {format_name}")
            
            if result.get("overall_success", False):
                print(f"  ✅ Upload success")
                print(f"  ✅ Ingestion success") 
                print(f"  ✅ Response received")
                print(f"  ✅ Citation correctness")
                print(f"  ✅ Content validation")
            else:
                # Show what failed
                checks = [
                    ("Upload", result.get("upload_success", False)),
                    ("Processing", result.get("processing_success", False)),
                    ("Query", result.get("query_success", False)),
                    ("Citations", result.get("citation_success", False)),
                    ("Content", result.get("content_success", False))
                ]
                
                for check_name, success in checks:
                    status_icon = "✅" if success else "❌"
                    print(f"  {status_icon} {check_name}")
                
                # Show error messages
                if "messages" in result:
                    for msg in result["messages"]:
                        if "failed" in msg.lower() or "error" in msg.lower():
                            print(f"    💬 {msg}")
            print()
        
        # Overall results
        overall_success = formats_passed == formats_tested
        
        print(f"Overall Results: {formats_passed}/{formats_tested} formats passed")
        
        if overall_success:
            print("\n🎉 ALL FORMAT VALIDATION TESTS PASSED!")
            print("✅ Complete ingestion → retrieval → citation pipeline working")
            print("✅ All supported formats (.pdf, .docx, .html, .md, .txt) validated")
            print("✅ Citation functionality confirmed across all formats")
            print("✅ Government-grade RAG system ready for production")
        else:
            print(f"\n❌ {formats_tested - formats_passed} FORMAT(S) FAILED!")
            print("Please review the failures above and fix the issues.")
            
            # Provide debugging guidance
            failed_formats = [
                name for name, result in self.results.items() 
                if not result.get("overall_success", False)
            ]
            print(f"Failed formats: {', '.join(failed_formats)}")
        
        return {
            "server_accessible": True,
            "formats_tested": formats_tested,
            "formats_passed": formats_passed,
            "overall_success": overall_success,
            "results": self.results
        }


def main():
    """Main execution function"""
    
    # Create validator instance
    validator = DocumentFormatValidator()
    
    # Run all tests
    summary = validator.run_all_tests()
    
    # Save results to file
    results_file = "format_validation_results.json"
    try:
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n📄 Detailed results saved to: {results_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save results file: {e}")
    
    # Exit with appropriate code
    exit_code = 0 if summary.get("overall_success", False) else 1
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
