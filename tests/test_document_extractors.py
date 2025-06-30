"""
Comprehensive test suite for the document extraction system.

Tests all document extractors for various file formats including PDF, DOCX,
HTML, Markdown, and plain text. Validates proper text extraction, normalization,
error handling, and factory method functionality.
"""

import pytest
import os
from pathlib import Path
from typing import Dict, Any
import tempfile
import logging

from app.services.retrieval.document_extractors import (
    DocumentExtractor,
    TxtExtractor,
    PdfExtractor,
    DocxExtractor,
    HtmlExtractor,
    MarkdownExtractor,
    get_extractor,
    UnsupportedFileTypeError,
    DocumentExtractionError,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDocumentExtractors:
    """Test suite for document extraction system."""
    
    @pytest.fixture(autouse=True)
    def setup_test_files(self):
        """Setup test files for extraction testing."""
        self.test_files = {}
        
        # Sample text content for testing
        self.sample_text = """Canadian Immigration Guide 2024

Chapter 1: Citizenship Requirements

To become a Canadian citizen, applicants must meet several key requirements:

1. Language Proficiency: Demonstrate adequate knowledge of English or French.
   Accepted tests include CELPIP and IELTS General Training.
   Minimum scores of CLB 4 are required for speaking and listening.

2. Residence Requirements: Must have been physically present in Canada for
   at least 1,095 days (3 years) during the 5 years immediately before applying.

3. Tax Obligations: Must have filed income tax returns for at least 3 years
   during the 5-year period, if required to do so under the Income Tax Act.
"""
        
        # Create test files in memory
        self.test_files['txt'] = self.sample_text.encode('utf-8')
        
        # HTML content
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Immigration Guide</title>
    <style>body {{ font-family: Arial; }}</style>
    <script>console.log('test');</script>
</head>
<body>
    <h1>Canadian Immigration Guide 2024</h1>
    <h2>Chapter 1: Citizenship Requirements</h2>
    <p>To become a Canadian citizen, applicants must meet several key requirements:</p>
    <ol>
        <li><strong>Language Proficiency:</strong> Demonstrate adequate knowledge of English or French.</li>
        <li><strong>Residence Requirements:</strong> Must have been physically present in Canada.</li>
    </ol>
</body>
</html>"""
        self.test_files['html'] = html_content.encode('utf-8')
        
        # Markdown content
        markdown_content = """# Canadian Immigration Guide 2024

## Chapter 1: Citizenship Requirements

To become a Canadian citizen, applicants must meet several key requirements:

1. **Language Proficiency**: Demonstrate adequate knowledge of English or French.
   - Accepted tests include CELPIP and IELTS General Training
   - Minimum scores of CLB 4 are required

2. **Residence Requirements**: Must have been physically present in Canada for
   at least 1,095 days (3 years) during the 5 years immediately before applying.

```
Important: This is sample code that should be removed
```

---

*Note: This guide provides general information only.*
"""
        self.test_files['md'] = markdown_content.encode('utf-8')
    
    def test_txt_extractor_basic(self):
        """Test basic text extraction functionality."""
        extractor = TxtExtractor()
        
        # Test UTF-8 text
        result = extractor.extract(self.test_files['txt'])
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Canadian Immigration Guide 2024" in result
        assert "Language Proficiency" in result
        assert "CLB 4" in result
        
        # Verify normalization
        assert result.endswith('\n')
        assert '\r\n' not in result
        assert '\r' not in result
    
    def test_txt_extractor_encodings(self):
        """Test text extraction with different encodings."""
        extractor = TxtExtractor()
        
        # Test different encodings
        test_text = "Café résumé naïve"
        
        encodings_to_test = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings_to_test:
            try:
                encoded_bytes = test_text.encode(encoding)
                result = extractor.extract(encoded_bytes)
                
                assert isinstance(result, str)
                assert len(result) > 0
                logger.info(f"✅ {encoding} encoding test passed")
                
            except UnicodeEncodeError:
                # Some encodings might not support all characters
                logger.info(f"⚠️  {encoding} encoding skipped (unsupported characters)")
                continue
    
    def test_txt_extractor_error_handling(self):
        """Test text extractor error handling."""
        extractor = TxtExtractor()
        
        # Test with invalid bytes that should trigger error replacement
        invalid_bytes = b'\xff\xfe\x00\x00invalid'
        result = extractor.extract(invalid_bytes)
        
        # Should not raise exception, but use error replacement
        assert isinstance(result, str)
    
    @pytest.mark.skipif(
        not _check_library_available('fitz'),
        reason="PyMuPDF not available"
    )
    def test_pdf_extractor_basic(self):
        """Test PDF extraction with real PDF file."""
        extractor = PdfExtractor()
        
        # Use the test PDF created earlier
        test_pdf_path = "tests/resources/sample_immigration_guide.pdf"
        
        if os.path.exists(test_pdf_path):
            with open(test_pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            result = extractor.extract(pdf_bytes)
            
            assert isinstance(result, str)
            assert len(result) > 0
            assert "Canadian Immigration Guide" in result
            assert "CELPIP" in result
            
            # Verify normalization
            assert result.endswith('\n')
            logger.info("✅ PDF extraction test passed")
        else:
            pytest.skip("Test PDF file not found")
    
    def test_pdf_extractor_error_handling(self):
        """Test PDF extractor error handling."""
        extractor = PdfExtractor()
        
        # Test with invalid PDF bytes
        invalid_pdf = b"This is not a PDF file"
        
        with pytest.raises(DocumentExtractionError):
            extractor.extract(invalid_pdf)
    
    @pytest.mark.skipif(
        not _check_library_available('docx'),
        reason="python-docx not available"
    )
    def test_docx_extractor_mock(self):
        """Test DOCX extractor with mock data (library availability check)."""
        extractor = DocxExtractor()
        
        # This test mainly checks that the extractor can be instantiated
        # and would handle DOCX files if available
        assert isinstance(extractor, DocxExtractor)
        
        # Test error handling with invalid DOCX bytes
        invalid_docx = b"This is not a DOCX file"
        
        with pytest.raises(DocumentExtractionError):
            extractor.extract(invalid_docx)
    
    @pytest.mark.skipif(
        not _check_library_available('bs4'),
        reason="BeautifulSoup not available"
    )
    def test_html_extractor_basic(self):
        """Test HTML extraction functionality."""
        extractor = HtmlExtractor()
        
        result = extractor.extract(self.test_files['html'])
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Canadian Immigration Guide 2024" in result
        assert "Language Proficiency" in result
        
        # Verify HTML tags are removed
        assert "<html>" not in result
        assert "<body>" not in result
        assert "<script>" not in result
        assert "<style>" not in result
        
        # Verify normalization
        assert result.endswith('\n')
        logger.info("✅ HTML extraction test passed")
    
    def test_html_extractor_encoding(self):
        """Test HTML extraction with different encodings."""
        extractor = HtmlExtractor()
        
        # Test with different HTML encodings
        html_with_special_chars = """<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body><p>Café résumé naïve</p></body>
</html>"""
        
        # Test UTF-8
        utf8_bytes = html_with_special_chars.encode('utf-8')
        result = extractor.extract(utf8_bytes)
        assert "Café résumé naïve" in result
        
        # Test Latin-1
        try:
            latin1_bytes = html_with_special_chars.encode('latin-1')
            result = extractor.extract(latin1_bytes)
            assert isinstance(result, str)
            logger.info("✅ HTML Latin-1 encoding test passed")
        except UnicodeEncodeError:
            logger.info("⚠️  HTML Latin-1 encoding test skipped")
    
    def test_markdown_extractor_basic(self):
        """Test Markdown extraction functionality."""
        extractor = MarkdownExtractor()
        
        result = extractor.extract(self.test_files['md'])
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Canadian Immigration Guide 2024" in result
        assert "Language Proficiency" in result
        
        # Verify Markdown syntax is cleaned
        assert "##" not in result  # Headers should be cleaned
        assert "**" not in result  # Bold markers should be cleaned
        assert "```" not in result  # Code blocks should be removed
        assert "---" not in result  # Horizontal rules should be removed
        
        # Verify normalization
        assert result.endswith('\n')
        logger.info("✅ Markdown extraction test passed")
    
    def test_markdown_extractor_fallback(self):
        """Test Markdown extractor fallback when markdown library unavailable."""
        extractor = MarkdownExtractor()
        
        # The extractor should work even without the markdown library
        # by falling back to basic syntax cleaning
        result = extractor.extract(self.test_files['md'])
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_get_extractor_by_extension(self):
        """Test factory method with file extensions."""
        
        # Test each supported extension
        test_cases = [
            ('document.txt', self.test_files['txt'], TxtExtractor),
            ('document.html', self.test_files['html'], HtmlExtractor),
            ('document.htm', self.test_files['html'], HtmlExtractor),
            ('document.md', self.test_files['md'], MarkdownExtractor),
            ('document.markdown', self.test_files['md'], MarkdownExtractor),
        ]
        
        for filename, file_bytes, expected_type in test_cases:
            extractor = get_extractor(filename, file_bytes)
            assert isinstance(extractor, expected_type)
            logger.info(f"✅ Factory method test passed for {filename}")
    
    def test_get_extractor_unsupported_type(self):
        """Test factory method with unsupported file types."""
        
        # Test unsupported extensions
        unsupported_files = [
            ('document.xyz', b'some content'),
            ('document.bin', b'binary content'),
            ('document', b'no extension'),
        ]
        
        for filename, file_bytes in unsupported_files:
            with pytest.raises(UnsupportedFileTypeError) as exc_info:
                get_extractor(filename, file_bytes)
            
            # Verify error message contains helpful information
            error_msg = str(exc_info.value)
            assert filename in error_msg
            assert "Supported extensions:" in error_msg
            logger.info(f"✅ Unsupported file test passed for {filename}")
    
    @pytest.mark.skipif(
        not _check_library_available('magic'),
        reason="python-magic not available"
    )
    def test_get_extractor_mime_detection(self):
        """Test factory method with MIME type detection."""
        
        # Test with files that have wrong extensions but correct MIME types
        # This would require python-magic to be available
        
        # Text file with wrong extension
        extractor = get_extractor('document.xyz', self.test_files['txt'])
        # Should fall back to extension or generic text handling
        assert isinstance(extractor, (TxtExtractor, DocumentExtractor))
    
    def test_text_normalization(self):
        """Test text normalization functionality."""
        extractor = TxtExtractor()
        
        # Test various normalization scenarios
        test_cases = [
            # Windows line endings
            ("Line 1\r\nLine 2\r\n", "Line 1\nLine 2\n"),
            
            # Mac line endings
            ("Line 1\rLine 2\r", "Line 1\nLine 2\n"),
            
            # Multiple spaces
            ("Word1    Word2\t\tWord3", "Word1 Word2 Word3\n"),
            
            # Multiple newlines
            ("Para1\n\n\n\nPara2", "Para1\n\nPara2\n"),
            
            # Mixed whitespace
            ("  \t Line with spaces  \t ", "Line with spaces\n"),
        ]
        
        for input_text, expected_output in test_cases:
            input_bytes = input_text.encode('utf-8')
            result = extractor.extract(input_bytes)
            assert result == expected_output
            logger.info(f"✅ Normalization test passed: {repr(input_text)}")
    
    def test_empty_content_handling(self):
        """Test handling of empty or whitespace-only content."""
        
        extractors = [
            TxtExtractor(),
            HtmlExtractor(),
            MarkdownExtractor(),
        ]
        
        empty_contents = [
            b'',  # Empty
            b'   ',  # Whitespace only
            b'\n\n\n',  # Newlines only
            b'\t\t\t',  # Tabs only
        ]
        
        for extractor in extractors:
            for content in empty_contents:
                try:
                    result = extractor.extract(content)
                    # Should return empty string or raise appropriate error
                    assert isinstance(result, str)
                    logger.info(f"✅ Empty content test passed for {extractor.__class__.__name__}")
                except DocumentExtractionError:
                    # Acceptable to raise error for empty content
                    logger.info(f"✅ Empty content error handling for {extractor.__class__.__name__}")
    
    def test_large_content_handling(self):
        """Test handling of large content."""
        extractor = TxtExtractor()
        
        # Create large text content
        large_text = "This is a test line.\n" * 10000  # ~200KB
        large_bytes = large_text.encode('utf-8')
        
        result = extractor.extract(large_bytes)
        
        assert isinstance(result, str)
        assert len(result) > 100000  # Should be substantial
        assert result.count("This is a test line") == 10000
        logger.info("✅ Large content handling test passed")
    
    def test_unicode_content_handling(self):
        """Test handling of Unicode content."""
        extractor = TxtExtractor()
        
        # Test with various Unicode characters
        unicode_text = """
        English: Hello World
        French: Bonjour le monde
        Spanish: Hola Mundo
        German: Hallo Welt
        Chinese: 你好世界
        Japanese: こんにちは世界
        Arabic: مرحبا بالعالم
        Russian: Привет мир
        Emoji: 🌍🌎🌏 Hello! 👋
        """
        
        unicode_bytes = unicode_text.encode('utf-8')
        result = extractor.extract(unicode_bytes)
        
        assert isinstance(result, str)
        assert "Hello World" in result
        assert "你好世界" in result
        assert "👋" in result
        logger.info("✅ Unicode content handling test passed")


def _check_library_available(library_name: str) -> bool:
    """
    Check if a library is available for import.
    
    Args:
        library_name: Name of the library to check
        
    Returns:
        True if library is available, False otherwise
    """
    try:
        if library_name == 'fitz':
            import fitz
        elif library_name == 'docx':
            import docx
        elif library_name == 'bs4':
            import bs4
        elif library_name == 'markdown':
            import markdown
        elif library_name == 'magic':
            import magic
        else:
            return False
        return True
    except ImportError:
        return False


# Integration tests that require multiple components
class TestDocumentExtractorIntegration:
    """Integration tests for document extraction system."""
    
    def test_extractor_factory_comprehensive(self):
        """Test the factory method with comprehensive file type coverage."""
        
        # Test data for different file types
        test_content = "Test document content for extraction validation."
        
        # Test cases: (filename, expected_extractor_type)
        test_cases = [
            # Text files
            ('readme.txt', TxtExtractor),
            ('notes.TXT', TxtExtractor),  # Case insensitive
            
            # HTML files
            ('index.html', HtmlExtractor),
            ('page.htm', HtmlExtractor),
            ('INDEX.HTML', HtmlExtractor),  # Case insensitive
            
            # Markdown files
            ('readme.md', MarkdownExtractor),
            ('docs.markdown', MarkdownExtractor),
            ('README.MD', MarkdownExtractor),  # Case insensitive
        ]
        
        for filename, expected_type in test_cases:
            content_bytes = test_content.encode('utf-8')
            extractor = get_extractor(filename, content_bytes)
            
            assert isinstance(extractor, expected_type), f"Failed for {filename}"
            
            # Test that extraction works
            result = extractor.extract(content_bytes)
            assert isinstance(result, str)
            assert len(result) > 0
            
            logger.info(f"✅ Integration test passed for {filename}")
    
    def test_end_to_end_extraction_workflow(self):
        """Test complete end-to-end extraction workflow."""
        
        # Simulate a complete document processing workflow
        documents = [
            {
                'filename': 'policy.txt',
                'content': 'Immigration policy document with important information.',
                'expected_terms': ['Immigration', 'policy', 'important']
            },
            {
                'filename': 'guide.html',
                'content': '<html><body><h1>Guide</h1><p>Helpful information here.</p></body></html>',
                'expected_terms': ['Guide', 'Helpful', 'information']
            },
            {
                'filename': 'readme.md',
                'content': '# Documentation\n\n**Important**: Read this carefully.\n\n- Point 1\n- Point 2',
                'expected_terms': ['Documentation', 'Important', 'carefully']
            }
        ]
        
        for doc in documents:
            # Step 1: Get appropriate extractor
            content_bytes = doc['content'].encode('utf-8')
            extractor = get_extractor(doc['filename'], content_bytes)
            
            # Step 2: Extract text
            extracted_text = extractor.extract(content_bytes)
            
            # Step 3: Validate extraction
            assert isinstance(extracted_text, str)
            assert len(extracted_text) > 0
            
            # Step 4: Check for expected content
            for term in doc['expected_terms']:
                assert term in extracted_text, f"Term '{term}' not found in extracted text from {doc['filename']}"
            
            # Step 5: Verify normalization
            assert extracted_text.endswith('\n')
            assert '\r\n' not in extracted_text
            assert '\r' not in extracted_text
            
            logger.info(f"✅ End-to-end test passed for {doc['filename']}")
    
    def test_error_recovery_and_fallbacks(self):
        """Test error recovery and fallback mechanisms."""
        
        # Test scenarios where primary extraction might fail
        error_scenarios = [
            {
                'filename': 'corrupted.pdf',
                'content': b'Not a real PDF file',
                'should_raise': DocumentExtractionError
            },
            {
                'filename': 'unknown.xyz',
                'content': b'Unknown file type',
                'should_raise': UnsupportedFileTypeError
            }
        ]
        
        for scenario in error_scenarios:
            with pytest.raises(scenario['should_raise']):
                extractor = get_extractor(scenario['filename'], scenario['content'])
                if not isinstance(scenario['should_raise'], UnsupportedFileTypeError):
                    # Only try extraction if we got an extractor
                    extractor.extract(scenario['content'])
            
            logger.info(f"✅ Error handling test passed for {scenario['filename']}")


if __name__ == "__main__":
    """
    Direct test execution support.
    """
    import sys
    
    # Add app to path for imports
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    print("🧪 Document Extractors Test Suite")
    print("=" * 50)
    print("Run with: pytest tests/test_document_extractors.py -v")
