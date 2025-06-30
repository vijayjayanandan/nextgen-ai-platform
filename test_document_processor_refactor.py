"""
Document Processor Refactor Validation Script

This script validates that the document processor has been successfully refactored
to use the unified extractor system instead of legacy format-specific methods.
"""

import asyncio
import os
import sys
import logging
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_document_processor_refactor():
    """
    Test the refactored document processor functionality.
    """
    
    print("🔧 Document Processor Refactor Validation")
    print("=" * 50)
    
    # Step 1: Test imports
    print("\n📋 Step 1: Import Validation")
    print("-" * 30)
    
    try:
        from app.services.retrieval.document_processor import DocumentProcessor
        from app.services.retrieval.document_extractors import get_extractor, DocumentExtractionError
        print("✅ Refactored DocumentProcessor imports successfully")
        print("✅ Document extractors integration confirmed")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Step 2: Verify method signatures
    print("\n🔍 Step 2: Method Signature Validation")
    print("-" * 40)
    
    try:
        # Check that the new method exists
        assert hasattr(DocumentProcessor, '_extract_content_with_extractor')
        print("✅ _extract_content_with_extractor method exists")
        
        # Check that obsolete methods are removed
        obsolete_methods = [
            '_read_text_content',
            '_read_pdf_content', 
            '_extract_pdf_with_pymupdf',
            '_extract_pdf_with_pdfminer'
        ]
        
        for method_name in obsolete_methods:
            if hasattr(DocumentProcessor, method_name):
                print(f"❌ Obsolete method {method_name} still exists")
                return False
            else:
                print(f"✅ Obsolete method {method_name} successfully removed")
        
        # Check that _normalize_text_content is still available (as required)
        assert hasattr(DocumentProcessor, '_normalize_text_content')
        print("✅ _normalize_text_content method preserved for fallback use")
        
    except Exception as e:
        print(f"❌ Method signature validation failed: {e}")
        return False
    
    # Step 3: Test unified extraction logic
    print("\n🏭 Step 3: Unified Extraction Logic Validation")
    print("-" * 45)
    
    try:
        # Create a mock document class for testing
        class MockDocument:
            def __init__(self, filename, storage_path, document_id="test-doc-123"):
                self.filename = filename
                self.storage_path = storage_path
                self.id = document_id
        
        # Create test files
        test_files = {}
        
        # Create a temporary directory for test files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test text file
            txt_path = os.path.join(temp_dir, "test.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("Canadian Immigration Guide 2024\n\nThis is a test document for validation.")
            test_files['txt'] = txt_path
            
            # Test HTML file
            html_path = os.path.join(temp_dir, "test.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write("""<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body>
    <h1>Canadian Immigration Guide 2024</h1>
    <p>This is a test document for validation.</p>
</body>
</html>""")
            test_files['html'] = html_path
            
            # Test Markdown file
            md_path = os.path.join(temp_dir, "test.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("""# Canadian Immigration Guide 2024

## Overview

This is a test document for validation.

**Important**: This tests the unified extractor system.
""")
            test_files['md'] = md_path
            
            # Test the extraction logic directly
            # Note: We can't fully test without a database session, but we can test the extraction method
            
            # Create a minimal processor instance (without DB for testing)
            class TestDocumentProcessor(DocumentProcessor):
                def __init__(self):
                    # Skip the parent __init__ for testing
                    pass
            
            processor = TestDocumentProcessor()
            
            # Test each file type
            for file_type, file_path in test_files.items():
                try:
                    mock_doc = MockDocument(f"test.{file_type}", file_path)
                    
                    # Test the unified extraction method
                    content = await processor._extract_content_with_extractor(mock_doc)
                    
                    assert content is not None
                    assert isinstance(content, str)
                    assert len(content) > 0
                    assert "Canadian Immigration Guide 2024" in content
                    
                    print(f"✅ Unified extraction test passed for {file_type.upper()} format")
                    
                except Exception as e:
                    print(f"❌ Unified extraction test failed for {file_type.upper()}: {e}")
                    return False
        
    except Exception as e:
        print(f"❌ Unified extraction logic validation failed: {e}")
        return False
    
    # Step 4: Test error handling
    print("\n⚠️  Step 4: Error Handling Validation")
    print("-" * 35)
    
    try:
        class TestDocumentProcessor(DocumentProcessor):
            def __init__(self):
                pass
        
        processor = TestDocumentProcessor()
        
        # Test with non-existent file
        mock_doc = MockDocument("nonexistent.txt", "/path/that/does/not/exist")
        content = await processor._extract_content_with_extractor(mock_doc)
        
        # Should return None for non-existent files
        assert content is None
        print("✅ Error handling test passed for non-existent file")
        
        # Test with unsupported file type
        with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w", delete=False) as f:
            f.write("unsupported content")
            f.flush()
            
            try:
                mock_doc = MockDocument("test.xyz", f.name)
                content = await processor._extract_content_with_extractor(mock_doc)
                
                # Should return None for unsupported file types
                assert content is None
                print("✅ Error handling test passed for unsupported file type")
                
            finally:
                os.unlink(f.name)
        
    except Exception as e:
        print(f"❌ Error handling validation failed: {e}")
        return False
    
    # Step 5: Verify integration points
    print("\n🔌 Step 5: Integration Points Validation")
    print("-" * 40)
    
    try:
        # Check that _read_document_content now uses the unified system
        import inspect
        
        # Get the source code of _read_document_content
        source = inspect.getsource(DocumentProcessor._read_document_content)
        
        # Verify it calls the unified extractor
        assert "_extract_content_with_extractor" in source
        print("✅ _read_document_content uses unified extractor system")
        
        # Verify it doesn't contain legacy format-specific logic
        legacy_patterns = [
            "content_type == \"application/pdf\"",
            "content_type == \"text/plain\"",
            "_read_pdf_content",
            "_read_text_content"
        ]
        
        for pattern in legacy_patterns:
            if pattern in source:
                print(f"❌ Legacy pattern '{pattern}' still found in _read_document_content")
                return False
        
        print("✅ Legacy format-specific logic successfully removed")
        
    except Exception as e:
        print(f"❌ Integration points validation failed: {e}")
        return False
    
    # Step 6: Summary
    print("\n📊 Step 6: Refactor Summary")
    print("-" * 30)
    
    print("✅ Document Processor Refactor Successfully Completed!")
    print()
    print("Key Changes Validated:")
    print("• Added import for unified document extractors")
    print("• Implemented _extract_content_with_extractor method")
    print("• Replaced format-specific logic with unified extraction")
    print("• Removed obsolete extraction methods")
    print("• Preserved _normalize_text_content for fallback use")
    print("• Maintained robust error handling")
    print()
    print("Benefits Achieved:")
    print("• Format-agnostic document processing")
    print("• Centralized extraction logic")
    print("• Reduced code duplication")
    print("• Improved maintainability")
    print("• Future extensibility for new formats")
    print("• Consistent text normalization")
    
    return True

def main():
    """Main entry point for the validation script"""
    try:
        success = asyncio.run(test_document_processor_refactor())
        
        if success:
            print("\n🎉 Document Processor Refactor Validation Completed Successfully!")
            print("The refactored system is ready for production use.")
            return 0
        else:
            print("\n❌ Document Processor Refactor Validation Failed!")
            print("Please check the error messages above and resolve issues.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Validation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
