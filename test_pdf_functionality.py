"""
Simple test script to verify PDF processing functionality.
"""

import asyncio
import os
from pathlib import Path

async def test_pdf_extraction():
    """Test PDF text extraction functionality"""
    
    print("🧪 Testing PDF processing functionality...")
    
    # Test 1: Check if PDF libraries are available
    try:
        import fitz  # PyMuPDF
        print("✅ PyMuPDF available")
    except ImportError:
        print("❌ PyMuPDF not available")
        return False
    
    try:
        from pdfminer.high_level import extract_text
        print("✅ pdfminer.six available")
    except ImportError:
        print("❌ pdfminer.six not available")
        return False
    
    # Test 2: Check if test PDF exists
    test_pdf_path = "tests/resources/sample_immigration_guide.pdf"
    if not os.path.exists(test_pdf_path):
        print(f"❌ Test PDF not found: {test_pdf_path}")
        return False
    
    print(f"✅ Test PDF found: {test_pdf_path}")
    
    # Test 3: Extract text using PyMuPDF
    try:
        doc = fitz.open(test_pdf_path)
        text_content = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            if page_text.strip():
                text_content.append(f"[Page {page_num + 1}]\n{page_text}")
        
        doc.close()
        
        if text_content:
            full_text = "\n\n".join(text_content)
            print(f"✅ PyMuPDF extraction successful: {len(full_text)} characters")
            
            # Check for expected content
            expected_terms = ["Canadian Immigration Guide", "CELPIP", "CLB 4", "1,095 days"]
            found_terms = [term for term in expected_terms if term in full_text]
            print(f"✅ Found {len(found_terms)}/{len(expected_terms)} expected terms: {found_terms}")
        else:
            print("❌ No text extracted by PyMuPDF")
            return False
            
    except Exception as e:
        print(f"❌ PyMuPDF extraction failed: {e}")
        return False
    
    # Test 4: Extract text using pdfminer
    try:
        text_content = extract_text(test_pdf_path)
        if text_content and len(text_content.strip()) > 0:
            print(f"✅ pdfminer extraction successful: {len(text_content)} characters")
        else:
            print("❌ No text extracted by pdfminer")
            return False
    except Exception as e:
        print(f"❌ pdfminer extraction failed: {e}")
        return False
    
    # Test 5: Test DocumentProcessor import
    try:
        from app.services.retrieval.document_processor import DocumentProcessor
        print("✅ DocumentProcessor imported successfully")
    except Exception as e:
        print(f"❌ DocumentProcessor import failed: {e}")
        return False
    
    print("🎉 All PDF functionality tests passed!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_pdf_extraction())
    if success:
        print("\n✅ PDF processing is ready for integration testing")
    else:
        print("\n❌ PDF processing setup needs attention")
