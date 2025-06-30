"""
Create a simple PDF using PyMuPDF for testing PDF processing.
"""

def create_simple_pdf():
    try:
        import fitz  # PyMuPDF
        
        # Create a new PDF document
        doc = fitz.open()
        
        # Page 1
        page1 = doc.new_page()
        text1 = """Canadian Immigration Guide 2024

Chapter 1: Citizenship Requirements

To become a Canadian citizen, applicants must meet several key requirements:

1. Language Proficiency: Demonstrate adequate knowledge of English or French. 
Accepted tests include CELPIP (Canadian English Language Proficiency Index Program) 
and IELTS General Training. Minimum scores of CLB 4 (Canadian Language Benchmark) 
are required for speaking and listening.

2. Residence Requirements: Must have been physically present in Canada for 
at least 1,095 days (3 years) during the 5 years immediately before applying.

3. Tax Obligations: Must have filed income tax returns for at least 3 years 
during the 5-year period, if required to do so under the Income Tax Act."""

        page1.insert_text((72, 72), text1, fontsize=11, fontname="helv")
        
        # Page 2
        page2 = doc.new_page()
        text2 = """Chapter 2: Application Process and Fees

The citizenship application process involves several steps:

Application Fee: The current fee for adult citizenship applications is $630 CAD. 
This includes the processing fee and the right of citizenship fee.

Processing Times: Standard processing time is approximately 18 months from 
the date we receive a complete application. However, processing times may vary 
depending on individual circumstances.

Expedited Processing: In exceptional circumstances, applications may be 
processed on an expedited basis for humanitarian grounds only. Regular requests 
for faster processing are not accepted.

Required Documents: Applicants must provide:
- Valid passport or travel document
- Proof of residence (rental agreements, utility bills, etc.)
- Tax documents (Notice of Assessment for each year)
- Language test results (valid for 2 years from test date)"""

        page2.insert_text((72, 72), text2, fontsize=11, fontname="helv")
        
        # Page 3
        page3 = doc.new_page()
        text3 = """Chapter 3: Express Entry vs Provincial Nominee Programs

Canada offers multiple immigration pathways for permanent residence:

Express Entry System:
- Average processing time: 6 months
- Requires Comprehensive Ranking System (CRS) score
- Minimum language requirement: CLB 7 for speaking and listening
- Covers Federal Skilled Worker, Canadian Experience Class, and Federal Skilled Trades programs

Provincial Nominee Program (PNP):
- Average processing time: 18 months (including federal processing)
- Requires provincial nomination from participating province/territory
- Minimum language requirement: CLB 4 for speaking and listening
- Each province has specific criteria and occupation lists

Key Differences:
The Express Entry system is faster but more competitive, requiring higher language 
scores and CRS points. Provincial Nominee Programs take longer but may be more 
accessible for candidates with specific skills needed by provinces.

Important Note: This guide provides general information only. Immigration laws and 
requirements change frequently. Always consult the official Government of Canada 
website or a qualified immigration consultant for the most current information."""

        page3.insert_text((72, 72), text3, fontsize=11, fontname="helv")
        
        # Save the PDF
        doc.save("tests/resources/sample_immigration_guide.pdf")
        doc.close()
        
        print("✅ Test PDF created successfully using PyMuPDF: tests/resources/sample_immigration_guide.pdf")
        return True
        
    except ImportError:
        print("❌ PyMuPDF not available. Using text file instead.")
        return False
    except Exception as e:
        print(f"❌ Error creating PDF: {e}")
        return False

if __name__ == "__main__":
    create_simple_pdf()
