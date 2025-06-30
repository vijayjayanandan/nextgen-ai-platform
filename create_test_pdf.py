"""
Script to create a test PDF for PDF processing validation.
"""

def create_test_pdf():
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        
        # Create PDF document
        doc = SimpleDocTemplate("tests/resources/sample_immigration_guide.pdf", pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
        )
        
        # Content for the PDF
        story = []
        
        # Page 1
        story.append(Paragraph("Canadian Immigration Guide 2024", title_style))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph("Chapter 1: Citizenship Requirements", heading_style))
        story.append(Paragraph("""
        To become a Canadian citizen, applicants must meet several key requirements:
        
        1. <b>Language Proficiency:</b> Demonstrate adequate knowledge of English or French. 
        Accepted tests include CELPIP (Canadian English Language Proficiency Index Program) 
        and IELTS General Training. Minimum scores of CLB 4 (Canadian Language Benchmark) 
        are required for speaking and listening.
        
        2. <b>Residence Requirements:</b> Must have been physically present in Canada for 
        at least 1,095 days (3 years) during the 5 years immediately before applying.
        
        3. <b>Tax Obligations:</b> Must have filed income tax returns for at least 3 years 
        during the 5-year period, if required to do so under the Income Tax Act.
        """, styles['Normal']))
        
        story.append(PageBreak())
        
        # Page 2
        story.append(Paragraph("Chapter 2: Application Process and Fees", heading_style))
        story.append(Paragraph("""
        The citizenship application process involves several steps:
        
        <b>Application Fee:</b> The current fee for adult citizenship applications is $630 CAD. 
        This includes the processing fee and the right of citizenship fee.
        
        <b>Processing Times:</b> Standard processing time is approximately 18 months from 
        the date we receive a complete application. However, processing times may vary 
        depending on individual circumstances.
        
        <b>Expedited Processing:</b> In exceptional circumstances, applications may be 
        processed on an expedited basis for humanitarian grounds only. Regular requests 
        for faster processing are not accepted.
        
        <b>Required Documents:</b> Applicants must provide:
        - Valid passport or travel document
        - Proof of residence (rental agreements, utility bills, etc.)
        - Tax documents (Notice of Assessment for each year)
        - Language test results (valid for 2 years from test date)
        """, styles['Normal']))
        
        story.append(PageBreak())
        
        # Page 3
        story.append(Paragraph("Chapter 3: Express Entry vs Provincial Nominee Programs", heading_style))
        story.append(Paragraph("""
        Canada offers multiple immigration pathways for permanent residence:
        
        <b>Express Entry System:</b>
        - Average processing time: 6 months
        - Requires Comprehensive Ranking System (CRS) score
        - Minimum language requirement: CLB 7 for speaking and listening
        - Covers Federal Skilled Worker, Canadian Experience Class, and Federal Skilled Trades programs
        
        <b>Provincial Nominee Program (PNP):</b>
        - Average processing time: 18 months (including federal processing)
        - Requires provincial nomination from participating province/territory
        - Minimum language requirement: CLB 4 for speaking and listening
        - Each province has specific criteria and occupation lists
        
        <b>Key Differences:</b>
        The Express Entry system is faster but more competitive, requiring higher language 
        scores and CRS points. Provincial Nominee Programs take longer but may be more 
        accessible for candidates with specific skills needed by provinces.
        """, styles['Normal']))
        
        story.append(Spacer(1, 20))
        story.append(Paragraph("<b>Important Note:</b> This guide provides general information only. " +
                              "Immigration laws and requirements change frequently. Always consult " +
                              "the official Government of Canada website or a qualified immigration " +
                              "consultant for the most current information.", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        print("✅ Test PDF created successfully: tests/resources/sample_immigration_guide.pdf")
        return True
        
    except ImportError:
        print("❌ reportlab not available. Creating a simple text-based PDF alternative...")
        
        # Create a simple text file as fallback
        content = """Canadian Immigration Guide 2024

Chapter 1: Citizenship Requirements

To become a Canadian citizen, applicants must meet several key requirements:

1. Language Proficiency: Demonstrate adequate knowledge of English or French. 
Accepted tests include CELPIP (Canadian English Language Proficiency Index Program) 
and IELTS General Training. Minimum scores of CLB 4 (Canadian Language Benchmark) 
are required for speaking and listening.

2. Residence Requirements: Must have been physically present in Canada for 
at least 1,095 days (3 years) during the 5 years immediately before applying.

3. Tax Obligations: Must have filed income tax returns for at least 3 years 
during the 5-year period, if required to do so under the Income Tax Act.

Chapter 2: Application Process and Fees

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
- Language test results (valid for 2 years from test date)

Chapter 3: Express Entry vs Provincial Nominee Programs

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
website or a qualified immigration consultant for the most current information.
"""
        
        with open("tests/resources/sample_immigration_guide.txt", "w", encoding="utf-8") as f:
            f.write(content)
        
        print("✅ Created text file as PDF alternative: tests/resources/sample_immigration_guide.txt")
        return False

if __name__ == "__main__":
    create_test_pdf()
