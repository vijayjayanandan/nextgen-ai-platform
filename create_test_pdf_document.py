"""
Create a test PDF document for format validation testing.
"""

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    
    def create_citizenship_pdf():
        """Create a PDF document about Canadian citizenship requirements"""
        
        filename = "test_documents/citizenship_requirements.pdf"
        
        # Create the PDF document
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title = Paragraph("Canadian Citizenship Requirements 2024", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Physical presence requirement
        heading1 = Paragraph("Physical Presence Requirement for Canadian Citizenship", styles['Heading1'])
        story.append(heading1)
        
        content1 = Paragraph(
            "To be eligible for Canadian citizenship, you must have been physically present in Canada "
            "for at least 1,095 days (3 years) during the 5 years immediately before the date of your application. "
            "This requirement ensures that new citizens have sufficient connection to Canada.",
            styles['Normal']
        )
        story.append(content1)
        story.append(Spacer(1, 12))
        
        # Language requirements
        heading2 = Paragraph("Language Requirements", styles['Heading1'])
        story.append(heading2)
        
        content2 = Paragraph(
            "You must demonstrate adequate knowledge of English or French through approved language tests "
            "such as CELPIP, IELTS, or TEF. This ensures you can participate fully in Canadian society.",
            styles['Normal']
        )
        story.append(content2)
        story.append(Spacer(1, 12))
        
        # Tax obligations
        heading3 = Paragraph("Tax Filing Obligations", styles['Heading1'])
        story.append(heading3)
        
        content3 = Paragraph(
            "You must file income tax returns for at least 3 years during the 5-year period "
            "if you are required to do so under the Income Tax Act.",
            styles['Normal']
        )
        story.append(content3)
        story.append(Spacer(1, 12))
        
        # Processing times
        heading4 = Paragraph("Processing Times", styles['Heading1'])
        story.append(heading4)
        
        content4 = Paragraph(
            "Current processing times for citizenship applications are approximately 12-18 months "
            "from the date we receive your complete application.",
            styles['Normal']
        )
        story.append(content4)
        
        # Build the PDF
        doc.build(story)
        print(f"✅ Created PDF: {filename}")
        return filename
    
    if __name__ == "__main__":
        create_citizenship_pdf()
        
except ImportError:
    print("❌ ReportLab not available. Creating simple PDF with alternative method...")
    
    # Alternative method using fpdf if available
    try:
        from fpdf import FPDF
        
        def create_simple_pdf():
            """Create a simple PDF using fpdf"""
            
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            
            # Title
            pdf.cell(0, 10, 'Canadian Citizenship Requirements 2024', 0, 1, 'C')
            pdf.ln(10)
            
            # Content
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Physical Presence Requirement for Canadian Citizenship', 0, 1)
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 5, 'To be eligible for Canadian citizenship, you must have been physically present in Canada for at least 1,095 days (3 years) during the 5 years immediately before the date of your application.')
            pdf.ln(5)
            
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Language Requirements', 0, 1)
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 5, 'You must demonstrate adequate knowledge of English or French through approved language tests such as CELPIP, IELTS, or TEF.')
            pdf.ln(5)
            
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Tax Filing Obligations', 0, 1)
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 5, 'You must file income tax returns for at least 3 years during the 5-year period if you are required to do so under the Income Tax Act.')
            pdf.ln(5)
            
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Processing Times', 0, 1)
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 5, 'Current processing times for citizenship applications are approximately 12-18 months from the date we receive your complete application.')
            
            filename = "test_documents/citizenship_requirements.pdf"
            pdf.output(filename)
            print(f"✅ Created PDF: {filename}")
            return filename
        
        if __name__ == "__main__":
            create_simple_pdf()
            
    except ImportError:
        print("❌ Neither ReportLab nor fpdf available. Please install one of them:")
        print("pip install reportlab")
        print("or")
        print("pip install fpdf2")
        
        # Create a simple text file as fallback
        content = """Canadian Citizenship Requirements 2024

Physical Presence Requirement for Canadian Citizenship:
To be eligible for Canadian citizenship, you must have been physically present in Canada for at least 1,095 days (3 years) during the 5 years immediately before the date of your application.

Language Requirements:
You must demonstrate adequate knowledge of English or French through approved language tests such as CELPIP, IELTS, or TEF.

Tax Filing Obligations:
You must file income tax returns for at least 3 years during the 5-year period if you are required to do so under the Income Tax Act.

Processing Times:
Current processing times for citizenship applications are approximately 12-18 months from the date we receive your complete application.
"""
        
        with open("test_documents/citizenship_requirements.txt", "w") as f:
            f.write(content)
        print("✅ Created fallback text file: test_documents/citizenship_requirements.txt")
