#!/usr/bin/env python3
"""
Simple markdown to PDF converter with basic formatting.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.colors import HexColor
import re


def clean_text(text):
    """Remove markdown formatting."""
    text = text.replace('**', '')
    text = text.replace('*', '')
    text = text.replace('`', '')
    text = text.replace('#', '')
    text = text.replace('- ', '• ')
    return text


def convert_md_to_pdf(md_file, pdf_file):
    """Convert simple markdown to PDF."""

    with open(md_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    doc = SimpleDocTemplate(
        pdf_file,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=22,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=HexColor('#1a1a2e')
    )

    h1_style = ParagraphStyle(
        'CustomH1',
        parent=styles['Heading1'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=12,
        textColor=HexColor('#16213e'),
    )

    h2_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Heading2'],
        fontSize=13,
        spaceBefore=15,
        spaceAfter=10,
        textColor=HexColor('#0f3460')
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        leading=14
    )

    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=8,
        spaceAfter=10,
        leftIndent=30,
        rightIndent=30,
        backgroundColor=HexColor('#f0f0f0')
    )

    story = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if not line.strip():
            story.append(Spacer(1, 6))
            i += 1
            continue

        # Title #
        if line.startswith('# ') and i < 5:
            text = clean_text(line[2:])
            story.append(Paragraph(text, title_style))
            story.append(Spacer(1, 20))

        # H1 ##
        elif line.startswith('## '):
            text = clean_text(line[3:])
            story.append(Paragraph(text, h1_style))

        # H2 ###
        elif line.startswith('### '):
            text = clean_text(line[4:])
            story.append(Paragraph(text, h2_style))

        # Code block
        elif line.strip().startswith('```'):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(clean_text(lines[i]))
                i += 1
            if code_lines:
                code_text = '<br/>'.join(code_lines)
                story.append(Paragraph(f'<font face="Courier">{code_text}</font>', code_style))

        # List item
        elif line.startswith('- ') or line.startswith('* '):
            text = '• ' + clean_text(line[2:])
            story.append(Paragraph(text, body_style))

        # Table row
        elif '|' in line and i > 0 and '|' in lines[i-1]:
            # Skip table formatting
            i += 1
            continue

        # Regular paragraph
        else:
            text = clean_text(line)
            if text:
                story.append(Paragraph(text, body_style))

        i += 1

    doc.build(story)
    return str(pdf_file)


if __name__ == '__main__':
    md_file = r'C:\Users\DELL\Documents\ML\rul-prediction-metaheuristic\docs\proyecto_rul_paper.md'
    pdf_file = r'C:\Users\DELL\Documents\ML\rul-prediction-metaheuristic\docs\proyecto_rul_paper.pdf'

    result = convert_md_to_pdf(md_file, pdf_file)
    print(f"PDF created: {result}")