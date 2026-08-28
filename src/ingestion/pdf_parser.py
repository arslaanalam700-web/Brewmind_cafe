"""PDF extraction utility with page tracking, layout preservation, and section detection."""
import re
from pathlib import Path
from typing import List, Dict, Any

# Regex patterns for common academic paper section headings
SECTION_PATTERN = re.compile(
    r"^(?:\d+\.?\s+|[I|V|X]+\.?\s+)?(Abstract|Introduction|Related Work|Background|Methodology|Method|Formulation|Approach|Algorithm|Experiments|Results|Discussion|Limitations|Conclusion|References|Broader Impact|Ethics Statement)",
    re.IGNORECASE | re.MULTILINE
)

def extract_with_pymupdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract pages and layout blocks using PyMuPDF (fitz)."""
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    pages_data = []
    
    current_section = "Abstract"
    
    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        text = page.get_text("text")
        
        # Check if a new section heading appears on this page
        matches = SECTION_PATTERN.findall(text)
        if matches:
            current_section = matches[0].strip()
            
        pages_data.append({
            "page": page_num,
            "text": text.strip(),
            "detected_section": current_section
        })
        
    doc.close()
    return pages_data

def extract_with_pypdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """Fallback PDF extraction using pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    pages_data = []
    current_section = "Abstract"
    
    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        text = page.extract_text() or ""
        
        matches = SECTION_PATTERN.findall(text)
        if matches:
            current_section = matches[0].strip()
            
        pages_data.append({
            "page": page_num,
            "text": text.strip(),
            "detected_section": current_section
        })
        
    return pages_data

def extract_paper_pages(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract text from PDF pages with section headings and metadata."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
    try:
        return extract_with_pymupdf(pdf_path)
    except Exception as e:
        print(f"PyMuPDF failed on {pdf_path.name} ({e}), falling back to pypdf...")
        return extract_with_pypdf(pdf_path)
