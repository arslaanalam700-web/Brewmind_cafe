"""Structure-aware chunker producing deterministic chunk_ids and rich metadata."""
import re
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """Clean excess whitespace, strange control characters, and line breaks."""
    # Replace multiple newlines with a double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace hyphenated line breaks (e.g., 'opti-\nmization' -> 'optimization')
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Normalize excessive horizontal whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def chunk_paper(
    paper_id: str,
    paper_title: str,
    pages_data: List[Dict[str, Any]],
    chunk_size_words: int = 350,
    overlap_words: int = 60
) -> List[Dict[str, Any]]:
    """
    Split paper pages into structured chunks respecting section boundaries and pages.
    Generates deterministic chunk_id: {paper_id}:p{page:02d}_c{idx:02d}
    """
    chunks = []
    
    for page_info in pages_data:
        page_num = page_info["page"]
        section = page_info.get("detected_section", "General")
        raw_text = clean_text(page_info["text"])
        
        if not raw_text or len(raw_text) < 50:
            continue
            
        # Split text into paragraphs
        paragraphs = [p.strip() for p in raw_text.split('\n\n') if p.strip()]
        
        current_chunk_words: List[str] = []
        page_chunk_index = 1
        
        for para in paragraphs:
            para_words = para.split()
            
            # If adding this paragraph exceeds chunk size and we already have content
            if len(current_chunk_words) + len(para_words) > chunk_size_words and len(current_chunk_words) >= 100:
                # Emit current chunk
                chunk_text = " ".join(current_chunk_words)
                chunk_id = f"{paper_id}:p{page_num:02d}_c{page_chunk_index:02d}"
                
                chunks.append({
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "paper_title": paper_title,
                    "page": page_num,
                    "section": section,
                    "text": chunk_text,
                    "word_count": len(current_chunk_words)
                })
                
                page_chunk_index += 1
                # Overlap: keep the last `overlap_words`
                current_chunk_words = current_chunk_words[-overlap_words:] + para_words
            else:
                current_chunk_words.extend(para_words)
                
        # Emit any trailing text for the page
        if current_chunk_words and len(current_chunk_words) >= 30:
            chunk_text = " ".join(current_chunk_words)
            chunk_id = f"{paper_id}:p{page_num:02d}_c{page_chunk_index:02d}"
            chunks.append({
                "chunk_id": chunk_id,
                "paper_id": paper_id,
                "paper_title": paper_title,
                "page": page_num,
                "section": section,
                "text": chunk_text,
                "word_count": len(current_chunk_words)
            })
            
    return chunks
