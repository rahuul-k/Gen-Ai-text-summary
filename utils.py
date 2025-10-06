import io
from typing import List
from PyPDF2 import PdfReader

def read_txt(file_like) -> str:
    content = file_like.read()
    if isinstance(content, bytes):
        content = content.decode('utf-8', errors='ignore')
    return content

def read_pdf(file_like) -> str:
    reader = PdfReader(file_like)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

def chunk_text(text: str, max_chars: int = 3000) -> List[str]:
    """
    Simple chunker that splits text into overlapping chunks not exceeding max_chars.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    overlap = 200
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks
