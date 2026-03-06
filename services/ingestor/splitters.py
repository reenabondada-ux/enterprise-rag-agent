# services/ingestor/splitters.py
from typing import List


def fixed_split(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Simple fixed-size splitter with overlap.
    Returns list of text chunks.
    """
    chunks = []
    i = 0
    text_length = len(text)

    while i < text_length:
        start = max(i - overlap, 0)
        end = min(i + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks
