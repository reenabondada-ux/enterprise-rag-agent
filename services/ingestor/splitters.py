# services/ingestor/splitters.py
import re
import tiktoken
from typing import List
from services.core.config import EMBEDDING_MODEL


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


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using a simple regex heuristic."""
    if not text:
        return []
    # Split on ., !, ? followed by whitespace. Keeps punctuation.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def sentence_split(
    text: str,
    max_chars: int = 1000,
    overlap_sentences: int = 1,
) -> List[str]:
    """
    Sentence-aware splitter.
    Packs sentences into chunks up to max_chars and overlaps by sentences.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If adding this sentence would exceed max_chars, flush current chunk.
        if current and current_len + sentence_len + 1 > max_chars:
            chunks.append(" ".join(current))

            # Start new chunk with overlap sentences.
            if overlap_sentences > 0:
                overlap = current[-overlap_sentences:]
                current = overlap.copy()
                current_len = sum(len(s) for s in current) + max(len(current) - 1, 0)
            else:
                current = []
                current_len = 0

        current.append(sentence)
        current_len += sentence_len + (1 if current_len > 0 else 0)

    if current:
        chunks.append(" ".join(current))

    return chunks


# For future: implement sentence_split with token counting using tiktoken for
# model-accurate splits. Pick max_tokens based on:
# (a) the embedding model’s input limit (typically 2048–4096 tokens) and your
#     desired chunk size (e.g. 200–300 tokens), and
# (b) how much context you want per chunk for retrieval (e.g. 2–5 sentences).
#     Note: 1 token ≈ 0.75 words in English, so 200 tokens ≈ 150 words.

# How to decide precisely:

# Use tiktoken to estimate token counts per sentence.
# Choose a max that keeps chunks 2–6 sentences on average.
# Validate with retrieval quality and LLM answer quality.
# For the current sample docs (short, doc‑style text), ~200–250 tokens is a good
# starting point.


def sentence_token_split(
    text: str,
    max_tokens: int = 200,
    overlap_sentences: int = 1,
    model: str = EMBEDDING_MODEL,
) -> List[str]:
    """
    Sentence-aware splitter with token limits (model-accurate).
    Uses tiktoken to count tokens and keeps sentence boundaries.
    """
    if not text:
        return []
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be >= 0")

    try:
        # Get model-specific encoding for accurate token counting
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        # Or use a fallback encoding (used by many models)
        encoding = tiktoken.get_encoding("cl100k_base")

    sentences = _split_sentences(text)
    if not sentences:
        return []

    # Precompute token counts for each sentence.
    # encoding.encode(s) returns list of token IDs. So len() gives token count
    sentence_tokens = [len(encoding.encode(s)) for s in sentences]

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for sentence, token_count in zip(sentences, sentence_tokens):
        if current and current_tokens + token_count > max_tokens:
            chunks.append(" ".join(current))

            if overlap_sentences > 0:
                overlap = current[-overlap_sentences:]
                current = overlap.copy()
                current_tokens = sum(len(encoding.encode(s)) for s in current)
            else:
                current = []
                current_tokens = 0

        current.append(sentence)
        current_tokens += token_count

    if current:
        chunks.append(" ".join(current))

    return chunks
