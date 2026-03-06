# services.shared package
from .config import (
    OPENAI_API_KEY,
    DATABASE_URL,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    CHAT_MODEL,
)
from .embeddings import embed_query_text, embed_texts
from .llm import generate_answer

__all__ = [
    "OPENAI_API_KEY",
    "DATABASE_URL",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "CHAT_MODEL",
    "embed_query_text",
    "embed_texts",
    "generate_answer",
]
