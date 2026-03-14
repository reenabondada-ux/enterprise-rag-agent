# services.core package
from .config import (
    OPENAI_API_KEY,
    DATABASE_URL,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    FAISS_INDEX_PATH,
)

__all__ = [
    "OPENAI_API_KEY",
    "DATABASE_URL",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "CHAT_MODEL",
    "FAISS_INDEX_PATH",
]
