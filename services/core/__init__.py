# services.core package
from .config import (
    OPENAI_API_KEY,
    DATABASE_URL,
    EMBEDDING_DIM,
    LOCAL_EMBEDDING_MODEL_SBERT,
    CHAT_MODEL,
    FAISS_INDEX_PATH,
)

__all__ = [
    "OPENAI_API_KEY",
    "DATABASE_URL",
    "EMBEDDING_DIM",
    "LOCAL_EMBEDDING_MODEL_SBERT",
    "CHAT_MODEL",
    "FAISS_INDEX_PATH",
]
