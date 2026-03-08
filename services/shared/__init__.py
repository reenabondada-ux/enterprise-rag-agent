# services.shared package
from .config import (
    OPENAI_API_KEY,
    DATABASE_URL,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    FAISS_INDEX_PATH,
)
from .embeddings import embed_query_text, embed_texts
from .llm import generate_answer
from .faiss_index import (
    load_or_create_index,
    save_index,
    normalize_vectors,
    normalize_query_vector,
    add_embeddings,
    search_index,
)

__all__ = [
    "OPENAI_API_KEY",
    "DATABASE_URL",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "CHAT_MODEL",
    "FAISS_INDEX_PATH",
    "embed_query_text",
    "embed_texts",
    "generate_answer",
    "load_or_create_index",
    "save_index",
    "normalize_vectors",
    "normalize_query_vector",
    "add_embeddings",
    "search_index",
]
