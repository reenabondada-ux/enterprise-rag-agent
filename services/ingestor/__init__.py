# services.ingestor package
from .embedders import embed_query_text, embed_texts

__all__ = [
    "embed_query_text",
    "embed_texts",
]
