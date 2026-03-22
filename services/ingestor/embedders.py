# services/ingestor/embedders.py
"""
Embedding adapter using a local SBERT model by default.
"""

from sentence_transformers import SentenceTransformer

from services.core.config import LOCAL_EMBEDDING_MODEL_SBERT

_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL_SBERT)


def embed_query_text(text: str) -> list[float]:
    vector = _model.encode(text, convert_to_numpy=True)
    return vector.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Batch embedding function for multiple texts.
    Returns list[list[float]] with the SBERT embedding dimension.
    """
    vectors = _model.encode(texts, convert_to_numpy=True)
    return vectors.tolist()
