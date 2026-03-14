# services/ingestor/embedders.py
"""
Embedding adapter with OpenAI as default provider.
Replace with Anthropic or other provider as needed.
"""

import time
from openai import OpenAI
from dotenv import load_dotenv
from services.core.config import OPENAI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM

load_dotenv()

_client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=15.0,  # request timeout
    max_retries=2,  # SDK-level retries
)


def embed_query_text(text: str) -> list[float]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing")

    # app-level retries (for transient failures)
    last_exception = None
    for attempt in range(3):
        try:
            response = _client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[text],
            )
            vector = response.data[0].embedding
            if len(vector) != EMBEDDING_DIM:
                raise ValueError(
                    "Embedding dimension mismatch: "
                    f"got {len(vector)}, expected {EMBEDDING_DIM}"
                )
            return vector
        except Exception as exception:
            last_exception = exception
            if attempt < 2:
                time.sleep(0.5 * (2**attempt))  # 0.5s, 1.0s
    raise RuntimeError(f"Embedding generation failed: {last_exception}")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Batch embedding function for multiple texts.
    Returns list[list[float]] with dimension EMBEDDING_DIM.
    Raises RuntimeError if API call fails.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing")

    # app-level retries (for transient failures)
    last_exception = None
    for attempt in range(3):
        try:
            response = _client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
            )
            vectors = [r.embedding for r in response.data]
            for i, vector in enumerate(vectors):
                if len(vector) != EMBEDDING_DIM:
                    raise ValueError(
                        "Embedding dimension mismatch for text "
                        f"{i}: got {len(vector)}, expected {EMBEDDING_DIM}"
                    )
            return vectors
        except Exception as exception:
            last_exception = exception
            if attempt < 2:
                time.sleep(0.5 * (2**attempt))  # 0.5s, 1.0s
    raise RuntimeError(f"Batch embedding generation failed: {last_exception}")
