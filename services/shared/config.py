# services/shared/config.py
"""
Centralized configuration for all services.
Loads from environment variables with sensible defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Database
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://rag_user:rag_pass@localhost:5432/rag_db"
)

# Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# OpenAI / Embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# FAISS index
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss/index.bin")


# Optional: Validation
def validate_config():
    """Raise if critical config is missing."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY must be set")
    if EMBEDDING_DIM <= 0:
        raise ValueError(f"Invalid EMBEDDING_DIM: {EMBEDDING_DIM}")
