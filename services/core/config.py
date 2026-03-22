# services/core/config.py
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
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
LOCAL_EMBEDDING_MODEL_SBERT = os.getenv(
    "LOCAL_EMBEDDING_MODEL_SBERT", "all-MiniLM-L6-v2"
)
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# FAISS index
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss/index.bin")
