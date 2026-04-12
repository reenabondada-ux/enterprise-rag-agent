# services/core/config.py
"""
Centralized configuration for all services.
Loads from environment variables with sensible defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Database
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in the environment or .env file")

# Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# OpenAI / Embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
LOCAL_EMBEDDING_MODEL_SBERT = os.getenv(
    "LOCAL_EMBEDDING_MODEL_SBERT", "all-MiniLM-L6-v2"
)

# Chat provider + models
CHAT_PROVIDER = os.getenv("CHAT_PROVIDER", "ollama")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    OPENAI_CHAT_MODEL if CHAT_PROVIDER == "openai" else OLLAMA_CHAT_MODEL,
)

# FAISS index
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss/index.bin")

# Reranker (optional)
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "5"))
