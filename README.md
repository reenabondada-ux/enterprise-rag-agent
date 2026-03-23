# Enterprise RAG Agent

Enterprise-Grade RAG Knowledge Agent — starter repo.

## Goal
A production-minded RAG agent that ingests documents, stores embeddings in Postgres, builds a FAISS index for vector search, and exposes a query API for semantic search + grounding.

## Quick start (local MVP)

1. Start local infra (Postgres + Redis) from the **infra** folder:   
    ```bash   
    cd <project-root-directory>/infra
    docker compose up -d
    ```

2. Create/activate the **root** virtual environment and ingest sample documents:   
    Note: As a one time activity run the following command in the .venv virtual environment before executing the ingestor.
    ```
    python -c "import nltk; nltk.download('punkt_tab')"
    ```
    ```bash   
    cd <project-root-directory>
    python -m venv .venv
    source .venv/bin/activate
    pip install -r services/api/requirements.txt
    PYTHONPATH=. python services/ingestor/ingest.py
    ```

3. Run the API server from the **repo root** with PYTHONPATH set:   
    ```bash   
    cd <project-root-directory>
    export HF_TOKEN="your_token_here"
    PYTHONPATH=. python -m uvicorn services.api.app.main:app --reload --port 8000
    ```
    
    
        Visit `http://localhost:8000/docs` for interactive API documentation.

    To debug run:
    ```
    PYTHONPATH=. python -m uvicorn services.api.app.main:app --reload --port 8000 --log-level debug
    ```

## Chat model (Ollama)

Environment variables:

- `OLLAMA_CHAT_MODEL` (default: `llama3.1:8b`)
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
    
## FAISS

- FAISS index path (configurable via `.env`):
    - `FAISS_INDEX_PATH=data/faiss/index.bin`
- Index reload endpoint:
    - `POST /faiss/reload`
- Ingestor post-hook (optional):
    - `FAISS_RELOAD_URL=http://localhost:8000/faiss/reload`

The FAISS index is updated during ingestion and can be reloaded by the API on demand.
