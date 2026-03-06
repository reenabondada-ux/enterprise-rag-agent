# Enterprise RAG Agent

Enterprise-Grade RAG Knowledge Agent — starter repo.

## Goal
A production-minded RAG agent that ingests documents, stores embeddings in pgvector (Postgres), and exposes a query API for semantic search + grounding.

## Quick start (local MVP)

1. Start local infra (Postgres + Redis) from the **infra** folder:   
    ```bash   
    cd <project-root-directory>/infra
    docker compose up -d
    ```

2. Create/activate the **root** virtual environment and ingest sample documents:   
    ```bash   
    cd <project-root-directory>
    python -m venv .venv
    source .venv/bin/activate
    pip install -r services/ingestor/requirements.txt
    PYTHONPATH=. python services/ingestor/ingest.py
    ```

3. Run the API server from the **repo root** with PYTHONPATH set:   
    ```bash   
    cd <project-root-directory>
    PYTHONPATH=. uvicorn services.api.app.main:app --reload --port 8000
    ```
    
    Visit `http://localhost:8000/docs` for interactive API documentation.
