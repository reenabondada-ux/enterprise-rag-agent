# services/api

Minimal FastAPI service for Enterprise RAG Agent.

## Quick start (local)

1. Create `.env` from the repo root `.env.example` and ensure DATABASE_URL points to your local Postgres.
2. Install deps:   
    ```bash   
    cd <project-root-directory>/services/api   
    pip install -r requirements.txt
    ```

3. Run the API server from the **repo root**:   
    ```bash   
    cd <project-root-directory>
    PYTHONPATH=. python -m uvicorn services.api.app.main:app --reload --port 8000
    ```
    
    Then visit `http://localhost:8000/docs` for interactive API docs.

## FAISS

- FAISS index path (configurable via `.env`):
    - `FAISS_INDEX_PATH=data/faiss/index.bin`
- Reload FAISS index:
    - `POST /faiss/reload`
