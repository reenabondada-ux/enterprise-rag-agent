# Enterprise RAG Agent - Architecture Diagram

This diagram shows the overall system components and their interactions.

```mermaid
graph TB
    Client[Client Application]
    
    subgraph "FastAPI Service"
        API[FastAPI Endpoints<br/>/query, /answer, /incident-answer]
        Enricher[Response Enricher<br/>Confidence & Actions]
    end
    
    subgraph "Vector Search"
        FAISS[FAISS Index<br/>In-Memory Vector Search]
        Postgres[(PostgreSQL<br/>Documents Table)]
    end
    
    subgraph "ML Services"
        Embedder[Sentence Transformers<br/>Query & Doc Embeddings]
        Reranker[Cross-Encoder Reranker<br/>Optional]
        LLM[LLM Service<br/>Ollama / OpenAI]
    end
    
    subgraph "Data Ingestion"
        Ingestor[Document Ingestor<br/>Chunking & Embedding]
        Docs[Source Documents<br/>Runbooks, Guides]
    end
    
    Client -->|HTTP Request| API
    API -->|1. Embed Query| Embedder
    API -->|2. Vector Search| FAISS
    FAISS -->|3. Fetch by IDs| Postgres
    API -->|4. Rerank Results| Reranker
    API -->|5. Generate Answer| LLM
    API -->|6. Enrich Response| Enricher
    Enricher -->|Response| Client
    
    Docs -->|Ingest| Ingestor
    Ingestor -->|Embed Chunks| Embedder
    Ingestor -->|Store Text| Postgres
    Ingestor -->|Build Index| FAISS
    Ingestor -.->|Reload| API
    
    style Client fill:#e1f5ff
    style API fill:#fff4e1
    style FAISS fill:#f0e1ff
    style Postgres fill:#f0e1ff
    style LLM fill:#e1ffe1
    style Enricher fill:#ffe1e1
```

## Components

### FastAPI Service
- **API Endpoints**: Handles `/query`, `/answer`, and `/incident-answer` requests
- **Response Enricher**: Adds confidence scores and recommended actions for incident response queries

### Vector Search
- **FAISS Index**: In-memory vector index for fast similarity search
- **PostgreSQL**: Persistent storage for document chunks with metadata

### ML Services
- **Embedder**: Sentence-transformers model for generating embeddings
- **Reranker**: Optional cross-encoder for improving retrieval quality
- **LLM**: Ollama (local) or OpenAI for answer generation

### Data Ingestion
- **Ingestor**: Processes documents, splits into chunks, generates embeddings
- **Source Documents**: Runbooks, troubleshooting guides, architecture docs

## Data Flow

1. **Ingestion** (offline): Documents → Chunking → Embedding → Storage (Postgres + FAISS)
2. **Query** (runtime): Query → Embedding → FAISS Search → Postgres Fetch → Rerank → LLM Generation → Enrichment → Response
