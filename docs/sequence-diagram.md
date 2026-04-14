# Enterprise RAG Agent - Sequence Diagram

This diagram shows the step-by-step flow for the `/incident-answer` endpoint.

```mermaid
sequenceDiagram
    actor User
    participant API as FastAPI<br/>/incident-answer
    participant Embedder as Sentence<br/>Transformers
    participant FAISS as FAISS<br/>Index
    participant DB as PostgreSQL<br/>Database
    participant Reranker as Cross-Encoder<br/>Reranker
    participant LLM as LLM Service<br/>(Ollama/OpenAI)
    participant Enricher as Response<br/>Enricher
    
    User->>API: POST /incident-answer<br/>{queryText, top_k}
    
    Note over API: Start Query Processing
    
    API->>Embedder: embed_query_text(queryText)
    Embedder->>Embedder: Generate embedding vector
    Embedder-->>API: query_embedding [768-dim]
    
    API->>API: normalize_query_vector()
    
    API->>FAISS: search_index(embedding, top_k)
    FAISS->>FAISS: Similarity search
    FAISS-->>API: [doc_ids, scores]
    
    API->>DB: SELECT * FROM documents<br/>WHERE id IN (ids)
    DB-->>API: [doc chunks with text]
    
    alt Reranker Enabled
        API->>Reranker: rerank_rows(query, rows, top_k)
        Reranker->>Reranker: Cross-encoder scoring
        Reranker-->>API: Re-ranked rows
    end
    
    API->>API: _build_context(rows, max_chars)
    Note over API: Concatenate top chunks<br/>within token limit
    
    API->>LLM: generate_llm_answer(query, context)
    LLM->>LLM: Generate grounded answer
    LLM-->>API: answer_text
    
    API->>Enricher: enrich_answer(query, answer, sources)
    Enricher->>Enricher: compute_confidence(sources)
    Enricher->>Enricher: infer_recommended_actions(query, answer)
    Enricher-->>API: enriched_response
    
    Note over API: Enriched Response:<br/>- query<br/>- summary<br/>- sources<br/>- confidence<br/>- recommended_actions
    
    API-->>User: JSON Response
```

## Flow Description

### 1. Query Reception
- User sends incident query to `/incident-answer` endpoint
- Includes query text and optional `top_k` parameter

### 2. Query Embedding
- Convert query text to vector using sentence-transformers
- Normalize the embedding vector for cosine similarity

### 3. Vector Search
- Search FAISS index for top-k similar document chunks
- Returns document IDs and similarity scores

### 4. Document Retrieval
- Fetch full document chunks from PostgreSQL by IDs
- Preserves order from FAISS results

### 5. Reranking (Optional)
- If enabled, use cross-encoder to re-score query-document pairs
- More accurate but slower than initial retrieval

### 6. Context Building
- Concatenate top-ranked chunks into context string
- Respects maximum character limit to fit LLM context window

### 7. Answer Generation
- Send query + context to LLM (Ollama or OpenAI)
- LLM generates grounded answer based on retrieved context

### 8. Response Enrichment
- **Confidence Score**: Based on top similarity score
- **Recommended Actions**: Extracted from answer or generated via heuristics
  - For timeout incidents: retry, check health, increase timeout
  - For file download failures: verify file, retry download
  - Generic fallbacks for other cases

### 9. Return Response
- Return enriched JSON with all metadata to user
- User receives actionable incident response guidance

## Timing Considerations

- **Fast Path** (FAISS only): ~100-300ms
- **With Reranker**: +200-500ms additional
- **LLM Generation**: +1-5s depending on provider and model
- **Total**: ~1-6s end-to-end
