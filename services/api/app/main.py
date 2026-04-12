# services/api/app/main.py
import logging
import psycopg2
import faiss
from threading import Lock
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from services.ingestor import embed_query_text
from services.core import DATABASE_URL, EMBEDDING_DIM, FAISS_INDEX_PATH
from services.core.llm import generate_llm_answer
from services.core.vector import (
    load_or_create_index,
    normalize_query_vector,
    search_index,
)
from services.core.reranker import rerank_rows
from starlette.concurrency import run_in_threadpool
from services.api.app.response_enricher import enrich_answer

load_dotenv()

logger = logging.getLogger("enterprise_rag_api")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

app = FastAPI(title="Enterprise RAG Agent - API")

FAISS_INDEX = None
FAISS_INDEX_LOCK = Lock()


class QueryRequest(BaseModel):
    queryText: str
    top_k: int = 5


class AnswerRequest(BaseModel):
    queryText: str
    top_k: int = 5
    max_context_chars: int = 4000


@app.on_event("startup")
def startup():
    # Basic DB connection test
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.close()
    except Exception as exception:
        logger.error(
            "db_connection_failed",
            extra={"error": str(exception)},
        )

    # Load FAISS index into memory for fast search
    _load_faiss_index()


def _load_faiss_index() -> None:
    global FAISS_INDEX
    with FAISS_INDEX_LOCK:
        FAISS_INDEX = load_or_create_index(
            EMBEDDING_DIM, FAISS_INDEX_PATH, DATABASE_URL
        )
        logger.info(
            "faiss_index_loaded",
            extra={"path": FAISS_INDEX_PATH},
        )


def _fetch_docs_by_ids(ids: list[int]) -> list[dict]:
    if not ids:
        return []
    connection = psycopg2.connect(DATABASE_URL)
    cursor = connection.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT id, title, text FROM documents WHERE id = ANY(%s)", (ids,))
    rows = cursor.fetchall()
    cursor.close()
    connection.close()
    logger.info("db_query_completed")
    row_map = {row["id"]: row for row in rows}
    return [row_map[i] for i in ids if i in row_map]


def _faiss_search(query_embedding: list[float], top_k: int) -> list[dict]:
    if FAISS_INDEX is None:
        _load_faiss_index()

    # Limit FAISS to 2 threads to prevent CPU contention and hanging of application
    faiss.omp_set_num_threads(2)
    logger.info("faiss_threads_set", extra={"threads": 2})

    ids, scores = search_index(FAISS_INDEX, query_embedding, top_k)
    # Filter out empty slots returned as -1
    filtered = [(i, s) for i, s in zip(ids, scores) if i != -1]
    if not filtered:
        logger.info("faiss_search_no_hits")
        return []
    filtered_ids, filtered_scores = zip(*filtered)
    logger.info("faiss_search_hits", extra={"hits": len(filtered_ids)})
    docs = _fetch_docs_by_ids(list(filtered_ids))
    similarity_map = {i: s for i, s in zip(filtered_ids, filtered_scores)}
    for doc in docs:
        doc["similarity"] = similarity_map.get(doc["id"])
    return docs


@app.post("/faiss/reload")
def reload_faiss_index():
    """Reload FAISS index from disk (for ingestion updates)."""
    _load_faiss_index()
    logger.info("faiss_index_reloaded", extra={"path": FAISS_INDEX_PATH})
    return {"status": "ok", "message": "FAISS index reloaded"}


@app.post("/query")
async def query(queryRequest: QueryRequest):
    """
    Minimal /query endpoint.
    - Intended to be replaced retrieval pipeline:
        * compute embedding for queryRequest.queryText
        * query documents using pgvector similarity operator (<->)
        * optionally re-rank with cross-encoder or LLM
    For now, returns top-k documents by similarity using a placeholder vector
    (so you can verify the DB + table wiring).
    """
    try:
        query_embedding = await run_in_threadpool(
            embed_query_text, queryRequest.queryText
        )

        query_embedding = normalize_query_vector(query_embedding)

        rows = _faiss_search(query_embedding, queryRequest.top_k)
        logger.info("faiss_search_completed")
        rows = await run_in_threadpool(
            rerank_rows, queryRequest.queryText, rows, queryRequest.top_k
        )
        if not rows:
            return {
                "query": queryRequest.queryText,
                "hits": [],
                "message": "No results found in the FAISS index.",
            }
        # convert rows to simple JSON-safe structure
        hits = []
        for r in rows:
            hits.append(
                {
                    "id": r["id"],
                    "title": r["title"],
                    "similarity": (
                        float(r["similarity"]) if r["similarity"] is not None else None
                    ),
                }
            )
        return {"query": queryRequest.queryText, "hits": hits}
    except Exception as exception:
        raise HTTPException(status_code=500, detail=str(exception))


def _build_context(rows, max_chars: int) -> list[str]:
    context_chunks = []
    total = 0
    for idx, row in enumerate(rows, 1):
        chunk = f"Source {idx}: {row['title']}\n{row['text']}"
        if total + len(chunk) > max_chars:
            break
        context_chunks.append(chunk)
        total += len(chunk)
    return context_chunks


async def generate_answer(
    queryText: str,
    top_k: int,
    max_context_chars: int,
) -> dict:
    try:
        query_embedding = await run_in_threadpool(embed_query_text, queryText)

        query_embedding = normalize_query_vector(query_embedding)

        rows = _faiss_search(query_embedding, top_k)
        logger.info("faiss_search_completed")
        rows = await run_in_threadpool(rerank_rows, queryText, rows, top_k)

        if not rows:
            return {
                "query": queryText,
                "answer": "I do not know based on the provided context.",
                "sources": [],
            }

        context_chunks = _build_context(rows, max_context_chars)
        answer_text = await run_in_threadpool(
            generate_llm_answer, queryText, context_chunks
        )

        sources = [
            {
                "id": r["id"],
                "title": r["title"],
                "similarity": (
                    float(r["similarity"]) if r["similarity"] is not None else None
                ),
            }
            for r in rows
        ]

        return {
            "query": queryText,
            "answer": answer_text,
            "sources": sources,
        }
    except Exception as exception:
        raise HTTPException(status_code=500, detail=str(exception))


@app.post("/answer")
async def answer(request: AnswerRequest):
    """
    Retrieval + rerank + generation endpoint.
    1) Embed query
    2) Retrieve top-k chunks from pgvector
    3) Generate answer grounded in retrieved context
    """
    base_answer = await generate_answer(
        request.queryText, request.top_k, request.max_context_chars
    )
    return base_answer


@app.post("/incident-answer")
async def incident_answer(request: AnswerRequest):
    """
    Retrieval + rerank + generation endpoint + enrich answer.
    1) Embed query
    2) Retrieve top-k chunks from pgvector
    3) Generate answer grounded in retrieved context
    4) Enrich answer with confidence score and recommended actions
    """
    base_answer = await generate_answer(
        request.queryText, request.top_k, request.max_context_chars
    )
    enriched_answer = enrich_answer(
        base_answer["query"], base_answer["answer"], base_answer["sources"]
    )
    return enriched_answer
