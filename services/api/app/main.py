# services/api/app/main.py
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.shared import embed_query_text, generate_answer
from services.shared import DATABASE_URL
from starlette.concurrency import run_in_threadpool

load_dotenv()

app = FastAPI(title="Enterprise RAG Agent - API")


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
        print("DB connection failed:", exception)


@app.post("/query")
async def query(queryRequest: QueryRequest):
    """
    Minimal /query endpoint.
    - Intended to be replaced retrieval pipeline:
        * compute embedding for queryRequest.queryText
        * query documents using pgvector similarity operator (<->)
        * optionally re-rank with cross-encoder or LLM
    For now, returns top-k documents by distance using a placeholder vector
    (so you can verify the DB + table wiring).
    """
    try:
        connection = psycopg2.connect(DATABASE_URL)
        cursor = connection.cursor(cursor_factory=RealDictCursor)

        query_embedding = await run_in_threadpool(
            embed_query_text, queryRequest.queryText
        )

        # Example SQL - requires pgvector extension and a vector column named embedding
        cursor.execute(
            "SELECT id, title, text, (embedding <-> %s::vector) as distance "
            "FROM documents ORDER BY distance LIMIT %s",
            (query_embedding, queryRequest.top_k),
        )
        rows = cursor.fetchall()
        cursor.close()
        connection.close()
        # convert rows to simple JSON-safe structure
        hits = []
        for r in rows:
            hits.append(
                {
                    "id": r["id"],
                    "title": r["title"],
                    "distance": (
                        float(r["distance"]) if r["distance"] is not None else None
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


@app.post("/answer")
async def answer(request: AnswerRequest):
    """
    Retrieval + generation endpoint.
    1) Embed query
    2) Retrieve top-k chunks from pgvector
    3) Generate answer grounded in retrieved context
    """
    try:
        connection = psycopg2.connect(DATABASE_URL)
        cursor = connection.cursor(cursor_factory=RealDictCursor)

        query_embedding = await run_in_threadpool(embed_query_text, request.queryText)
        cursor.execute(
            "SELECT id, title, text, (embedding <-> %s::vector) as distance "
            "FROM documents ORDER BY distance LIMIT %s",
            (query_embedding, request.top_k),
        )
        rows = cursor.fetchall()
        cursor.close()
        connection.close()

        if not rows:
            return {
                "query": request.queryText,
                "answer": "I do not know based on the provided context.",
                "sources": [],
            }

        context_chunks = _build_context(rows, request.max_context_chars)
        answer_text = await run_in_threadpool(
            generate_answer, request.queryText, context_chunks
        )

        sources = [
            {
                "id": r["id"],
                "title": r["title"],
                "distance": float(r["distance"]) if r["distance"] is not None else None,
            }
            for r in rows
        ]

        return {
            "query": request.queryText,
            "answer": answer_text,
            "sources": sources,
        }
    except Exception as exception:
        raise HTTPException(status_code=500, detail=str(exception))
