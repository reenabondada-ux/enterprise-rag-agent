# services/ingestor/ingest.py
import json
import logging
import os
import urllib.request
from datetime import datetime, timezone

import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from psycopg2.extras import RealDictCursor
from splitters import fixed_split
from services.shared import embed_texts
from services.shared import DATABASE_URL, EMBEDDING_DIM, FAISS_INDEX_PATH
from services.shared import (
    load_or_create_index,
    normalize_vectors,
    normalize_query_vector,
    add_embeddings,
    save_index,
)

STANDARD_LOG_FIELDS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {
            k: v for k, v in record.__dict__.items() if k not in STANDARD_LOG_FIELDS
        }
        if extras:
            log.update(extras)
        return json.dumps(log, default=str)


logger = logging.getLogger("enterprise_rag_ingestor")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

load_dotenv()


def ensure_schema():
    connection = psycopg2.connect(DATABASE_URL)
    cursor = connection.cursor()

    # Create vector extension (pgvector) and documents table
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    # register pgvector adapter after extension exists for vector de/serialization
    register_vector(connection)
    cursor.execute(
        f"""    
        CREATE TABLE IF NOT EXISTS documents (        
            id SERIAL PRIMARY KEY,        
            title TEXT,        
            text TEXT,        
            embedding vector({EMBEDDING_DIM})    
        );
        """
    )

    connection.commit()
    cursor.close()
    connection.close()
    logger.info("schema_ensured")


def update_faiss_index(doc_ids, embeddings) -> None:
    if not doc_ids or not embeddings:
        return
    index = load_or_create_index(EMBEDDING_DIM, FAISS_INDEX_PATH, DATABASE_URL)
    index = add_embeddings(index, doc_ids, embeddings)
    save_index(index, FAISS_INDEX_PATH)


def upsert_doc(title, text, embedding):
    connection = psycopg2.connect(DATABASE_URL)
    register_vector(connection)
    cursor = connection.cursor(cursor_factory=RealDictCursor)
    cursor.execute(
        (
            "INSERT INTO documents (title, text, embedding) "
            "VALUES (%s, %s, %s) RETURNING id"
        ),
        (title, text, embedding),
    )
    doc_id = cursor.fetchone()["id"]
    connection.commit()
    cursor.close()
    connection.close()
    return doc_id


def ingest_text_file(path, title=None):
    """Ingest a single text file, splitting into chunks."""
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    title = title or os.path.basename(path)
    # simple fixed splitter; replace with semantic chunker later
    chunks = fixed_split(text, chunk_size=1000, overlap=200)
    embeddings = normalize_vectors(embed_texts(chunks))
    doc_ids = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        doc_id = upsert_doc(f"{title}_chunk_{i}", chunk, emb)
        doc_ids.append(doc_id)
    update_faiss_index(doc_ids, embeddings)
    logger.info(
        "file_ingested",
        extra={"title": title, "chunks": len(chunks)},
    )


def ingest_directory(dir_path, extension=".txt"):
    """Ingest all files with given extension from a directory."""
    if not os.path.isdir(dir_path):
        raise ValueError(f"Directory not found: {dir_path}")

    files = [f for f in os.listdir(dir_path) if f.endswith(extension)]
    if not files:
        print(f"No {extension} files found in {dir_path}")
        return

    logger.info("ingest_started", extra={"files": len(files)})
    for filename in files:
        file_path = os.path.join(dir_path, filename)
        try:
            ingest_text_file(file_path)
        except Exception as e:
            logger.error(
                "ingest_failed",
                extra={"file": filename, "error": str(e)},
            )

    logger.info("ingest_completed", extra={"files": len(files)})


def _notify_faiss_reload() -> None:
    reload_url = os.getenv("FAISS_RELOAD_URL", "http://localhost:8000/faiss/reload")
    if not reload_url:
        logger.info("faiss_reload_skipped", extra={"reason": "missing_url"})
        return
    try:
        req = urllib.request.Request(reload_url, method="POST")
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status != 200:
                logger.error(
                    "faiss_reload_failed",
                    extra={"status": response.status},
                )
            else:
                logger.info("faiss_reload_notified")
    except Exception as exc:
        logger.error(
            "faiss_reload_failed",
            extra={"error": str(exc)},
        )


if __name__ == "__main__":
    ensure_schema()

    # Ingest sample documents from data/samples/ directory
    samples_dir = os.path.join(os.path.dirname(__file__), "../../data/samples")
    if os.path.isdir(samples_dir):
        logger.info("ingest_samples", extra={"dir": samples_dir})
        ingest_directory(samples_dir, extension=".txt")
        _notify_faiss_reload()
    else:
        logger.warning("samples_missing", extra={"dir": samples_dir})
        logger.info("creating_sample_doc")
        sample_text = (
            "This is a small sample document used as an example for the RAG ingest "
            "flow."
        )
        sample_embedding = normalize_query_vector(embed_texts([sample_text])[0])
        doc_id = upsert_doc("Sample doc", sample_text, sample_embedding)
        update_faiss_index([doc_id], [sample_embedding])
        logger.info("sample_doc_inserted")
        _notify_faiss_reload()
