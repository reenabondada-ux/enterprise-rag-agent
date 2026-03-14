# services/core/vector/faiss_index.py
"""
FAISS index helpers for vector search.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Tuple

import fcntl

import faiss
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector


logger = logging.getLogger("enterprise_rag_faiss")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_or_create_index(
    dimension: int, index_path: str, db_dsn: str | None = None
) -> faiss.Index:
    """
    Load a FAISS index from disk if it exists; otherwise create a new index
    with the given vector dimension and return an IndexFlatIP for cosine similarity.
        Note: IndexFlatIP does inner product search, so we need to L2 normalize
        If Magnitude matters, consider using IndexFlatL2 and skip normalization.
    """
    if os.path.exists(index_path) and os.path.getsize(index_path) > 0:
        with _index_file_lock(index_path):
            try:
                return faiss.read_index(index_path)
            except Exception as exc:
                # Fall back to rebuild if index is corrupted and DB is available
                if db_dsn:
                    logger.error(
                        "faiss_index_load_failed",
                        extra={"path": index_path, "error": str(exc)},
                    )
                    try:
                        return _rebuild_index_from_postgres(
                            dimension, index_path, db_dsn
                        )
                    except Exception as rebuild_exc:
                        logger.error(
                            "faiss_rebuild_failed",
                            extra={
                                "path": index_path,
                                "error": str(rebuild_exc),
                            },
                        )
                        return faiss.IndexFlatIP(dimension)
                logger.error(
                    "faiss_index_load_failed",
                    extra={"path": index_path, "error": str(exc)},
                )
                return faiss.IndexFlatIP(dimension)
    return faiss.IndexFlatIP(dimension)


def save_index(index: faiss.Index, index_path: str) -> None:
    """
    Persist the FAISS index to disk at the given path.
    """
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    # Write to a temp file then atomically replace to avoid partial reads
    tmp_path = f"{index_path}.tmp"
    with _index_file_lock(index_path):
        faiss.write_index(index, tmp_path)
        os.replace(tmp_path, index_path)


def normalize_vectors(vectors: list[list[float]]) -> list[list[float]]:
    """
    Normalize vectors for cosine similarity with IndexFlatIP
    """
    vectors_array = np.asarray(vectors, dtype="float32")
    if vectors_array.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    faiss.normalize_L2(vectors_array)
    return vectors_array.tolist()


def normalize_query_vector(vector: list[float]) -> list[float]:
    """
    Normalize a single vector for cosine similarity with IndexFlatIP
    """
    vector_array = np.asarray([vector], dtype="float32")
    faiss.normalize_L2(vector_array)
    return vector_array[0].tolist()


def add_embeddings(
    index: faiss.Index, ids: list[int], vectors: list[list[float]]
) -> faiss.Index:
    """
    Add vectors to the FAISS index using the provided integer IDs.
    IDs should match the corresponding rows in the metadata store (Postgres).
    """
    if not ids or not vectors:
        return index
    if len(ids) != len(vectors):
        raise ValueError("ids and vectors must have the same length")

    vectors_array = np.asarray(vectors, dtype="float32")
    id_array = np.asarray(ids, dtype="int64")

    # Check vectors_array dimension consistency
    if vectors_array.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    if vectors_array.shape[1] != index.d:
        raise ValueError(
            f"Vector dimension mismatch: "
            f"got {vectors_array.shape[1]}, expected {index.d}"
        )

    # Ensure index is an IndexIDMap to support custom IDs (postgres documentIds)
    if not isinstance(index, faiss.IndexIDMap):
        index = faiss.IndexIDMap(index)
    index.add_with_ids(vectors_array, id_array)
    return index


def search_index(
    index: faiss.Index, query_vector: list[float], top_k: int
) -> Tuple[list[int], list[float]]:
    """
    Search the FAISS index with a single query vector and return a tuple
    of (ids, distances) for the top_k nearest neighbors.
    """
    if top_k <= 0:
        return ([], [])

    vector_query = np.asarray([query_vector], dtype="float32")
    # Normalize query for cosine similarity with IndexFlatIP
    faiss.normalize_L2(vector_query)
    distances, ids = index.search(vector_query, top_k)

    return ids[0].tolist(), distances[0].tolist()


def _rebuild_index_from_postgres(
    dimension: int, index_path: str, db_dsn: str
) -> faiss.Index:
    """
    Rebuild FAISS index from Postgres embeddings.
    Assumes embeddings stored in Postgres are already normalized.
    """
    index = faiss.IndexFlatIP(dimension)
    logger.info("faiss_rebuild_started", extra={"path": index_path})
    connection = psycopg2.connect(db_dsn)
    # register pgvector adapter for vector de/serialization
    register_vector(connection)
    cursor = connection.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT id, embedding FROM documents")
    rows = cursor.fetchall()
    cursor.close()
    connection.close()

    if not rows:
        logger.warning("faiss_rebuild_empty", extra={"path": index_path})
        save_index(index, index_path)
        return index

    ids = [row["id"] for row in rows]
    vectors = [row["embedding"] for row in rows]
    index = add_embeddings(index, ids, vectors)
    save_index(index, index_path)
    logger.info(
        "faiss_rebuild_completed",
        extra={"path": index_path, "count": len(ids)},
    )
    return index


@contextmanager
def _index_file_lock(index_path: str):
    """
    Best-effort file lock for index read/write operations.
    Uses fcntl on Unix; falls back to no-op if unavailable.
    """
    lock_path = f"{index_path}.lock"
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    lock_file = None
    try:
        lock_file = open(lock_path, "w")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        except Exception:
            # If fcntl is unavailable, continue without locking
            pass
        yield
    finally:
        try:
            if lock_file:
                try:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                except Exception:
                    pass
                lock_file.close()
        except Exception:
            pass
