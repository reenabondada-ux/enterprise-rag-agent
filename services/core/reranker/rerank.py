"""
Optional cross-encoder reranking for retrieved rows.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Iterable

from sentence_transformers import CrossEncoder

from services.core.config import (
    RERANKER_ENABLED,
    RERANKER_MODEL,
    RERANKER_TOP_K,
)

logger = logging.getLogger("enterprise_rag_reranker")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

_RERANKER_MODEL = None
_MODEL_LOCK = Lock()


def _get_model() -> CrossEncoder:
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None:
        with _MODEL_LOCK:
            if _RERANKER_MODEL is None:
                logger.info(
                    "reranker_model_loading",
                    extra={"model": RERANKER_MODEL},
                )
                _RERANKER_MODEL = CrossEncoder(RERANKER_MODEL)
                logger.info(
                    "reranker_model_loaded",
                    extra={"model": RERANKER_MODEL},
                )
    return _RERANKER_MODEL


def rerank_rows(
    query_text: str,
    rows: list[dict],
    top_k: int | None = None,
) -> list[dict]:
    """
    Rerank retrieved rows using a cross-encoder model.

    Returns rows sorted by rerank score (desc) with metadata preserved.
    If reranking is disabled or fails, returns input rows unchanged.
    """
    if not rows:
        return rows
    if not RERANKER_ENABLED:
        logger.info("reranker_skipped_disabled")
        return rows

    try:
        model = _get_model()
        pairs: list[tuple[str, str]] = [
            (query_text, (row.get("text") or "")) for row in rows
        ]
        scores: Iterable[float] = model.predict(pairs)
        ranked_rows: list[dict] = []
        for row, score in zip(rows, scores):
            ranked_row = dict(row)
            ranked_row["rerank_score"] = float(score)
            ranked_rows.append(ranked_row)
        ranked_rows.sort(key=lambda r: r.get("rerank_score", 0.0), reverse=True)
        limit = top_k or RERANKER_TOP_K or len(ranked_rows)
        return ranked_rows[: min(limit, len(ranked_rows))]
    except Exception as exc:
        logger.error(
            "reranker_failed",
            extra={"error": str(exc)},
        )
        return rows
