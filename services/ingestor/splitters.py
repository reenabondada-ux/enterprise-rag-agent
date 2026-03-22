# services/ingestor/splitters.py
from typing import Dict, List, Optional

import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from services.core.config import LOCAL_EMBEDDING_MODEL_SBERT

nltk.download("punkt", quiet=True)

_sbert_tokenizer = None


def approx_token_count(text: str) -> int:
    """
    Approximate token count using the SBERT tokenizer.
    """
    global _sbert_tokenizer
    if _sbert_tokenizer is None:
        _sbert_tokenizer = AutoTokenizer.from_pretrained(LOCAL_EMBEDDING_MODEL_SBERT)
    return len(_sbert_tokenizer.encode(text))


class SemanticBoundarySplitter:
    """
    Splits long text to semantically-cohesive segments using sentence-level embeddings.
    Uses a local SBERT model.
    """

    def __init__(
        self,
        sbert_model_name: str = LOCAL_EMBEDDING_MODEL_SBERT,
        cos_threshold: float = 0.72,
        min_chunk_sentences: int = 3,
        max_chunk_sentences: int = 60,
        batch_size: int = 64,
    ):
        self.cos_threshold = cos_threshold
        self.min_chunk_sentences = min_chunk_sentences
        self.max_chunk_sentences = max_chunk_sentences
        self.batch_size = batch_size
        self.model = SentenceTransformer(sbert_model_name)

    # Internal method to get embeddings for sentences using SBERT
    def _embed_sentences(self, sents: List[str]) -> np.ndarray:
        emb = self.model.encode(
            sents,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)

    def split_text(self, text: str) -> List[str]:
        """
        Returns a list of segments (strings). Keeps sentence order.
        """
        sents = sent_tokenize(
            text
        )  # split sentences with nltk (natural language toolkit)
        if len(sents) == 0:
            return []

        if len(sents) <= self.max_chunk_sentences:
            return [" ".join(sents)]

        # Get sentence embeddings, normalise & compute similarity b/w adjacent sents
        embeddings = self._embed_sentences(sents)
        emb_norm = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        )
        sims = np.sum(emb_norm[:-1] * emb_norm[1:], axis=1)

        # Identify cut points where similarity drops below threshold
        cut_indices = []
        for i, sim in enumerate(sims):
            if sim < self.cos_threshold:
                cut_indices.append(i + 1)

        # Group sentences into segments based on cut points,
        # while enforcing min/max sentence limits
        segments = []
        start = 0
        for cut in cut_indices:
            if cut - start < self.min_chunk_sentences:
                continue
            seg_sents = sents[start:cut]
            if len(seg_sents) > self.max_chunk_sentences:
                for i in range(0, len(seg_sents), self.max_chunk_sentences):
                    segments.append(
                        " ".join(seg_sents[i : i + self.max_chunk_sentences])
                    )
            else:
                segments.append(" ".join(seg_sents))
            start = cut

        if start < len(sents):
            rem = sents[start:]
            if len(rem) > self.max_chunk_sentences:
                for i in range(0, len(rem), self.max_chunk_sentences):
                    segments.append(" ".join(rem[i : i + self.max_chunk_sentences]))
            else:
                segments.append(" ".join(rem))

        # Merge short segments (less than 40 tokens) with neighbors to avoid tiny chunks
        merged = []
        for seg in segments:
            if len(merged) == 0:
                merged.append(seg)
            else:
                cur_tokens = approx_token_count(seg)
                if cur_tokens < 40:
                    merged[-1] = merged[-1] + " " + seg
                else:
                    merged.append(seg)
        return merged


def semantic_split_document(
    text: str,
    metadata: Optional[Dict] = None,
    semantic_splitter: Optional[SemanticBoundarySplitter] = None,
    structure_separators: Optional[List[str]] = None,
    chunk_size_tokens: int = 480,
    chunk_overlap_tokens: int = 80,
    semantic_threshold_chars: int = 3000,
) -> List[Dict]:
    """
    Hierarchical recursive splitting with semantic boundaries inside large sections.
    Returns list of chunks: {"text": ..., "metadata": {...}}.
    """
    metadata = metadata or {}
    if structure_separators is None:
        structure_separators = [
            "\n\n### ",  # H3 headings
            "\n\n## ",  # H2 headings
            "\n\n# ",  # H1 headings
            "\f",  # page break (PDF Form feed)
            "\n\n",  # Paragraphs
            "\n",  # Line breaks
            " ",  # word
            "",  # character-level fallback
        ]

    if semantic_splitter is None:
        semantic_splitter = SemanticBoundarySplitter()

    # LangChain splitter used for final chunk size enforcement & recursive split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_tokens,
        chunk_overlap=chunk_overlap_tokens,
        separators=structure_separators,
    )

    # Step 1: Top-level split by structure (e.g. sections) using large chunk size
    high_level_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000000,
        chunk_overlap=0,
        separators=structure_separators,
    )
    top_sections = high_level_splitter.split_text(text)

    # Step 2: For each top-level section, if it's above semantic_threshold_chars,
    # split by semantic boundaries; otherwise keep as is.
    final_chunks: List[Dict] = []
    for sec in top_sections:
        if len(sec) <= semantic_threshold_chars:
            subchunks = splitter.split_text(sec)
            for sc in subchunks:
                final_chunks.append({"text": sc, "metadata": metadata})
        else:
            paras = [p for p in sec.split("\n\n") if p.strip()]
            para_chunks: List[str] = []
            for p in paras:
                if len(p) <= semantic_threshold_chars:
                    para_chunks.append(p)
                else:
                    segs = semantic_splitter.split_text(p)
                    para_chunks.extend(segs)
            for pc in para_chunks:
                subchunks = splitter.split_text(pc)
                for sc in subchunks:
                    final_chunks.append({"text": sc, "metadata": metadata})
    return final_chunks
