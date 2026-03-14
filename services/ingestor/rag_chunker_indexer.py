# rag_chunker_indexer.py
"""
Production-ready-ish pipeline:
- Load documents (PDF/Markdown/Plain text/transcripts) via LangChain loaders
- Structure-aware recursive splitting (headings -> paragraphs -> sentences)
- Semantic boundary detection inside large sections 
    (SBERT by default, option to use OpenAI)
- Final chunking with token-size constraints and overlap
- Embed with OpenAIEmbeddings, normalize, index into FAISS IndexFlatIP
- Provide simple search() that returns top-K docs + cosine score

Usage:
    from rag_chunker_indexer import RAGIndexer
    indexer = RAGIndexer(openai_api_key="...", use_sbert_for_split=True)
    indexer.index_directory("data/", save_dir="index_store")
    results = indexer.search("How does billing work?", k=5)
"""

import os
import json
from typing import List, Dict, Optional

import numpy as np
import faiss
from tqdm.auto import tqdm

# LangChain components
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings
from langchain.embeddings import OpenAIEmbeddings

# Optional local model for cheaper semantic splitting
from sentence_transformers import SentenceTransformer

# Token counting (approx) using tiktoken for OpenAI models (if installed)

try:
    import tiktoken
except Exception:
    tiktoken = None

# Sentence tokenizer
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)


# ---------------------------
# Utilities
# ---------------------------
def approx_token_count(text: str, model_name: str = "text-embedding-3-small") -> int:
    """
    Approximate token count. If tiktoken installed, use it; otherwise use heuristic.
    """
    if tiktoken is not None:
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    # fallback heuristic: 4 chars per token
    return max(1, len(text) // 4)


def to_numpy32(x: List[List[float]]) -> np.ndarray:
    arr = np.array(x, dtype=np.float32)
    return arr


# ---------------------------
# Semantic boundary detector
# ---------------------------
class SemanticBoundarySplitter:
    """
    Splits long text to semantically-cohesive segments using sentence-level embeddings
    By default uses a local SBERT model (fast + cheap).
    Optionally can be configured to use an external embedding function (e.g., OpenAI)
    by passing embed_fn that accepts list[str] -> np.array.
    """

    def __init__(
        self,
        use_openai_for_split: bool = False,
        openai_embed_fn=None,
        sbert_model_name: str = "all-MiniLM-L6-v2",
        cos_threshold: float = 0.72,
        min_chunk_sentences: int = 3,
        max_chunk_sentences: int = 60,
        batch_size: int = 64,
    ):
        self.use_openai_for_split = use_openai_for_split
        self.openai_embed_fn = openai_embed_fn
        self.cos_threshold = cos_threshold
        self.min_chunk_sentences = min_chunk_sentences
        self.max_chunk_sentences = max_chunk_sentences
        self.batch_size = batch_size

        if not use_openai_for_split:
            self.model = SentenceTransformer(sbert_model_name)
        else:
            if openai_embed_fn is None:
                raise ValueError(
                    "If use_openai_for_split=True, " "provide openai_embed_fn"
                )

    def _embed_sentences(self, sents: List[str]) -> np.ndarray:
        if self.use_openai_for_split:
            emb = self.openai_embed_fn(sents)
            return np.array(emb, dtype=np.float32)
        else:
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
        sents = sent_tokenize(text)
        if len(sents) == 0:
            return []

        if len(sents) <= self.max_chunk_sentences:
            return [" ".join(sents)]

        embeddings = self._embed_sentences(sents)
        # Normalize embeddings row-wise
        # Then compute cosine similarity between consecutive sentences
        emb_norm = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        )
        sims = np.sum(emb_norm[:-1] * emb_norm[1:], axis=1)  # length = n-1

        # mark cuts when similarity < threshold
        cut_indices = []
        for i, sim in enumerate(sims):
            if sim < self.cos_threshold:
                cut_indices.append(i + 1)

        # assemble segments ensuring min_chunk_sentences and not exceeding max
        segments = []
        start = 0
        for cut in cut_indices:
            if cut - start < self.min_chunk_sentences:
                continue
            seg_sents = sents[start:cut]
            # if segment too long, hardware window it
            if len(seg_sents) > self.max_chunk_sentences:
                for i in range(0, len(seg_sents), self.max_chunk_sentences):
                    segments.append(
                        " ".join(seg_sents[i : i + self.max_chunk_sentences])
                    )
                else:
                    segments.append(" ".join(seg_sents))
                start = cut
        # final
        if start < len(sents):
            rem = sents[start:]
            if len(rem) > self.max_chunk_sentences:
                for i in range(0, len(rem), self.max_chunk_sentences):
                    segments.append(" ".join(rem[i : i + self.max_chunk_sentences]))
            else:
                segments.append(" ".join(rem))

        # merge tiny segments with previous if needed
        merged = []
        for seg in segments:
            if len(merged) == 0:
                merged.append(seg)
            else:
                cur_tokens = approx_token_count(seg)
                if cur_tokens < 40:  # tiny threshold in tokens; merge
                    merged[-1] = merged[-1] + " " + seg
                else:
                    merged.append(seg)
        return merged


# ---------------------------
# RAG Indexer class
# ---------------------------
class RAGIndexer:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_model: str = "text-embedding-3-small",
        chunk_size_tokens: int = 480,
        chunk_overlap_tokens: int = 80,
        structure_separators: Optional[List[str]] = None,
        semantic_splitter: Optional[SemanticBoundarySplitter] = None,
        use_sbert_for_split: bool = True,
    ):
        # API key
        if openai_api_key is None:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError(
                "OpenAI API key required "
                "(pass openai_api_key or set OPENAI_API_KEY env var)"
            )
        os.environ["OPENAI_API_KEY"] = openai_api_key

        self.openai_model = openai_model
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

        # structure-aware separators (LangChain splitter will try them in order)
        if structure_separators is None:
            self.structure_separators = [
                "\n\n### ",
                "\n\n## ",
                "\n\n# ",
                "\f",
                "\n\n",
                "\n",
                " ",
                "",
            ]
        else:
            self.structure_separators = structure_separators

        # LangChain splitter used for final chunk size enforcement & recursive split
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size_tokens,
            chunk_overlap=self.chunk_overlap_tokens,
            separators=self.structure_separators,
        )

        # Semantic split helper inside large sections
        if semantic_splitter is None:
            self.semantic_splitter = (
                SemanticBoundarySplitter(use_openai_for_split=False)
                if use_sbert_for_split
                else SemanticBoundarySplitter(
                    use_openai_for_split=True,
                    openai_embed_fn=self._openai_embed_batch,
                )
            )
        else:
            self.semantic_splitter = semantic_splitter

        # OpenAI embeddings via LangChain wrapper
        self.openai_embeddings = OpenAIEmbeddings(model=self.openai_model, chunk_size=1)
        self.index = None
        self.id_to_meta = []  # list of metadata dicts aligned with index IDs
        self.id_to_text = []  # original chunk texts aligned with index IDs

    # ---------------------------
    # Loading documents
    # ---------------------------
    def load_documents_from_directory(
        self,
        directory: str,
        recursive: bool = True,
        include_patterns: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Uses LangChain DirectoryLoader with common loaders to produce
        a list of Documents (langchain.schema.Document)
        We'll return plain tuples of (text, metadata)
        """
        loader = DirectoryLoader(
            directory,
            glob=include_patterns or ["**/*.pdf", "**/*.md", "**/*.txt"],
            recursive=recursive,
        )
        docs = loader.load()
        results = []
        for d in docs:
            meta = d.metadata if hasattr(d, "metadata") else {}
            txt = d.page_content if hasattr(d, "page_content") else str(d)
            results.append({"text": txt, "metadata": meta})
        return results

    def _openai_embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Wrapper to produce embeddings for a list of strings
        using self.openai_embeddings. Returns ndarray (n, dim) float32.
        """
        embs = []
        # LangChain OpenAIEmbeddings supports batch via embed_documents
        embs = self.openai_embeddings.embed_documents(texts)
        return np.array(embs, dtype=np.float32)

    # ---------------------------
    # Hierarchical recursive splitting with semantic boundaries inside large sections
    # ---------------------------
    def process_document(
        self, text: str, metadata: Dict = None, semantic_threshold_chars: int = 3000
    ) -> List[Dict]:
        """
        Returns list of chunks: {"text": ..., "metadata": {...}}
        Steps:
          - Top-level splitter by structure (keeps headings where present)
          - For each top-level chunk that is large: split into paragraphs
          - Apply semantic splitting for large paragraphs
          - Final pass: Enforce token-size with split_text & apply chunk overlap
        """
        metadata = metadata or {}
        # First pass: try to split by high-level separators (headings/pages).
        # Use the splitter but with large chunk_size so we just split by separators.
        # We'll reuse RecursiveCharacterTextSplitter by temporarily setting
        # chunk_size huge to force splitting by separators only.
        high_level_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000000, chunk_overlap=0, separators=self.structure_separators
        )
        top_sections = high_level_splitter.split_text(text)

        final_chunks = []
        for sec in top_sections:
            if len(sec) <= semantic_threshold_chars:
                # just apply final splitter to ensure token limit
                subchunks = self.splitter.split_text(sec)
                for sc in subchunks:
                    final_chunks.append({"text": sc, "metadata": metadata})
            else:
                # big section: try paragraph splitting
                paras = [p for p in sec.split("\n\n") if p.strip()]
                para_chunks = []
                for p in paras:
                    if len(p) <= semantic_threshold_chars:
                        para_chunks.append(p)
                    else:
                        # run semantic boundary detection
                        segs = self.semantic_splitter.split_text(p)
                        para_chunks.extend(segs)
                # final enforcement
                for pc in para_chunks:
                    subchunks = self.splitter.split_text(pc)
                    for sc in subchunks:
                        final_chunks.append({"text": sc, "metadata": metadata})
        return final_chunks

    # ---------------------------
    # Embedding + indexing
    # ---------------------------
    def index_chunks(
        self,
        chunks: List[Dict],
        persist_dir: Optional[str] = None,
        show_progress: bool = True,
    ):
        """
        Given list of {"text": ..., "metadata": ...}, produce OpenAI embeddings,
        normalize, and add to FAISS IndexFlatIP
        Also store id->metadata and id->text arrays.
        """
        texts = [c["text"] for c in chunks]
        metas = [c.get("metadata", {}) for c in chunks]
        # embed in batches using OpenAI embeddings wrapper
        batch_size = 16
        all_embs = []
        for i in tqdm(
            range(0, len(texts), batch_size),
            disable=not show_progress,
            desc="Embedding batches",
        ):
            batch_texts = texts[i : i + batch_size]
            batch_emb = self._openai_embed_batch(batch_texts)
            all_embs.append(batch_emb)
        # concatenate all batch embeddings into single array of shape (N, dim)
        all_embs = np.vstack(all_embs).astype(np.float32)

        # Normalize L2 (in-place)
        faiss.normalize_L2(all_embs)

        # Initialize FAISS index if needed
        dim = all_embs.shape[1]
        if self.index is None:
            # inner product on normalized vectors => cosine similarity
            self.index = faiss.IndexFlatIP(dim)
        else:
            # verify same dim
            if self.index.d != dim:
                raise ValueError(
                    f"Embedding dim mismatch: index dim {self.index.d} "
                    f"vs embeddings {dim}"
                )

        # add embeddings
        self.index.add(all_embs)

        # append metadata/text
        self.id_to_meta.extend(metas)
        self.id_to_text.extend(texts)

        # optionally persist
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            # save faiss index
            faiss.write_index(self.index, os.path.join(persist_dir, "faiss_index.idx"))
            # save metadata & texts
            with open(
                os.path.join(persist_dir, "meta.jsonl"), "w", encoding="utf-8"
            ) as f:
                for m, t in zip(self.id_to_meta, self.id_to_text):
                    rec = {"metadata": m, "text": t}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return len(all_embs)

    # ---------------------------
    # High-level indexing from directory
    # ---------------------------
    def index_directory(
        self,
        directory: str,
        save_dir: Optional[str] = None,
        include_patterns: Optional[List[str]] = None,
    ):
        """
        Loads files from directory (PDF/MD/TXT), processes and indexes them.
        """
        raw_docs = self.load_documents_from_directory(
            directory, include_patterns=include_patterns
        )
        all_chunks = []
        for doc in tqdm(raw_docs, desc="Processing documents"):
            chunks = self.process_document(
                doc["text"], metadata=doc.get("metadata", {})
            )
            # attach source info if available
            for ch in chunks:
                ch["metadata"] = ch["metadata"] if ch["metadata"] else {}
                ch["metadata"].update(
                    {"source": doc.get("metadata", {}).get("source", None)}
                )
            all_chunks.extend(chunks)

        total = self.index_chunks(all_chunks, persist_dir=save_dir)
        return total

    # ---------------------------
    # Search / query
    # ---------------------------
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Returns top-k matches with cosine scores and metadata.
        """
        q_emb = np.array(self.openai_embeddings.embed_query(query), dtype=np.float32)
        # Reshape to (1, dim) and normalize
        q_emb = q_emb.reshape(1, -1)
        faiss.normalize_L2(q_emb)
        if self.index is None:
            raise ValueError("Index is empty. Build the index first.")
        # search
        distances, indices = self.index.search(q_emb, k)
        distances = distances[0]  # cosine similarity
        indices = indices[0]
        results = []
        for score, idx in zip(distances, indices):
            if idx < 0 or idx >= len(self.id_to_text):
                continue
            results.append(
                {
                    "score": float(score),
                    "text": self.id_to_text[idx],
                    "metadata": self.id_to_meta[idx],
                }
            )
        return results

    # ---------------------------
    # Persistence helpers
    # ---------------------------
    def save_local(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        if self.index is None:
            raise ValueError("No index to save")
        faiss.write_index(self.index, os.path.join(out_dir, "faiss_index.idx"))
        with open(os.path.join(out_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
            for m, t in zip(self.id_to_meta, self.id_to_text):
                rec = {"metadata": m, "text": t}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def load_local(self, out_dir: str):
        idx_path = os.path.join(out_dir, "faiss_index.idx")
        meta_path = os.path.join(out_dir, "meta.jsonl")
        if not os.path.exists(idx_path):
            raise FileNotFoundError(idx_path)
        self.index = faiss.read_index(idx_path)
        self.id_to_meta = []
        self.id_to_text = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line.strip())
                self.id_to_meta.append(rec.get("metadata", {}))
                self.id_to_text.append(rec.get("text", ""))


# ---------------------------
# Example CLI-like usage (if run as script)
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build FAISS index for "
        "a directory of docs using hierarchical semantic splitting."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory with PDFs/MD/TXT"
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key or use env var OPENAI_API_KEY",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="index_store",
        help="Where to save index & metadata",
    )
    parser.add_argument(
        "--use_sbert_for_split",
        action="store_true",
        help="Use local SBERT for semantic splitting (default)",
    )
    args = parser.parse_args()
    indexer = RAGIndexer(
        openai_api_key=args.openai_api_key, use_sbert_for_split=args.use_sbert_for_split
    )
    count = indexer.index_directory(args.data_dir, save_dir=args.out_dir)
    print(f"Indexed {count} embeddings -> saved to {args.out_dir}")

    # Simple interactive query
    while True:
        q = input("Query (or 'exit'): ")
        if q.strip().lower() in ("exit", "quit"):
            break
        res = indexer.search(q, k=5)
        for r in res:
            print(
                f"[score={r['score']:.4f}] "
                f"{r['metadata'].get('source','')} --- "
                f"{r['text'][:300]}...\n"
            )
