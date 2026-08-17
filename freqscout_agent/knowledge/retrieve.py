"""
Knowledge Retriever
-------------------
Runs hybrid BM25 + dense retrieval against the knowledge base.

Scoring: combined_score = alpha * bm25_norm + (1-alpha) * dense_norm
         where alpha=0.5 by default (ADRE setting).

Context expansion: after selecting top-k chunks, includes ±1 adjacent chunks
(same source, consecutive chunk_ids) for coherence.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

TOOL_NAME = "retrieve"


@dataclass
class RetrievedChunk:
    chunk_id: str
    source: str
    section: str
    page: Optional[int]
    text: str
    bm25_score: float
    dense_score: float
    combined_score: float


class KnowledgeRetriever:

    TOOL_NAME = TOOL_NAME

    def __init__(
        self,
        index_dir: str = "freqscout_agent/knowledge/index_store",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        alpha: float = 0.5,
        top_k: int = 5,
        context_expansion: int = 1,
    ):
        self.index_dir = Path(index_dir)
        self.embed_model_id = embed_model
        self.alpha = alpha
        self.top_k = top_k
        self.context_expansion = context_expansion

        self._chunks: List[Dict] = []
        self._bm25 = None
        self._tokenised: List[List[str]] = []
        self._faiss_index = None
        self._embed_model = None
        self._embeddings: Optional[np.ndarray] = None

        self._load_index()

    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """Retrieve top-k chunks for the given query string."""
        if not self._chunks:
            logger.warning("Knowledge base is empty. Returning empty results.")
            return []

        bm25_scores  = self._bm25_score(query)
        dense_scores = self._dense_score(query)

        # Normalise both to [0, 1]
        bm25_norm  = self._minmax(bm25_scores)
        dense_norm = self._minmax(dense_scores)

        combined = self.alpha * bm25_norm + (1 - self.alpha) * dense_norm

        top_indices = np.argsort(combined)[::-1][: self.top_k]

        # Context expansion: add adjacent chunks from same source
        expanded = set(top_indices.tolist())
        for idx in top_indices:
            for delta in range(-self.context_expansion, self.context_expansion + 1):
                neighbour = idx + delta
                if 0 <= neighbour < len(self._chunks):
                    if self._chunks[neighbour]["source"] == self._chunks[idx]["source"]:
                        expanded.add(neighbour)

        # Sort by combined score descending
        ordered = sorted(expanded, key=lambda i: combined[i], reverse=True)

        results = []
        for i in ordered:
            c = self._chunks[i]
            results.append(RetrievedChunk(
                chunk_id=c["chunk_id"],
                source=c["source"],
                section=c["section"],
                page=c.get("page"),
                text=c["text"],
                bm25_score=float(bm25_scores[i]),
                dense_score=float(dense_scores[i]),
                combined_score=float(combined[i]),
            ))
        logger.debug("Retrieved %d chunks for query: %s…", len(results), query[:60])
        return results

    # ------------------------------------------------------------------

    def _load_index(self):
        chunks_path = self.index_dir / "chunks.jsonl"
        if not chunks_path.exists():
            logger.warning("No knowledge index found at %s. Run KnowledgeBaseBuilder.build() first.", self.index_dir)
            return

        with open(chunks_path, encoding="utf-8") as f:
            self._chunks = [json.loads(line) for line in f if line.strip()]
        logger.info("Loaded %d knowledge chunks.", len(self._chunks))

        # BM25
        bm25_path = self.index_dir / "bm25.pkl"
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
                self._bm25 = data["bm25"]
                self._tokenised = data["tokenised"]

        # FAISS + embeddings
        faiss_path = self.index_dir / "faiss.index"
        embed_path = self.index_dir / "embed_matrix.npy"
        if faiss_path.exists():
            try:
                import faiss
                self._faiss_index = faiss.read_index(str(faiss_path))
                if embed_path.exists():
                    self._embeddings = np.load(str(embed_path))
            except ImportError:
                logger.warning("faiss not installed; dense retrieval disabled.")

    # ------------------------------------------------------------------

    def _bm25_score(self, query: str) -> np.ndarray:
        if self._bm25 is None:
            return np.zeros(len(self._chunks))
        tokens = re.findall(r"[a-zA-Z0-9]+", query.lower())
        return np.array(self._bm25.get_scores(tokens), dtype=float)

    def _dense_score(self, query: str) -> np.ndarray:
        if self._faiss_index is None:
            return np.zeros(len(self._chunks))
        try:
            from sentence_transformers import SentenceTransformer
            if self._embed_model is None:
                self._embed_model = SentenceTransformer(self.embed_model_id)
            q_vec = self._embed_model.encode([query], normalize_embeddings=True).astype("float32")
            _, _, scores = self._faiss_index.search_and_reconstruct(q_vec, len(self._chunks))
            # search returns distances; with IndexFlatIP these are dot products (cosine)
            _distances, _indices = self._faiss_index.search(q_vec, len(self._chunks))
            # Map back to chunk order
            scores_arr = np.zeros(len(self._chunks))
            for rank, (idx, dist) in enumerate(zip(_indices[0], _distances[0])):
                if 0 <= idx < len(self._chunks):
                    scores_arr[idx] = float(dist)
            return scores_arr
        except Exception as exc:
            logger.warning("Dense retrieval failed: %s", exc)
            return np.zeros(len(self._chunks))

    @staticmethod
    def _minmax(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        if hi == lo:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)
