"""
Knowledge Base Builder
----------------------
Builds and saves a hybrid BM25 + FAISS knowledge index from parsed chunks.

Storage:
  - chunks.jsonl        — all chunks as JSON lines
  - bm25.pkl            — serialised BM25Okapi index (rank_bm25)
  - faiss.index         — FAISS IndexFlatIP over MiniLM-L6-v2 embeddings
  - embed_matrix.npy    — raw embedding matrix (for reloading without re-encoding)
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


class KnowledgeBaseBuilder:

    def __init__(
        self,
        index_dir: str = "freqscout_agent/knowledge/index_store",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embed_model_id = embed_model

    # ------------------------------------------------------------------

    def build(self, chunks: List[Dict], force: bool = False):
        """
        Build and save the BM25 + FAISS index from `chunks`.

        Parameters
        ----------
        chunks : list of dicts from parse_pdfs.parse_all()
        force  : rebuild even if index already exists
        """
        chunks_path = self.index_dir / "chunks.jsonl"
        if chunks_path.exists() and not force:
            logger.info("Index already exists at %s. Use force=True to rebuild.", self.index_dir)
            return

        logger.info("Building knowledge base from %d chunks…", len(chunks))

        # Save raw chunks
        with open(chunks_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        # BM25
        self._build_bm25(chunks)

        # Dense (FAISS)
        self._build_faiss(chunks)

        logger.info("Knowledge base built and saved to %s.", self.index_dir)

    # ------------------------------------------------------------------

    def _build_bm25(self, chunks: List[Dict]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed. BM25 index skipped.")
            return

        tokenised = [self._tokenise(c["text"]) for c in chunks]
        bm25 = BM25Okapi(tokenised)
        bm25_path = self.index_dir / "bm25.pkl"
        with open(bm25_path, "wb") as f:
            pickle.dump({"bm25": bm25, "tokenised": tokenised}, f)
        logger.info("BM25 index saved (%d docs).", len(chunks))

    def _build_faiss(self, chunks: List[Dict]):
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            logger.warning("FAISS/sentence-transformers not installed (%s). Dense index skipped.", exc)
            return

        logger.info("Encoding %d chunks with %s…", len(chunks), self.embed_model_id)
        model = SentenceTransformer(self.embed_model_id)
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        embeddings = embeddings.astype("float32")

        np.save(str(self.index_dir / "embed_matrix.npy"), embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product on normalised vectors = cosine
        index.add(embeddings)
        faiss.write_index(index, str(self.index_dir / "faiss.index"))
        logger.info("FAISS index saved (%d vectors, dim=%d).", len(chunks), dim)

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        import re
        return re.findall(r"[a-zA-Z0-9]+", text.lower())
