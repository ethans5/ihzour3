# retrieval/embedding_retriever.py
from __future__ import annotations

from typing import List, Dict, Any
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class EmbeddingRetriever:
    def __init__(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray, model_name: str):
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: chunks={len(chunks)} vs embeddings={embeddings.shape[0]} "
                "(il faut même ordre et même taille)."
            )

        self.chunks = chunks

        # Important: embeddings doivent être en float32 pour accélérer + réduire RAM
        # (ça ne change quasiment pas les résultats)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32, copy=False)
        self.embeddings = embeddings

        # Le modèle est chargé UNE fois ici (c'est déjà ton cas)
        self.model = SentenceTransformer(model_name)

        # (1) Cache: évite de ré-encoder la même requête plusieurs fois
        # clé = texte de la requête, valeur = embedding normalisé shape (768,)
        self._query_cache: Dict[str, np.ndarray] = {}

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode + normalize query, with cache."""
        q = query.strip()
        if q in self._query_cache:
            return self._query_cache[q]

        emb = self.model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        emb = emb.squeeze().astype(np.float32, copy=False)  # shape (768,)
        self._query_cache[q] = emb
        return emb

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if k <= 0:
            return []

        # Encode query (cached)
        q = self._encode_query(query)  # shape (768,)

        # Cosine similarity because embeddings are normalized
        # scores shape: (N,)
        scores = self.embeddings @ q

        # (2) Top-K faster: argpartition instead of full argsort
        n = scores.shape[0]
        k = min(k, n)

        # indices of top-k (unordered), then sort only those k by score desc
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        out: List[Dict[str, Any]] = []
        for rank, i in enumerate(top_idx, start=1):
            c = self.chunks[int(i)]
            out.append({
                "rank": rank,
                "score": float(scores[int(i)]),
                "chunk_id": c.get("chunk_id"),
                "parent_id": c.get("parent_id"),
                "filename": c.get("filename"),
                "date": c.get("date"),
                "parliament": c.get("parliament"),
                "text": c.get("text", ""),
            })
        return out
