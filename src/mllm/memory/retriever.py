"""Hybrid retriever: dense + sparse fusion (RRF) over DiskVectorStore.

Embedders are pluggable:
- HashEmbedder: zero-dependency, deterministic char-ngram hashing (tests/offline)
- SentenceTransformerEmbedder: lazy import, e.g. bge-small / nomic-embed (prod)
"""
from __future__ import annotations

import hashlib
from typing import List, Dict, Optional

import numpy as np

from .store import DiskVectorStore


class HashEmbedder:
    """Deterministic hash embedding (no deps). Weak but offline-capable.

    Used for tests and as a cold-start fallback. Production should use
    SentenceTransformerEmbedder with bge-small-en-v1.5 (33M params, 384-dim).
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = t.lower().split()
            # unigrams + bigrams hashed
            grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for g in grams[:512]:
                h = int(hashlib.md5(g.encode()).hexdigest(), 16)
                vecs[i, h % self.dim] += 1.0
            n = np.linalg.norm(vecs[i])
            if n > 0:
                vecs[i] /= n
        return vecs


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True),
                          dtype=np.float32)


class HybridRetriever:
    def __init__(self, store: DiskVectorStore, embedder=None,
                 dense_weight: float = 1.0, sparse_weight: float = 1.0,
                 rrf_k: int = 60):
        self.store = store
        self.embedder = embedder or HashEmbedder(store.dim)
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def index(self, texts: List[str], metas: Optional[List[dict]] = None,
              batch_size: int = 256):
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            vecs = self.embedder.encode(chunk)
            self.store.add(chunk, vecs, (metas or [{}] * len(texts))[i:i + batch_size])

    def search(self, query: str, k: int = 5, overfetch: int = 20) -> List[Dict]:
        # dense
        qv = self.embedder.encode([query])[0]
        dense = self.store.dense_search(qv, k=overfetch) if self.dense_weight > 0 else []
        sparse = self.store.sparse_search(query, k=overfetch) if self.sparse_weight > 0 else []
        # reciprocal rank fusion
        fused: Dict[int, float] = {}
        for rank, (gid, _) in enumerate(dense):
            fused[gid] = fused.get(gid, 0.0) + self.dense_weight / (self.rrf_k + rank + 1)
        for rank, (gid, _) in enumerate(sparse):
            fused[gid] = fused.get(gid, 0.0) + self.sparse_weight / (self.rrf_k + rank + 1)
        top = sorted(fused.items(), key=lambda x: -x[1])[:k]
        docs = {d["id"]: d for d in self.store.get([gid for gid, _ in top])}
        return [{"id": gid, "score": s,
                 "text": docs[gid]["text"], "meta": docs[gid]["meta"]}
                for gid, s in top if gid in docs]
