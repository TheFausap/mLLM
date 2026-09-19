"""DiskVectorStore: SQLite text + memmapped int8 vectors + BM25 sidecar.

Design for the DGX Spark's 4TB SSD:
- texts + metadata in SQLite (WAL mode, indexed)
- dense vectors int8-quantized, stored as numpy memmap shards (no RAM blowup)
- BM25 over tokenized texts (rank_bm25 if installed, else pure-python TF-IDF)

20M chunks x 384-dim int8 ≈ 7.7GB. Search streams over shards.
Torch-free so the memory server can run alongside training.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from typing import List, Dict, Optional, Tuple

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower()[:8000])


class DiskVectorStore:
    def __init__(self, path: str, dim: int = 384, shard_size: int = 200_000):
        self.path = path
        self.dim = dim
        self.shard_size = shard_size
        os.makedirs(path, exist_ok=True)
        self.db_path = os.path.join(path, "texts.db")
        self.con = sqlite3.connect(self.db_path)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS chunks "
            "(id INTEGER PRIMARY KEY, text TEXT, meta TEXT)")
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT)")
        self.con.commit()
        # bm25 corpus cache (built lazily)
        self._bm25 = None
        self._bm25_ids: List[int] = []
        self._scales: Optional[np.ndarray] = None

    # -- write --------------------------------------------------------------
    def add(self, texts: List[str], vectors: np.ndarray,
            metas: Optional[List[dict]] = None):
        """Append chunks. vectors: (N, dim) float32."""
        assert vectors.shape[1] == self.dim
        cur = self.con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        metas = metas or [{}] * len(texts)
        self.con.executemany(
            "INSERT INTO chunks (id, text, meta) VALUES (?,?,?)",
            [(cur + i, t, json.dumps(m)) for i, (t, m) in enumerate(zip(texts, metas))])
        self.con.commit()
        # quantize int8 per-row with stored scale
        v = vectors.astype(np.float32)
        scale = np.abs(v).max(axis=1, keepdims=True).clip(min=1e-8) / 127.0
        q = np.clip(np.round(v / scale), -127, 127).astype(np.int8)
        scales = scale.squeeze(1).astype(np.float32)
        for i in range(len(texts)):
            gid = cur + i
            shard = gid // self.shard_size
            off = gid % self.shard_size
            sp = os.path.join(self.path, f"vec_{shard:04d}.npy")
            qp = os.path.join(self.path, f"scale_{shard:04d}.npy")
            if not os.path.exists(sp):
                m = np.memmap(sp, dtype=np.int8, mode="w+",
                              shape=(self.shard_size, self.dim))
                m[:] = 0
                del m
                s = np.memmap(qp, dtype=np.float32, mode="w+",
                              shape=(self.shard_size,))
                s[:] = 1.0
                del s
            m = np.memmap(sp, dtype=np.int8, mode="r+", shape=(self.shard_size, self.dim))
            m[off] = q[i]
            del m
            s = np.memmap(qp, dtype=np.float32, mode="r+", shape=(self.shard_size,))
            s[off] = scales[i]
            del s
        self._bm25 = None  # invalidate

    def __len__(self):
        return self.con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def get(self, ids: List[int]) -> List[Dict]:
        out = []
        for i in ids:
            row = self.con.execute("SELECT text, meta FROM chunks WHERE id=?", (i,)).fetchone()
            if row:
                out.append({"id": i, "text": row[0], "meta": json.loads(row[1] or "{}")})
        return out

    # -- dense search ---------------------------------------------------------
    def _all_scales(self, n: int) -> np.ndarray:
        scales = np.ones(n, dtype=np.float32)
        for shard in range(n // self.shard_size + 1):
            qp = os.path.join(self.path, f"scale_{shard:04d}.npy")
            if not os.path.exists(qp):
                continue
            s = np.memmap(qp, dtype=np.float32, mode="r", shape=(self.shard_size,))
            lo, hi = shard * self.shard_size, min(n, (shard + 1) * self.shard_size)
            scales[lo:hi] = np.array(s[:hi - lo])
            del s
        return scales

    def dense_search(self, query_vec: np.ndarray, k: int = 8) -> List[Tuple[int, float]]:
        """Brute-force int8 search over shards (exact, streams from disk)."""
        n = len(self)
        if n == 0:
            return []
        q = query_vec.astype(np.float32)
        qn = q / (np.linalg.norm(q) + 1e-8)
        best: List[Tuple[int, float]] = []
        # accumulate top-k across shards
        import heapq
        heap: List[Tuple[float, int]] = []
        for shard in range(n // self.shard_size + 1):
            sp = os.path.join(self.path, f"vec_{shard:04d}.npy")
            if not os.path.exists(sp):
                continue
            lo = shard * self.shard_size
            hi = min(n, lo + self.shard_size)
            m = np.memmap(sp, dtype=np.int8, mode="r", shape=(self.shard_size, self.dim))
            block = np.array(m[:hi - lo]).astype(np.float32)
            del m
            scales = self._all_scales(n)[lo:hi]
            block = block * scales[:, None]
            norms = np.linalg.norm(block, axis=1, keepdims=True).clip(min=1e-8)
            sims = (block / norms) @ qn
            for j, s in enumerate(sims):
                gid = lo + j
                if len(heap) < k:
                    heapq.heappush(heap, (float(s), gid))
                elif s > heap[0][0]:
                    heapq.heapreplace(heap, (float(s), gid))
        return sorted([(gid, s) for s, gid in heap], key=lambda x: -x[1])

    # -- sparse (BM25 / TF-IDF fallback) ---------------------------------------
    def _ensure_bm25(self):
        if self._bm25 is not None:
            return
        rows = self.con.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
        self._bm25_ids = [r[0] for r in rows]
        corpus = [tokenize(r[1]) for r in rows]
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = ("bm25", BM25Okapi(corpus))
        except ImportError:
            # pure-python TF-IDF fallback
            from collections import Counter
            import math
            df = Counter()
            tfs = []
            for doc in corpus:
                c = Counter(doc)
                tfs.append(c)
                for t in c:
                    df[t] += 1
            N = max(1, len(corpus))
            idf = {t: math.log(1 + N / (1 + f)) for t, f in df.items()}
            self._bm25 = ("tfidf", (tfs, idf))

    def sparse_search(self, query: str, k: int = 8) -> List[Tuple[int, float]]:
        self._ensure_bm25()
        if not self._bm25_ids:
            return []
        kind, obj = self._bm25
        qtok = tokenize(query)
        if kind == "bm25":
            scores = obj.get_scores(qtok)
        else:
            from collections import Counter
            tfs, idf = obj
            qc = Counter(qtok)
            scores = np.zeros(len(tfs))
            for i, tf in enumerate(tfs):
                scores[i] = sum(qc[t] * tf.get(t, 0) * idf.get(t, 0.0) for t in qc)
        idx = np.argsort(-scores)[:k]
        return [(self._bm25_ids[i], float(scores[i])) for i in idx if scores[i] > 0]

    # -- kv helpers (episodic memory, config) -----------------------------------
    def kv_set(self, key: str, value: str):
        self.con.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, value))
        self.con.commit()

    def kv_get(self, key: str) -> Optional[str]:
        row = self.con.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def close(self):
        self.con.close()
