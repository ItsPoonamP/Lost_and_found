"""
search.py — FAISS vector index for fast similarity search.

Uses IndexFlatIP (Inner Product) on L2-normalized embeddings,
which is equivalent to cosine similarity. Score range: [0.0, 1.0].

faiss_map: { faiss_position → observation DB id }
Both index and map are persisted to disk automatically.
"""

import faiss
import json
import numpy as np
import os
import config


class VectorSearch:
    def __init__(self):
        self._load_or_create()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load_or_create(self):
        if os.path.exists(config.FAISS_INDEX) and os.path.exists(config.FAISS_MAP):
            self.index = faiss.read_index(config.FAISS_INDEX)
            with open(config.FAISS_MAP) as f:
                raw = json.load(f)
            # JSON keys are always strings; convert back to int
            self.faiss_map = {int(k): int(v) for k, v in raw.items()}
            self._next_id = max(self.faiss_map.keys(), default=-1) + 1
            print(f"  FAISS index loaded — {self.index.ntotal} vectors")
        else:
            # IndexFlatIP = exact cosine search (no approximation needed at small scale)
            self.index    = faiss.IndexFlatIP(config.EMBEDDING_DIM)
            self.faiss_map = {}
            self._next_id  = 0

    def save(self):
        faiss.write_index(self.index, config.FAISS_INDEX)
        with open(config.FAISS_MAP, "w") as f:
            json.dump(self.faiss_map, f)

    # ── Write ──────────────────────────────────────────────────────────────────

    def add(self, embedding: np.ndarray, obs_id: int) -> int:
        """
        Add one embedding to the index.
        Returns the faiss_id (position in the index).
        """
        vec = embedding.reshape(1, -1).astype(np.float32)
        self.index.add(vec)
        faiss_id = self._next_id
        self.faiss_map[faiss_id] = obs_id
        self._next_id += 1
        return faiss_id

    def update_obs_id(self, faiss_id: int, obs_id: int):
        """Update mapping after DB insert gives us the real obs_id."""
        self.faiss_map[faiss_id] = obs_id

    # ── Read ───────────────────────────────────────────────────────────────────

    def search(self, query: np.ndarray, top_k: int = None) -> list[tuple[int, float]]:
        """
        Args:
            query: normalized float32 numpy array (EMBEDDING_DIM,)
            top_k: number of results

        Returns:
            list of (obs_id, cosine_score) sorted by score descending
        """
        if self.index.ntotal == 0:
            return []

        k = min(top_k or config.TOP_K, self.index.ntotal)
        vec = query.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            obs_id = self.faiss_map.get(int(idx))
            if obs_id is not None and obs_id > 0:
                results.append((obs_id, float(score)))

        return results   # already sorted by FAISS (highest score first)

    @property
    def total(self) -> int:
        return self.index.ntotal
