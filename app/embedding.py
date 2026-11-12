"""Local sentence embeddings, L2-normalized (dot product = cosine similarity)."""

from __future__ import annotations

import threading

import numpy as np

from app.config import settings


class Embedder:
    def __init__(self, model_name: str | None = None):
        self.name = model_name or settings.EMBEDDING_MODEL_NAME
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.name)
            return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Texts -> float32 matrix (N, dim), each row of norm 1."""
        vectors = self._get_model().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
