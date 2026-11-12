"""Fake embedder (no model download) and app factory."""

import re
import zlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.cache.store import SemanticCache
from app.llm import MockLLM
from app.main import create_app

DIM = 256
# Synonyms mapped to the same bucket, so paraphrases get a high similarity.
SYNONYMS = {"parcel": "order", "package": "order", "where's": "where", "reset": "forgot", "deliver": "ship"}
STOPWORDS = {"the", "a", "my", "is", "do", "you", "i", "can", "to", "please", "of", "for", "how"}


class FakeEmbedder:
    name = "fake-bag-of-words"

    def encode(self, texts):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9']+", text.lower()):
                word = SYNONYMS.get(word, word)
                if word not in STOPWORDS:
                    out[i, zlib.crc32(word.encode()) % DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    def encode_one(self, text):
        return self.encode([text])[0]


@pytest.fixture()
def make_client():
    def _make(threshold=0.8, guard=True, ttl=3600, max_entries=1000, latency_ms=0):
        llm = MockLLM(latency_ms=latency_ms)
        cache = SemanticCache(ttl_seconds=ttl, max_entries=max_entries, guard_enabled=guard)
        app = create_app(llm=llm, embedder=FakeEmbedder(), cache=cache, threshold=threshold)
        return TestClient(app), llm, cache

    return _make
