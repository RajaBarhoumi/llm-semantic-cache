"""In-memory semantic cache.

Entries are partitioned by (namespace, model, system prompt): an answer is only
ever reused for the same tenant, the same model and the same instructions.
Inside a partition, lookup is:

1. exact match on the normalized question (dictionary, no embedding needed);
2. semantic match: brute-force cosine similarity against every live entry
   (one matrix product, fast up to ~100k entries per partition), then the best
   candidate above the threshold that also passes the entity guard.

Expired entries are skipped at lookup and purged lazily. Beyond
`max_entries`, the least recently used entry is evicted.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass

import numpy as np

from app.cache.guards import compatible, normalize

PartitionKey = tuple[str, str, str]  # (namespace, model, system prompt hash)

_ids = itertools.count(1)


@dataclass
class Entry:
    id: int
    partition: PartitionKey
    question: str
    normalized: str
    answer: str
    vector: np.ndarray
    created_at: float
    expires_at: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    hits: int = 0
    last_used_at: float = 0.0


@dataclass
class LookupResult:
    kind: str | None  # "exact" | "semantic" | None (miss)
    entry: Entry | None = None
    similarity: float | None = None
    # Best candidate above the threshold that the entity guard rejected, if any.
    guard_rejected: Entry | None = None
    guard_rejected_similarity: float | None = None


class _Partition:
    """Entries of one partition plus a growable vector buffer.

    Rows are appended in O(1) amortized (capacity doubling). A removed entry
    leaves a dead row that search ignores; the buffer is compacted once more
    than half of the rows are dead.
    """

    def __init__(self):
        self.entries: dict[int, Entry] = {}
        self.exact: dict[str, int] = {}
        self._vectors: np.ndarray | None = None  # (capacity, dim)
        self._alive: np.ndarray | None = None  # (capacity,) bool
        self._row_ids: list[int | None] = []
        self._rows: dict[int, int] = {}  # entry id -> row
        self._dead = 0

    def insert(self, entry: Entry) -> None:
        if self._vectors is None:
            self._vectors = np.zeros((16, entry.vector.shape[0]), dtype=np.float32)
            self._alive = np.zeros(16, dtype=bool)
        row = len(self._row_ids)
        if row == self._vectors.shape[0]:
            self._vectors = np.concatenate([self._vectors, np.zeros_like(self._vectors)])
            self._alive = np.concatenate([self._alive, np.zeros_like(self._alive)])
        self._vectors[row] = entry.vector
        self._alive[row] = True
        self._row_ids.append(entry.id)
        self._rows[entry.id] = row
        self.entries[entry.id] = entry
        self.exact[entry.normalized] = entry.id

    def remove(self, entry_id: int) -> Entry | None:
        entry = self.entries.pop(entry_id, None)
        if entry is None:
            return None
        if self.exact.get(entry.normalized) == entry_id:
            del self.exact[entry.normalized]
        row = self._rows.pop(entry_id)
        self._alive[row] = False
        self._row_ids[row] = None
        self._dead += 1
        if self._dead > len(self._row_ids) // 2:
            self._compact()
        return entry

    def _compact(self) -> None:
        live = [i for i in self._row_ids if i is not None]
        self._vectors = self._alive = None
        self._row_ids, self._rows, self._dead = [], {}, 0
        for entry_id in live:
            self.insert(self.entries[entry_id])

    def similarities(self, vector: np.ndarray) -> tuple[list[int | None], np.ndarray]:
        """Cosine similarity with every row; dead rows get -inf."""
        n = len(self._row_ids)
        if n == 0:
            return [], np.empty(0, dtype=np.float32)
        sims = self._vectors[:n] @ vector
        sims[~self._alive[:n]] = -np.inf
        return self._row_ids, sims


class SemanticCache:
    def __init__(self, ttl_seconds: int, max_entries: int, guard_enabled: bool = True):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.guard_enabled = guard_enabled
        self._partitions: dict[PartitionKey, _Partition] = {}
        self._lock = threading.Lock()
        self._last_purge = float("-inf")

    # ------------------------------------------------------------------ lookup

    def lookup_exact(self, key: PartitionKey, question: str, now: float | None = None) -> LookupResult:
        now = time.time() if now is None else now
        with self._lock:
            part = self._partitions.get(key)
            entry_id = part.exact.get(normalize(question)) if part else None
            entry = part.entries.get(entry_id) if entry_id is not None else None
            if entry is None or entry.expires_at <= now:
                return LookupResult(kind=None)
            self._touch(entry, now)
            return LookupResult(kind="exact", entry=entry, similarity=1.0)

    def lookup_semantic(
        self, key: PartitionKey, question: str, vector: np.ndarray, threshold: float, now: float | None = None
    ) -> LookupResult:
        now = time.time() if now is None else now
        with self._lock:
            part = self._partitions.get(key)
            if part is None or not part.entries:
                return LookupResult(kind=None)
            row_ids, sims = part.similarities(vector)
            # Only sort the candidates above the threshold, not the whole partition.
            candidates = np.flatnonzero(sims >= threshold)
            order = candidates[np.argsort(-sims[candidates])]

            result = LookupResult(kind=None)
            for idx in order:
                similarity = float(sims[idx])
                entry = part.entries[row_ids[idx]]
                if entry.expires_at <= now:
                    continue
                if self.guard_enabled and not compatible(question, entry.question):
                    if result.guard_rejected is None:
                        result.guard_rejected, result.guard_rejected_similarity = entry, similarity
                    continue
                self._touch(entry, now)
                result.kind, result.entry, result.similarity = "semantic", entry, similarity
                return result
            return result

    def nearest(self, key: PartitionKey, vector: np.ndarray, k: int = 5) -> list[tuple[Entry, float]]:
        """Diagnostic: the k most similar entries, without threshold or guard."""
        with self._lock:
            part = self._partitions.get(key)
            if part is None or not part.entries:
                return []
            row_ids, sims = part.similarities(vector)
            top = np.argsort(-sims)[: min(k, len(part.entries))]
            return [(part.entries[row_ids[i]], float(sims[i])) for i in top]

    # ------------------------------------------------------------------ writes

    def add(
        self,
        key: PartitionKey,
        question: str,
        answer: str,
        vector: np.ndarray,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        now: float | None = None,
    ) -> Entry:
        now = time.time() if now is None else now
        normalized = normalize(question)
        with self._lock:
            part = self._partitions.setdefault(key, _Partition())
            previous = part.exact.get(normalized)
            if previous is not None:
                part.remove(previous)
            entry = Entry(
                id=next(_ids),
                partition=key,
                question=question,
                normalized=normalized,
                answer=answer,
                vector=np.asarray(vector, dtype=np.float32),
                created_at=now,
                expires_at=now + self.ttl_seconds,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                last_used_at=now,
            )
            part.insert(entry)
            self._purge_expired(now)
            self._evict_lru()
            return entry

    def delete(self, entry_id: int) -> bool:
        with self._lock:
            return any(part.remove(entry_id) is not None for part in self._partitions.values())

    def clear(self, namespace: str | None = None) -> int:
        with self._lock:
            keys = [k for k in self._partitions if namespace is None or k[0] == namespace]
            removed = sum(len(self._partitions[k].entries) for k in keys)
            for k in keys:
                del self._partitions[k]
            return removed

    # ------------------------------------------------------------------ helpers

    def size(self) -> int:
        with self._lock:
            return sum(len(p.entries) for p in self._partitions.values())

    def entries(self, namespace: str | None = None) -> list[Entry]:
        with self._lock:
            return [
                e for k, p in self._partitions.items() if namespace is None or k[0] == namespace for e in p.entries.values()
            ]

    @staticmethod
    def _touch(entry: Entry, now: float) -> None:
        entry.hits += 1
        entry.last_used_at = now

    def _purge_expired(self, now: float) -> None:
        # Runs at most once per second: a full scan on every insert would be O(N).
        if now - self._last_purge < 1.0:
            return
        self._last_purge = now
        for part in self._partitions.values():
            for entry in [e for e in part.entries.values() if e.expires_at <= now]:
                part.remove(entry.id)

    def _evict_lru(self) -> None:
        total = sum(len(p.entries) for p in self._partitions.values())
        if total <= self.max_entries:
            return
        everything = sorted(
            (e for p in self._partitions.values() for e in p.entries.values()), key=lambda e: e.last_used_at
        )
        for entry in everything[: total - self.max_entries]:
            self._partitions[entry.partition].remove(entry.id)
