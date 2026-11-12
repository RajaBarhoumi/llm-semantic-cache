"""The /ask flow: exact lookup, semantic lookup, otherwise call the LLM and store.

Identical questions arriving at the same time are coalesced ("single flight"):
only the first one calls the LLM, the others wait for its answer.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import deque
from dataclasses import dataclass, field

from app.cache.guards import normalize
from app.cache.store import PartitionKey, SemanticCache
from app.llm import LLMClient, cost_usd


def percentile(values, p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))]


@dataclass
class Stats:
    requests: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    coalesced: int = 0
    misses: int = 0
    bypassed: int = 0
    guard_rejections: int = 0
    cost_spent_usd: float = 0.0
    cost_saved_usd: float = 0.0
    tokens_saved: int = 0
    hit_latencies: deque = field(default_factory=lambda: deque(maxlen=5000))
    miss_latencies: deque = field(default_factory=lambda: deque(maxlen=5000))

    def as_dict(self) -> dict:
        hits = self.exact_hits + self.semantic_hits + self.coalesced
        served = hits + self.misses
        return {
            "requests": self.requests,
            "hits": {"exact": self.exact_hits, "semantic": self.semantic_hits, "coalesced": self.coalesced, "total": hits},
            "misses": self.misses,
            "bypassed": self.bypassed,
            "hit_rate": round(hits / served, 4) if served else None,
            "guard_rejections": self.guard_rejections,
            "cost_spent_usd": round(self.cost_spent_usd, 6),
            "cost_saved_usd": round(self.cost_saved_usd, 6),
            "tokens_saved": self.tokens_saved,
            "latency_ms": {
                "hit_p50": percentile(self.hit_latencies, 50),
                "hit_p95": percentile(self.hit_latencies, 95),
                "miss_p50": percentile(self.miss_latencies, 50),
                "miss_p95": percentile(self.miss_latencies, 95),
            },
        }


class CacheService:
    def __init__(self, llm: LLMClient, embedder, cache: SemanticCache, threshold: float, default_system_prompt: str):
        self.llm = llm
        self.embedder = embedder
        self.cache = cache
        self.threshold = threshold
        self.default_system_prompt = default_system_prompt
        self.stats = Stats()
        self._inflight: dict[tuple[PartitionKey, str], asyncio.Future] = {}

    def partition(self, namespace: str, system_prompt: str) -> PartitionKey:
        prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:12]
        return (namespace, self.llm.model, prompt_hash)

    async def embed(self, text: str):
        # Embedding is CPU-bound: keep it off the event loop.
        return await asyncio.to_thread(self.embedder.encode_one, text)

    async def ask(self, question: str, namespace: str = "default", system_prompt: str | None = None, bypass_cache: bool = False) -> dict:
        start = time.perf_counter()
        system_prompt = system_prompt or self.default_system_prompt
        key = self.partition(namespace, system_prompt)
        self.stats.requests += 1

        if bypass_cache:
            completion = await self.llm.complete(system_prompt, question)
            cost = cost_usd(completion.input_tokens, completion.output_tokens)
            self.stats.bypassed += 1
            self.stats.cost_spent_usd += cost
            return self._response(completion.text, "bypass", None, start, completion=completion, cost=cost)

        # 1. Exact match: no embedding needed.
        found = self.cache.lookup_exact(key, question)
        if found.entry is not None:
            self.stats.exact_hits += 1
            return self._hit(found, start)

        # 2. Semantic match.
        vector = await self.embed(question)
        found = self.cache.lookup_semantic(key, question, vector, self.threshold)
        if found.guard_rejected is not None:
            self.stats.guard_rejections += 1
        if found.entry is not None:
            self.stats.semantic_hits += 1
            return self._hit(found, start)

        # 3. Miss: call the LLM once per identical in-flight question.
        flight_key = (key, normalize(question))
        if flight_key in self._inflight:
            entry = await self._inflight[flight_key]
            self.stats.coalesced += 1
            self._record_saving(entry)
            self.stats.hit_latencies.append((time.perf_counter() - start) * 1000)
            return self._response(entry.answer, "hit", "coalesced", start, entry=entry, similarity=1.0)

        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._inflight[flight_key] = future
        try:
            completion = await self.llm.complete(system_prompt, question)
            cost = cost_usd(completion.input_tokens, completion.output_tokens)
            entry = self.cache.add(
                key,
                question,
                completion.text,
                vector,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                cost_usd=cost,
                latency_ms=completion.latency_ms,
            )
            future.set_result(entry)
        except Exception as exc:
            future.set_exception(exc)
            future.exception()  # mark as retrieved when nobody else is waiting
            raise
        finally:
            del self._inflight[flight_key]

        self.stats.misses += 1
        self.stats.cost_spent_usd += cost
        # Waiting for a rate-limit slot is a client-side choice, not latency of the cache or the model.
        self.stats.miss_latencies.append((time.perf_counter() - start) * 1000 - completion.queued_ms)
        return self._response(
            completion.text,
            "miss",
            None,
            start,
            entry=entry,
            completion=completion,
            cost=cost,
            guard_rejected=found.guard_rejected,
            guard_similarity=found.guard_rejected_similarity,
        )

    # ------------------------------------------------------------------ helpers

    def _record_saving(self, entry) -> None:
        self.stats.cost_saved_usd += entry.cost_usd
        self.stats.tokens_saved += entry.input_tokens + entry.output_tokens

    def _hit(self, found, start: float) -> dict:
        self._record_saving(found.entry)
        self.stats.hit_latencies.append((time.perf_counter() - start) * 1000)
        return self._response(found.entry.answer, "hit", found.kind, start, entry=found.entry, similarity=found.similarity)

    def _response(
        self,
        answer: str,
        status: str,
        kind: str | None,
        start: float,
        entry=None,
        similarity: float | None = None,
        completion=None,
        cost: float = 0.0,
        guard_rejected=None,
        guard_similarity: float | None = None,
    ) -> dict:
        is_hit = status == "hit"
        queued_ms = completion.queued_ms if completion else 0.0
        return {
            "answer": answer,
            "cache": {
                "status": status,  # hit | miss | bypass
                "kind": kind,  # exact | semantic | coalesced | None
                "similarity": round(similarity, 4) if similarity is not None else None,
                "matched_question": entry.question if is_hit and entry else None,
                "entry_id": entry.id if entry else None,
                "guard_rejected_question": guard_rejected.question if guard_rejected else None,
                "guard_rejected_similarity": round(guard_similarity, 4) if guard_similarity is not None else None,
            },
            "model": self.llm.model,
            "usage": {
                "input_tokens": completion.input_tokens if completion else 0,
                "output_tokens": completion.output_tokens if completion else 0,
            },
            "cost_usd": round(cost, 8),
            "saved_cost_usd": round(entry.cost_usd, 8) if is_hit and entry else 0.0,
            # Processing time; the rate-limit wait (if any) is reported separately.
            "latency_ms": round((time.perf_counter() - start) * 1000 - queued_ms, 2),
            "queued_ms": round(queued_ms, 2),
        }
