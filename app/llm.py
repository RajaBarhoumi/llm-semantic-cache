"""Upstream LLM clients: OpenAI (real) and a deterministic mock (offline)."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Protocol

from app.config import settings


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    # Time spent waiting for a rate-limit slot before the call (not part of the model's latency).
    queued_ms: float = 0.0


class LLMClient(Protocol):
    model: str

    async def complete(self, system_prompt: str, question: str) -> Completion: ...


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * settings.PRICE_INPUT_PER_MTOK + output_tokens * settings.PRICE_OUTPUT_PER_MTOK) / 1_000_000


class RateLimiter:
    """Spaces calls at least 60 / rpm seconds apart (free tiers limit requests per minute)."""

    def __init__(self, requests_per_minute: int):
        self.interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        if not self.interval:
            return
        async with self._lock:
            now = time.monotonic()
            if self._next > now:
                await asyncio.sleep(self._next - now)
            self._next = max(now, self._next) + self.interval


class OpenAIClient:
    """OpenAI, or any OpenAI-compatible provider through OPENAI_BASE_URL."""

    def __init__(self, model: str | None = None):
        self.model = model or settings.OPENAI_MODEL
        self.base_url = settings.OPENAI_BASE_URL or "https://api.openai.com/v1"
        self._client = None  # created on first call: the app starts without a key
        self._limiter = RateLimiter(settings.LLM_MAX_REQUESTS_PER_MINUTE)

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL or None, max_retries=settings.OPENAI_MAX_RETRIES
            )
        return self._client

    async def complete(self, system_prompt: str, question: str) -> Completion:
        queued_start = time.perf_counter()
        await self._limiter.wait()
        queued_ms = (time.perf_counter() - queued_start) * 1000
        start = time.perf_counter()
        response = await self._get_client().chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": question}],
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.MAX_OUTPUT_TOKENS,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        usage = response.usage
        return Completion(
            text=response.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            queued_ms=queued_ms,
        )


class MockLLM:
    """Deterministic answers; token counts are estimated (about 4 characters per token)."""

    def __init__(self, latency_ms: int | None = None):
        self.model = "mock"
        self.latency_ms = settings.MOCK_LATENCY_MS if latency_ms is None else latency_ms
        self.calls = 0

    async def complete(self, system_prompt: str, question: str) -> Completion:
        start = time.perf_counter()
        self.calls += 1
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000)
        text = f"[mock answer] {question.strip()}"
        return Completion(
            text=text,
            input_tokens=(len(system_prompt) + len(question)) // 4 + 8,
            output_tokens=len(text) // 4 + 40,
            latency_ms=(time.perf_counter() - start) * 1000,
        )


def build_llm(provider: str | None = None) -> LLMClient:
    provider = (provider or settings.LLM_PROVIDER).lower()
    if provider == "openai":
        return OpenAIClient()
    if provider == "mock":
        return MockLLM()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (expected 'openai' or 'mock')")
