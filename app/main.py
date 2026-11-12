"""Entry point: uvicorn app.main:app --reload"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.cache.store import SemanticCache
from app.config import settings
from app.embedding import Embedder
from app.llm import LLMClient, build_llm
from app.service import CacheService


def create_app(
    llm: LLMClient | None = None,
    embedder=None,
    cache: SemanticCache | None = None,
    threshold: float | None = None,
) -> FastAPI:
    """Build the app; every dependency can be injected (tests, evaluation)."""
    embedder = embedder or Embedder()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Load the embedding model at startup, not on the first request (it takes seconds).
        await asyncio.to_thread(embedder.encode_one, "warm-up")
        yield

    app = FastAPI(title="LLM Semantic Cache", version="1.0.0", lifespan=lifespan)
    app.state.service = CacheService(
        llm=llm or build_llm(),
        embedder=embedder,
        cache=cache
        or SemanticCache(
            ttl_seconds=settings.CACHE_TTL_SECONDS,
            max_entries=settings.CACHE_MAX_ENTRIES,
            guard_enabled=settings.ENTITY_GUARD_ENABLED,
        ),
        threshold=settings.SIMILARITY_THRESHOLD if threshold is None else threshold,
        default_system_prompt=settings.DEFAULT_SYSTEM_PROMPT,
    )
    app.include_router(router)
    return app


app = create_app()
