"""HTTP API of the semantic cache."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from app.api.schemas import AskRequest, LookupRequest
from app.cache.guards import compatible
from app.service import CacheService

router = APIRouter()


def _service(request: Request) -> CacheService:
    return request.app.state.service


@router.post("/ask")
async def ask(payload: AskRequest, request: Request):
    """Answer from the cache when a question with the same meaning was already answered."""
    return await _service(request).ask(payload.question, payload.namespace, payload.system_prompt, payload.bypass_cache)


@router.post("/cache/lookup")
async def lookup(payload: LookupRequest, request: Request):
    """Diagnostic: nearest cached questions and what the cache would decide, without calling the LLM."""
    service = _service(request)
    key = service.partition(payload.namespace, payload.system_prompt or service.default_system_prompt)
    vector = await service.embed(payload.question)
    return {
        "threshold": service.threshold,
        "guard_enabled": service.cache.guard_enabled,
        "candidates": [
            {
                "entry_id": entry.id,
                "question": entry.question,
                "similarity": round(similarity, 4),
                "above_threshold": similarity >= service.threshold,
                "guard_ok": compatible(payload.question, entry.question),
            }
            for entry, similarity in service.cache.nearest(key, vector, payload.k)
        ],
    }


@router.get("/cache/entries")
async def list_entries(request: Request, namespace: str | None = Query(None)):
    entries = _service(request).cache.entries(namespace)
    return [
        {"id": e.id, "namespace": e.partition[0], "question": e.question, "hits": e.hits, "cost_usd": e.cost_usd, "expires_at": e.expires_at}
        for e in sorted(entries, key=lambda e: -e.hits)
    ]


@router.delete("/cache/entries/{entry_id}")
async def delete_entry(entry_id: int, request: Request):
    if not _service(request).cache.delete(entry_id):
        raise HTTPException(404, "Unknown entry")
    return {"deleted": entry_id}


@router.delete("/cache")
async def clear(request: Request, namespace: str | None = Query(None)):
    return {"removed": _service(request).cache.clear(namespace)}


@router.get("/stats")
async def stats(request: Request):
    service = _service(request)
    return {**service.stats.as_dict(), "entries": service.cache.size()}


@router.get("/health")
async def health(request: Request):
    service = _service(request)
    return {
        "status": "ok",
        "llm": service.llm.model,
        "embedding_model": getattr(service.embedder, "name", "custom"),
        "threshold": service.threshold,
        "guard_enabled": service.cache.guard_enabled,
    }
