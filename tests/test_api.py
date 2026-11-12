"""HTTP API and service behavior."""

import asyncio

from app.cache.store import SemanticCache
from app.llm import MockLLM
from app.service import CacheService
from tests.conftest import FakeEmbedder


def ask(client, question, **extra):
    res = client.post("/ask", json={"question": question, **extra})
    assert res.status_code == 200
    return res.json()


def test_miss_then_exact_then_semantic_hit(make_client):
    client, llm, _ = make_client()
    first = ask(client, "Where is my order?")
    assert first["cache"]["status"] == "miss" and first["cost_usd"] > 0

    exact = ask(client, "where is my order")
    assert exact["cache"]["kind"] == "exact" and exact["cost_usd"] == 0
    assert exact["saved_cost_usd"] == first["cost_usd"]

    semantic = ask(client, "where's my parcel?")
    assert semantic["cache"]["kind"] == "semantic"
    assert semantic["cache"]["matched_question"] == "Where is my order?"
    assert semantic["answer"] == first["answer"]
    assert llm.calls == 1


def test_guard_turns_a_false_hit_into_a_miss(make_client):
    client, llm, _ = make_client(threshold=0.5)
    ask(client, "invoice for order 1001")
    res = ask(client, "invoice for order 1002")
    assert res["cache"]["status"] == "miss"
    assert res["cache"]["guard_rejected_question"] == "invoice for order 1001"
    assert llm.calls == 2


def test_namespaces_and_system_prompts_do_not_share_answers(make_client):
    client, llm, _ = make_client()
    ask(client, "Where is my order?", namespace="shop-a")
    assert ask(client, "Where is my order?", namespace="shop-b")["cache"]["status"] == "miss"
    assert ask(client, "Where is my order?", namespace="shop-a", system_prompt="Answer in French.")["cache"]["status"] == "miss"
    assert llm.calls == 3


def test_bypass_neither_reads_nor_writes(make_client):
    client, llm, cache = make_client()
    ask(client, "Where is my order?")
    res = ask(client, "Where is my order?", bypass_cache=True)
    assert res["cache"]["status"] == "bypass"
    assert cache.size() == 1 and llm.calls == 2


def test_stats(make_client):
    client, _, _ = make_client()
    ask(client, "Where is my order?")
    ask(client, "where is my order")
    ask(client, "where's my parcel")
    ask(client, "How do I reset my password?")
    stats = client.get("/stats").json()
    assert stats["requests"] == 4
    assert stats["hits"] == {"exact": 1, "semantic": 1, "coalesced": 0, "total": 2}
    assert stats["misses"] == 2 and stats["hit_rate"] == 0.5
    assert stats["cost_saved_usd"] > 0 and stats["entries"] == 2


def test_lookup_entries_and_deletion(make_client):
    client, _, _ = make_client()
    entry_id = ask(client, "Where is my order?")["cache"]["entry_id"]
    lookup = client.post("/cache/lookup", json={"question": "where's my parcel"}).json()
    assert lookup["candidates"][0]["question"] == "Where is my order?"
    assert lookup["candidates"][0]["above_threshold"] is True
    assert len(client.get("/cache/entries").json()) == 1
    assert client.delete(f"/cache/entries/{entry_id}").status_code == 200
    assert client.delete(f"/cache/entries/{entry_id}").status_code == 404
    assert client.delete("/cache").json() == {"removed": 0}


def test_identical_concurrent_questions_call_the_llm_once():
    llm = MockLLM(latency_ms=50)
    service = CacheService(llm, FakeEmbedder(), SemanticCache(3600, 100), threshold=0.8, default_system_prompt="sys")

    async def run():
        return await asyncio.gather(*(service.ask("Where is my order?") for _ in range(5)))

    results = asyncio.run(run())
    assert llm.calls == 1
    assert sorted(r["cache"]["status"] for r in results) == ["hit"] * 4 + ["miss"]
    assert service.stats.coalesced == 4


def test_llm_error_propagates_and_releases_the_flight():
    class Failing(MockLLM):
        async def complete(self, system_prompt, question):
            raise RuntimeError("upstream down")

    service = CacheService(Failing(), FakeEmbedder(), SemanticCache(3600, 100), threshold=0.8, default_system_prompt="sys")
    for _ in range(2):  # the second call must not hang on a stale in-flight future
        try:
            asyncio.run(service.ask("Where is my order?"))
        except RuntimeError as exc:
            assert str(exc) == "upstream down"
    assert service._inflight == {}


def test_rate_limit_wait_is_not_counted_as_latency():
    from app.llm import Completion

    class Paced(MockLLM):
        async def complete(self, system_prompt, question):
            await asyncio.sleep(0.2)  # simulated wait for a rate-limit slot
            return Completion(text="ok", input_tokens=10, output_tokens=5, latency_ms=1.0, queued_ms=200.0)

    service = CacheService(Paced(), FakeEmbedder(), SemanticCache(3600, 100), threshold=0.8, default_system_prompt="sys")
    res = asyncio.run(service.ask("Where is my order?"))
    assert res["queued_ms"] == 200.0
    assert res["latency_ms"] < 100
    assert service.stats.miss_latencies[0] < 100
