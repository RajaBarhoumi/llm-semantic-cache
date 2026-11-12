"""Cache store: partitions, expiry, eviction, guard."""

import numpy as np

from app.cache.store import SemanticCache
from tests.conftest import FakeEmbedder

E = FakeEmbedder()
KEY = ("default", "mock", "abc")


def add(cache, question, key=KEY, now=0.0):
    return cache.add(key, question, f"answer: {question}", E.encode_one(question), cost_usd=0.001, now=now)


def test_exact_and_semantic_lookup():
    cache = SemanticCache(ttl_seconds=100, max_entries=10)
    add(cache, "Where is my order?")
    assert cache.lookup_exact(KEY, "where is my ORDER", now=1).kind == "exact"
    q = "where's my parcel"
    found = cache.lookup_semantic(KEY, q, E.encode_one(q), threshold=0.8, now=1)
    assert found.kind == "semantic" and found.similarity >= 0.8


def test_partitions_are_isolated():
    cache = SemanticCache(ttl_seconds=100, max_entries=10)
    add(cache, "Where is my order?", key=("tenant-a", "mock", "abc"))
    assert cache.lookup_exact(("tenant-b", "mock", "abc"), "Where is my order?", now=1).kind is None
    assert cache.lookup_exact(("tenant-a", "mock", "other-prompt"), "Where is my order?", now=1).kind is None


def test_expired_entries_are_ignored():
    cache = SemanticCache(ttl_seconds=10, max_entries=10)
    add(cache, "Where is my order?", now=0)
    assert cache.lookup_exact(KEY, "Where is my order?", now=11).kind is None
    q = "where's my parcel"
    assert cache.lookup_semantic(KEY, q, E.encode_one(q), 0.5, now=11).kind is None


def test_lru_eviction():
    cache = SemanticCache(ttl_seconds=100, max_entries=2)
    add(cache, "question one", now=0)
    add(cache, "question two", now=1)
    cache.lookup_exact(KEY, "question one", now=2)  # one is now more recent than two
    add(cache, "question three", now=3)
    assert cache.size() == 2
    assert cache.lookup_exact(KEY, "question two", now=4).kind is None
    assert cache.lookup_exact(KEY, "question one", now=4).kind == "exact"


def test_guard_rejects_and_reports_the_candidate():
    cache = SemanticCache(ttl_seconds=100, max_entries=10, guard_enabled=True)
    add(cache, "invoice for order 1001")
    q = "invoice for order 1002"
    found = cache.lookup_semantic(KEY, q, E.encode_one(q), threshold=0.5, now=1)
    assert found.kind is None
    assert found.guard_rejected.question == "invoice for order 1001"

    unguarded = SemanticCache(ttl_seconds=100, max_entries=10, guard_enabled=False)
    add(unguarded, "invoice for order 1001")
    assert unguarded.lookup_semantic(KEY, q, E.encode_one(q), threshold=0.5, now=1).kind == "semantic"


def test_readding_a_question_replaces_it():
    cache = SemanticCache(ttl_seconds=100, max_entries=10)
    add(cache, "Where is my order?")
    add(cache, "where is my order")
    assert cache.size() == 1


def test_delete_and_clear():
    cache = SemanticCache(ttl_seconds=100, max_entries=10)
    entry = add(cache, "question one")
    add(cache, "question two", key=("other", "mock", "abc"))
    assert cache.delete(entry.id) and not cache.delete(entry.id)
    assert cache.clear("other") == 1 and cache.size() == 0


def test_nearest_returns_sorted_candidates():
    cache = SemanticCache(ttl_seconds=100, max_entries=10)
    add(cache, "where is my order")
    add(cache, "reset my password")
    results = cache.nearest(KEY, E.encode_one("where's my parcel"), k=2)
    assert results[0][0].question == "where is my order"
    assert results[0][1] >= results[1][1]
    assert np.isclose(np.linalg.norm(results[0][0].vector), 1.0)


def test_buffer_growth_and_compaction_keep_lookups_correct():
    cache = SemanticCache(ttl_seconds=100, max_entries=1000)
    entries = [add(cache, f"question number {i}") for i in range(100)]
    for entry in entries[:60]:  # more than half dead -> compaction
        cache.delete(entry.id)
    assert cache.size() == 40
    for i in (60, 99):
        q = f"question number {i}"
        found = cache.lookup_semantic(KEY, q, E.encode_one(q), threshold=0.99, now=1)
        assert found.kind == "semantic" and found.entry.question == q
    assert cache.lookup_exact(KEY, "question number 10", now=1).kind is None
