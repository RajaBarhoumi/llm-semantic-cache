"""Replay realistic support traffic through the cache and measure what it saves.

    python scripts/replay.py --llm mock                     # offline, estimated costs
    python scripts/replay.py --llm openai --with-baseline   # real model, real costs and latency

Traffic: requests drawn from the held-out (test) intents of eval/intents.json
with a Zipf distribution (a few questions are very frequent, most are rare),
each request being a random paraphrase of its intent. Fixed seed.

A hit is correct when the cached question belongs to the same intent as the
request; otherwise it is a false hit (a wrong answer served from the cache).
`--with-baseline` replays the same traffic without cache to measure the real
no-cache cost and latency.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.cache.guards import normalize  # noqa: E402
from app.cache.store import SemanticCache  # noqa: E402
from app.config import settings  # noqa: E402
from app.embedding import Embedder  # noqa: E402
from app.llm import build_llm  # noqa: E402
from app.service import CacheService, percentile  # noqa: E402

INTENTS_PATH = ROOT / "eval" / "intents.json"


def load_intents(path: Path = INTENTS_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["intents"]


def build_traffic(intents: list[dict], n: int = 300, seed: int = 42, split: str = "test", zipf_s: float = 1.1):
    rng = random.Random(seed)
    pool = [i for i in intents if split == "all" or i["split"] == split]
    rng.shuffle(pool)
    weights = [1 / (rank + 1) ** zipf_s for rank in range(len(pool))]
    return [(intent["id"], rng.choice(intent["variants"])) for intent in rng.choices(pool, weights, k=n)]


def variant_to_intent(intents: list[dict]) -> dict[str, str]:
    return {v: i["id"] for i in intents for v in i["variants"]}


async def replay(service: CacheService, traffic, intent_of: dict[str, str]) -> dict:
    rows, seen_intents, seen_exact = [], set(), set()
    for intent_id, question in traffic:
        res = await service.ask(question)
        status, kind = res["cache"]["status"], res["cache"]["kind"]
        matched_intent = intent_of.get(res["cache"]["matched_question"])
        rows.append(
            {
                "intent": intent_id,
                "question": question,
                "status": status,
                "kind": kind,
                "similarity": res["cache"]["similarity"],
                "matched_question": res["cache"]["matched_question"],
                "correct_hit": status == "hit" and matched_intent == intent_id,
                "false_hit": status == "hit" and matched_intent != intent_id,
                "reachable": intent_id in seen_intents,  # a correct hit was possible
                "exact_repeat": normalize(question) in seen_exact,
                "guard_rejected": res["cache"]["guard_rejected_question"] is not None,
                "latency_ms": res["latency_ms"],
                "cost_usd": res["cost_usd"],
                "saved_cost_usd": res["saved_cost_usd"],
            }
        )
        seen_intents.add(intent_id)
        seen_exact.add(normalize(question))
    return summarize(rows)


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    hits = [r for r in rows if r["status"] == "hit"]
    misses = [r for r in rows if r["status"] == "miss"]
    reachable = [r for r in rows if r["reachable"]]
    spent = sum(r["cost_usd"] for r in rows)
    saved = sum(r["saved_cost_usd"] for r in rows)
    return {
        "requests": n,
        "llm_calls": len(misses),
        "hits": len(hits),
        "exact_hits": sum(r["kind"] == "exact" for r in hits),
        "semantic_hits": sum(r["kind"] == "semantic" for r in hits),
        "hit_rate": len(hits) / n,
        "exact_only_hit_rate": sum(r["exact_repeat"] for r in rows) / n,
        "ideal_hit_rate": len(reachable) / n,
        "correct_hits": sum(r["correct_hit"] for r in rows),
        "false_hits": sum(r["false_hit"] for r in rows),
        "guard_rejections": sum(r["guard_rejected"] for r in rows),
        "recall_of_reachable": sum(r["correct_hit"] for r in reachable) / len(reachable) if reachable else 0.0,
        "cost_spent_usd": spent,
        "cost_saved_usd": saved,
        "savings_ratio": saved / (spent + saved) if spent + saved else 0.0,
        "hit_latency_ms": {"p50": percentile([r["latency_ms"] for r in hits], 50), "p95": percentile([r["latency_ms"] for r in hits], 95)},
        "miss_latency_ms": {"p50": percentile([r["latency_ms"] for r in misses], 50), "p95": percentile([r["latency_ms"] for r in misses], 95)},
        "rows": rows,
    }


async def baseline(llm, traffic) -> dict:
    """Same traffic, no cache: every request calls the model."""
    from app.llm import cost_usd

    costs, latencies = [], []
    for _intent, question in traffic:
        completion = await llm.complete(settings.DEFAULT_SYSTEM_PROMPT, question)
        costs.append(cost_usd(completion.input_tokens, completion.output_tokens))
        latencies.append(completion.latency_ms)
    return {"cost_usd": sum(costs), "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95), "mean": statistics.mean(latencies)}}


def build_service(llm, embedder, threshold: float, guard: bool) -> CacheService:
    cache = SemanticCache(settings.CACHE_TTL_SECONDS, settings.CACHE_MAX_ENTRIES, guard_enabled=guard)
    return CacheService(llm, embedder, cache, threshold, settings.DEFAULT_SYSTEM_PROMPT)


def fmt_ms(value) -> str:
    return "n/a" if value is None else f"{value:.0f} ms" if value >= 10 else f"{value:.1f} ms"


def render(result: dict, llm_name: str, threshold: float, base: dict | None, traffic_size: int) -> str:
    r = result
    lines = [
        f"# Replay with `{llm_name}` ({date.today().isoformat()})",
        "",
        f"- LLM endpoint: {settings.OPENAI_BASE_URL or 'https://api.openai.com/v1'}"
        f" (costs priced at ${settings.PRICE_INPUT_PER_MTOK} / ${settings.PRICE_OUTPUT_PER_MTOK} per million input / output tokens)",
        f"- Traffic: {traffic_size} requests, held-out intents, Zipf distribution, seed 42",
        f"- Embedding model `{settings.EMBEDDING_MODEL_NAME}`, threshold {threshold}, entity guard on",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Hit rate | **{100 * r['hit_rate']:.1f}%** ({r['hits']}/{r['requests']}: {r['exact_hits']} exact, {r['semantic_hits']} semantic) |",
        f"| Hit rate of an exact-match cache on the same traffic | {100 * r['exact_only_hit_rate']:.1f}% |",
        f"| Best possible hit rate (intent already seen) | {100 * r['ideal_hit_rate']:.1f}% |",
        f"| False hits (wrong answer served) | **{r['false_hits']}** |",
        f"| Hits blocked by the entity guard | {r['guard_rejections']} |",
        f"| LLM calls | {r['llm_calls']} instead of {r['requests']} |",
        f"| Latency of a hit, p50 / p95 | {fmt_ms(r['hit_latency_ms']['p50'])} / {fmt_ms(r['hit_latency_ms']['p95'])} |",
        f"| Latency of a miss, p50 / p95 | {fmt_ms(r['miss_latency_ms']['p50'])} / {fmt_ms(r['miss_latency_ms']['p95'])} |",
        f"| Cost with cache | ${r['cost_spent_usd']:.5f} |",
        f"| Cost avoided by hits | ${r['cost_saved_usd']:.5f} ({100 * r['savings_ratio']:.1f}%) |",
    ]
    if base:
        lines += [
            f"| **Measured cost without cache** | ${base['cost_usd']:.5f} |",
            f"| **Measured saving** | **{100 * (1 - r['cost_spent_usd'] / base['cost_usd']):.1f}%** |",
            f"| Measured LLM latency without cache, p50 / p95 | {fmt_ms(base['latency_ms']['p50'])} / {fmt_ms(base['latency_ms']['p95'])} |",
        ]
    return "\n".join(lines) + "\n"


async def main_async(args) -> None:
    intents = load_intents()
    traffic = build_traffic(intents, n=args.requests)
    llm = build_llm(args.llm)
    embedder = Embedder()
    embedder.encode_one("warm-up")  # model loading is excluded from latencies
    service = build_service(llm, embedder, args.threshold, guard=True)

    result = await replay(service, traffic, variant_to_intent(intents))
    base = await baseline(build_llm(args.llm), traffic) if args.with_baseline else None

    report = render(result, getattr(llm, "model", args.llm), args.threshold, base, len(traffic))
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / f"replay_{args.llm}.md").write_text(report, encoding="utf-8")
    (out / f"replay_{args.llm}_details.json").write_text(json.dumps(result["rows"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(report)
    print(f"Report written to {out / f'replay_{args.llm}.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default=settings.LLM_PROVIDER, help="openai | mock")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--threshold", type=float, default=settings.SIMILARITY_THRESHOLD)
    parser.add_argument("--with-baseline", action="store_true", help="also replay without cache (doubles the LLM calls)")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
