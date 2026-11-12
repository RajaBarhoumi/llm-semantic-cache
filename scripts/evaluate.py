"""Offline evaluation of the semantic cache (no API key needed).

    python scripts/evaluate.py

1. Pair classification: every pair of questions from eval/intents.json is
   labeled "same answer" (same intent) or "different answer". Hard negatives
   are pairs of confusable intents (same group). The threshold is calibrated
   on the dev intents and reported on the held-out test intents, for each
   embedding model, with and without the entity guard.
2. Traffic replay with the mock LLM (see scripts/replay.py).
3. Latency: embedding a question, and similarity search at 1k / 10k / 100k entries.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.cache.guards import compatible  # noqa: E402
from app.cache.store import SemanticCache  # noqa: E402
from app.config import settings  # noqa: E402
from app.embedding import Embedder  # noqa: E402
from app.llm import MockLLM  # noqa: E402
from app.service import percentile  # noqa: E402
from scripts.replay import build_service, build_traffic, load_intents, replay, variant_to_intent  # noqa: E402

MODELS = ["all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2"]
# Calibration rule: lowest threshold whose false-hit rate on dev hard negatives stays <= this.
MAX_HARD_FALSE_HIT_RATE = 0.01
THRESHOLDS = [round(t, 2) for t in np.arange(0.50, 0.99, 0.01)]


def build_pairs(intents: list[dict], split: str) -> list[dict]:
    pool = [i for i in intents if i["split"] == split]
    pairs = []
    for intent in pool:
        for a, b in itertools.combinations(intent["variants"], 2):
            pairs.append({"a": a, "b": b, "same": True, "hard": False})
    for x, y in itertools.combinations(pool, 2):
        hard = x["group"] == y["group"]
        for a in x["variants"]:
            for b in y["variants"]:
                pairs.append({"a": a, "b": b, "same": False, "hard": hard})
    return pairs


def similarities(embedder: Embedder, pairs: list[dict]) -> np.ndarray:
    texts = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
    vectors = dict(zip(texts, embedder.encode(texts)))
    return np.array([float(vectors[p["a"]] @ vectors[p["b"]]) for p in pairs])


def pair_metrics(sims: np.ndarray, pairs: list[dict], guard_ok: np.ndarray, threshold: float, guard: bool) -> dict:
    same = np.array([p["same"] for p in pairs])
    hard = np.array([p["hard"] for p in pairs])
    predicted = sims >= threshold
    if guard:
        predicted &= guard_ok
    tp = int((predicted & same).sum())
    fp = int((predicted & ~same).sum())
    fp_hard = int((predicted & hard).sum())
    return {
        "threshold": threshold,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / int(same.sum()),
        "false_hit_rate": fp / int((~same).sum()),
        "hard_false_hit_rate": fp_hard / int(hard.sum()),
        "false_hits": fp,
        "hard_false_hits": fp_hard,
    }


def auc(sims: np.ndarray, pairs: list[dict]) -> float:
    """Probability that a random positive pair scores above a random negative pair."""
    same = np.array([p["same"] for p in pairs])
    ranks = sims.argsort().argsort() + 1
    n_pos, n_neg = int(same.sum()), int((~same).sum())
    return float((ranks[same].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def calibrate(sims, pairs, guard_ok, guard: bool) -> tuple[float, bool]:
    """Return (threshold, rule_met). When no threshold meets the rule, the highest one is returned."""
    for t in THRESHOLDS:
        if pair_metrics(sims, pairs, guard_ok, t, guard)["hard_false_hit_rate"] <= MAX_HARD_FALSE_HIT_RATE:
            return t, True
    return THRESHOLDS[-1], False


def unique_stream(intents: list[dict], split: str = "test", seed: int = 7) -> list[tuple[str, str]]:
    """Every question of the split exactly once, shuffled: no exact repeat is possible."""
    stream = [(i["id"], v) for i in intents if i["split"] == split for v in i["variants"]]
    import random

    random.Random(seed).shuffle(stream)
    return stream


def bench_search(dim: int, sizes=(1_000, 10_000, 100_000), queries: int = 200) -> list[dict]:
    rng = np.random.default_rng(0)
    out = []
    for n in sizes:
        cache = SemanticCache(ttl_seconds=10**9, max_entries=n + 1)
        vectors = rng.standard_normal((n, dim)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        key = ("bench", "mock", "x")
        start = time.perf_counter()
        for i in range(n):
            cache.add(key, f"q{i}", "a", vectors[i], now=0.0)
        insert_us = (time.perf_counter() - start) / n * 1e6
        timings = []
        for q in vectors[rng.integers(0, n, queries)]:
            t0 = time.perf_counter()
            cache.lookup_semantic(key, "probe", q, threshold=0.85, now=0.0)
            timings.append((time.perf_counter() - t0) * 1000)
        out.append({"entries": n, "lookup_p50_ms": percentile(timings, 50), "lookup_p95_ms": percentile(timings, 95), "insert_us": insert_us, "memory_mb": n * dim * 4 / 1e6})
    return out


def main() -> None:
    intents = load_intents()
    dev, test = build_pairs(intents, "dev"), build_pairs(intents, "test")
    guard_dev = np.array([compatible(p["a"], p["b"]) for p in dev])
    guard_test = np.array([compatible(p["a"], p["b"]) for p in test])

    lines = [
        f"# Semantic cache metrics ({date.today().isoformat()})",
        "",
        f"- Dataset: {len(intents)} intents, {sum(len(i['variants']) for i in intents)} questions (eval/intents.json)",
        f"- Dev pairs: {len(dev)} ({sum(p['same'] for p in dev)} same-answer, {sum(p['hard'] for p in dev)} hard negatives)",
        f"- Test pairs: {len(test)} ({sum(p['same'] for p in test)} same-answer, {sum(p['hard'] for p in test)} hard negatives)",
        f"- Calibration rule (dev): lowest threshold with at most {MAX_HARD_FALSE_HIT_RATE:.0%} false hits on hard negatives",
        "",
        "## Same answer or not? (held-out test pairs)",
        "",
        "| Model | Entity guard | Threshold (from dev) | Precision | Recall | False hits, all negatives | False hits, hard negatives | AUC |",
        "|---|---|---|---|---|---|---|---|",
    ]

    details, best, embedders = {}, None, {}
    for model in MODELS:
        embedder = Embedder(model)
        embedders[model] = embedder
        sims_dev, sims_test = similarities(embedder, dev), similarities(embedder, test)
        model_auc = auc(sims_test, test)
        for guard in (False, True):
            t, rule_met = calibrate(sims_dev, dev, guard_dev, guard)
            dev_m = pair_metrics(sims_dev, dev, guard_dev, t, guard)
            m = pair_metrics(sims_test, test, guard_test, t, guard)
            details[f"{model}|guard={guard}"] = {"dev": dev_m, "test": m, "auc": model_auc, "rule_met": rule_met}
            t_cell = f"{t:.2f}" if rule_met else f"none meets the rule (shown at {t:.2f})"
            lines.append(
                f"| `{model}` | {'on' if guard else 'off'} | {t_cell} | {100 * m['precision']:.1f}% | {100 * m['recall']:.1f}% "
                f"| {100 * m['false_hit_rate']:.2f}% ({m['false_hits']}) | {100 * m['hard_false_hit_rate']:.1f}% ({m['hard_false_hits']}) | {model_auc:.3f} |"
            )
            # Model choice uses dev recall only (no peeking at the test split).
            if guard and rule_met and (best is None or dev_m["recall"] > best[2]):
                best = (model, t, dev_m["recall"], sims_test)

    model, threshold, _dev_recall, sims_test = best
    lines += [
        "",
        f"Selected on dev: **`{model}`**, threshold **{threshold:.2f}**, entity guard on.",
        "",
        f"## Threshold sweep (test pairs, `{model}`, guard on)",
        "",
        "| Threshold | Precision | Recall | False hits, hard negatives |",
        "|---|---|---|---|",
    ]
    for t in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        m = pair_metrics(sims_test, test, guard_test, t, True)
        lines.append(f"| {t:.2f} | {100 * m['precision']:.1f}% | {100 * m['recall']:.1f}% | {100 * m['hard_false_hit_rate']:.1f}% |")

    # Replays with the mock LLM, at the service default threshold: guard on, guard off, exact match only.
    intent_of = variant_to_intent(intents)
    embedder = embedders[model]
    default_t = settings.SIMILARITY_THRESHOLD
    configs = [("semantic + guard", default_t, True), ("semantic, no guard", default_t, False), ("exact match only", 1.01, True)]
    replays = {}
    for traffic_name, traffic in [("zipf", build_traffic(intents)), ("unique", unique_stream(intents))]:
        replays[traffic_name] = {
            name: asyncio.run(replay(build_service(MockLLM(), embedder, t, guard), traffic, intent_of)) for name, t, guard in configs
        }
        title = (
            f"{len(traffic)} requests drawn with a Zipf distribution (repeats included)"
            if traffic_name == "zipf"
            else f"{len(traffic)} requests, every question phrased differently (no exact repeat possible)"
        )
        lines += [
            "",
            f"## Traffic replay: {title}",
            "",
            f"Held-out intents, mock LLM, threshold {default_t}.",
            "",
            "| Cache | Hit rate | Correct hits | False hits | Reachable requests served |",
            "|---|---|---|---|---|",
        ]
        for name, r in replays[traffic_name].items():
            lines.append(f"| {name} | {100 * r['hit_rate']:.1f}% | {r['correct_hits']} | {r['false_hits']} | {100 * r['recall_of_reachable']:.1f}% |")
        best_possible = replays[traffic_name]["semantic + guard"]["ideal_hit_rate"]
        lines.append(f"\nBest possible hit rate (intent already seen before): {100 * best_possible:.1f}%.")

    # Latency.
    texts = [v for i in intents for v in i["variants"]][:200]
    embed_ms = []
    for text in texts:
        t0 = time.perf_counter()
        embedder.encode_one(text)
        embed_ms.append((time.perf_counter() - t0) * 1000)
    dim = embedder.encode_one("dimension").shape[0]
    bench = bench_search(dim)
    lines += [
        "",
        "## Latency (CPU)",
        "",
        f"Embedding one question with `{model}`: p50 {percentile(embed_ms, 50):.1f} ms, p95 {percentile(embed_ms, 95):.1f} ms.",
        "",
        "| Entries in the partition | Search p50 | Search p95 | Insert (mean) | Vector memory |",
        "|---|---|---|---|---|",
    ]
    for b in bench:
        lines.append(f"| {b['entries']:,} | {b['lookup_p50_ms']:.2f} ms | {b['lookup_p95_ms']:.2f} ms | {b['insert_us']:.0f} µs | {b['memory_mb']:.0f} MB |")

    report = "\n".join(lines) + "\n"
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "metrics.md").write_text(report, encoding="utf-8")
    replay_rows = {f"{t}|{k}": v.pop("rows") for t, group in replays.items() for k, v in group.items()}
    (out / "eval_details.json").write_text(
        json.dumps({"pairs": details, "replays": replays, "replay_rows": replay_rows, "search_bench": bench}, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(report)
    print(f"Report written to {out / 'metrics.md'}")


if __name__ == "__main__":
    main()
