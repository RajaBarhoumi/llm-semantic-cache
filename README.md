# LLM Semantic Cache

A FastAPI proxy in front of an LLM that **stops paying twice for the same question**. When a
question means the same thing as one already answered ("How do I reset my password?" / "I forgot
my password"), it returns the stored answer instead of calling the model.

The hard part is not finding similar questions, it is **never serving a wrong answer**: "invoice
for order 1001" and "invoice for order 1002" are 97.5% similar for an embedding model, and must
never share an answer. This project measures both sides.

## Table of contents

- [How it works](#how-it-works)
- [Installation](#installation)
- [Trying it](#trying-it)
- [API](#api)
- [Metrics](#metrics)
- [Limitations](#limitations)
- [Configuration](#configuration)
- [Project structure](#project-structure)

## How it works

```
question
   │
   ├─ 1. exact match (normalized text) ───────────────────────────► hit, no embedding needed
   │
   ├─ 2. embed the question, find the most similar cached question
   │        similarity ≥ threshold AND entity guard OK ────────────► hit
   │        similarity ≥ threshold BUT different numbers/names ───► miss (guard_rejected)
   │
   └─ 3. call the LLM, store question + answer + cost ────────────► miss
```

- **Partitions.** An answer is only reused for the same `namespace` (tenant, product), the same
  model and the same system prompt. Changing any of them gives a separate cache.
- **Entity guard.** Numbers, ids, emails and capitalized names are "anchors": a semantic hit is
  allowed only if each question's anchors also appear in the other. "Order 1001" never matches
  "order 1002"; "Canada" never matches "France"; "PayPal" still matches "paypal".
- **Single flight.** Identical questions arriving at the same time trigger one LLM call; the
  others wait for its answer.
- **Lifecycle.** Entries expire after `CACHE_TTL_SECONDS`; beyond `CACHE_MAX_ENTRIES`, the least
  recently used entry is evicted. `bypass_cache: true` skips the cache for time-sensitive
  questions.
- **Storage.** In memory, no database: brute-force cosine search over a growable NumPy buffer
  (one matrix product), fast up to about 100k entries per partition (see
  [Metrics](#metrics)).

## Installation

Requirements: Python 3.11 or later.

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Open `.env` and set `OPENAI_API_KEY`. Without a key, set `LLM_PROVIDER=mock`: answers are fake
but the whole cache works.

```bash
uvicorn app.main:app --reload
```

The embedding model (`all-MiniLM-L6-v2`, about 90 MB) is downloaded on first launch and loaded at
startup, so the first request is as fast as the others.

## Trying it

[requests.http](requests.http) walks through every case: a miss, a semantic hit, an exact hit,
a known false hit, a guard rejection (order 1001 vs 1002), namespace isolation, bypass, the
lookup diagnostic and the stats. Open it with the VS Code **REST Client** extension.

With curl:

```bash
curl -s -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question": "How do I reset my password?"}'
```

```bash
curl -s -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question": "I forgot my password"}'
```

The second call returns the first answer, without calling the model (mock LLM shown):

```json
{
  "answer": "[mock answer] How do I reset my password?",
  "cache": {"status": "hit", "kind": "semantic", "similarity": 0.7798,
            "matched_question": "How do I reset my password?", "entry_id": 1,
            "guard_rejected_question": null, "guard_rejected_similarity": null},
  "model": "mock",
  "usage": {"input_tokens": 0, "output_tokens": 0},
  "cost_usd": 0.0,
  "saved_cost_usd": 3.555e-05,
  "latency_ms": 22.58
}
```

Tests (fake embedder, no model download, no key):

```bash
python -m pytest -q
```

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/ask` | `question`, optional `namespace`, `system_prompt`, `bypass_cache`. Returns the answer and the cache decision. |
| POST | `/cache/lookup` | Diagnostic: nearest cached questions, their similarity, threshold and guard result. Never calls the LLM. |
| GET | `/stats` | Requests, exact / semantic / coalesced hits, hit rate, guard rejections, cost spent and saved, hit vs miss latency |
| GET | `/cache/entries` | Cached questions, most used first |
| DELETE | `/cache/entries/{id}` | Remove one entry (e.g. a wrong answer) |
| DELETE | `/cache` | Empty the cache (`?namespace=` to empty one namespace) |
| GET | `/health` | LLM, embedding model, threshold, guard status |

`cache.status` is `hit`, `miss` or `bypass`; `cache.kind` is `exact`, `semantic` or `coalesced`.

## Metrics

### Dataset

[eval/intents.json](eval/intents.json): 33 customer-support intents of a fictitious online shop,
166 questions. Each intent lists paraphrases that deserve the same answer. Intents are grouped
into families of look-alikes (cancel vs track an order, reset vs change a password, order 1001 vs
1002, Canada vs France): pairs from the same family are the **hard negatives**.

Whole families are split into **dev** (17 intents, to choose the threshold and the model) and
**test** (16 intents, never seen during calibration). All numbers below are on test unless
stated. Measured on November 11, 2026, CPU only, with `python scripts/evaluate.py`; full report
in [reports/metrics.md](reports/metrics.md).

### 1. Same answer or not? (3,160 test pairs)

Threshold chosen on dev: the lowest value with at most 1% false hits on hard negatives.

| Model | Entity guard | Threshold | Precision | Recall | False hits on hard negatives |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | off | 0.92 | 28.6% | 1.2% | 1.3% |
| **`all-MiniLM-L6-v2`** | **on** | **0.72** | **100%** | **33.8%** | **0%** |
| `paraphrase-multilingual-MiniLM-L12-v2` | off | 0.92 | 16.7% | 0.6% | 1.3% |
| `paraphrase-multilingual-MiniLM-L12-v2` | on | 0.74 | 94.7% | 22.5% | 0.5% |

**Without the guard, there is no useful safe threshold.** Look-alike questions with different
numbers or names score 0.95 or more, so the threshold has to go up to 0.92. Recall then collapses
to about 1%, and false hits remain. The guard removes those cases, which lets the threshold drop
to 0.72.

Recall is per pair: not every paraphrase is close to every other one. In a cache, a question
only needs to be close to **one** stored phrasing, which is why the hit rates below are much
higher.

### 2. Traffic replay (mock LLM, threshold 0.72)

**Realistic mix:** 300 requests on the test intents, a few popular questions and a long tail
(Zipf distribution), exact repeats included.

| Cache | Hit rate | False hits (wrong answer served) |
|---|---|---|
| Exact match only | 77.3% | 0 |
| **Semantic + guard** | **84.7%** | **0** |
| Semantic, no guard | 85.7% | **44** |

**Worst case for an exact cache:** 80 requests, every question phrased differently.

| Cache | Hit rate | False hits |
|---|---|---|
| Exact match only | 0% | 0 |
| **Semantic + guard** | **37.5%** | **0** |
| Semantic, no guard | 42.5% | 6 |

The best possible hit rate (the intent was already seen before) is 94.7% and 80.0%: the cache
serves 89.4% and 46.9% of what it could have. The guard gives up 1 to 5 points of hit rate to go
from 44 wrong answers to 0.

### 3. Latency (CPU)

| Step | p50 | p95 |
|---|---|---|
| Exact hit | 0.1 ms | |
| Embedding a question (`all-MiniLM-L6-v2`) | 21 ms | 25 ms |
| Search, 1,000 entries | 0.08 ms | 0.11 ms |
| Search, 10,000 entries | 1.1 ms | 1.3 ms |
| Search, 100,000 entries | 8.5 ms | 9.6 ms |

A semantic hit costs about 25 ms, almost entirely the embedding. 100,000 cached questions use
154 MB of vectors.

### 4. Real model: cost and latency saved

The same 300 requests, replayed through the cache with a real model, then replayed again without
cache to measure the real baseline. Model `openai/gpt-oss-20b` on Groq's free tier, so the actual
bill was $0; costs are the token counts **priced at gpt-4o-mini rates** ($0.15 / $0.60 per
million input / output tokens), i.e. what this traffic would cost on that model. Full report:
[reports/replay_openai.md](reports/replay_openai.md).

| Metric | Without cache | With cache |
|---|---|---|
| LLM calls | 300 | **46** |
| Cost (priced at gpt-4o-mini rates) | $0.0331 | **$0.0049** |
| **Measured saving** | | **85.2%** |
| Wrong answers served | | **0** |
| Latency of an answer, p50 / p95 | 507 ms / 1,123 ms (model call) | 0.1 ms / 40 ms on a hit |

- **Hit rate 84.7%** (168 exact, 86 semantic), the same as the offline replay: the decision to
  hit or miss does not depend on the model, only on the questions.
- **A hit is about 5,000 times faster than a model call at the median** (0.1 ms vs 507 ms: most
  hits on this traffic are exact matches, which skip the embedding), and about 28 times faster at
  p95 (40 ms vs 1,123 ms), where semantic hits pay about 25 ms for the embedding.
- The run paced calls at 25 per minute for the free tier. The report's "latency of a miss" row
  counted that wait by mistake, so it is not used here; the code now reports it separately
  (`queued_ms`), see the note in the report.
- The saving depends on the traffic: this one has a few very frequent questions. On traffic where
  every question is phrased differently, the offline replay serves 37.5% from the cache instead
  of 84.7%.

### Running the evaluations

Offline, no key needed (calibration, replays with the mock LLM, latency):

```bash
python scripts/evaluate.py
```

With a real model: replays the same 300 requests through the cache, then again without cache,
to measure the real cost and latency saved (about 346 calls). Works with OpenAI or any
OpenAI-compatible provider, including free tiers such as Groq (presets in `.env.example`).

```bash
python scripts/replay.py --llm openai --with-baseline
```

The report is written to `reports/replay_openai.md`.

## Limitations

- **Same words, different action.** "How do I change my password?" scores 0.87 against "How do I
  reset my password?" and is served the cached answer. Similarity cannot separate verbs that
  differ in meaning but share context; the guard only covers numbers, ids, emails and names. For
  such flows, use `bypass_cache`, a separate namespace, or a stricter threshold (0.90 in the
  sweep, at the cost of most hits).
- **Calibrated on 166 questions** written by the same author as the intents: the numbers are
  indicative, roughly ±10 points. Recalibrate on your own traffic with `scripts/evaluate.py`.
- **Personal or time-sensitive answers** ("where is my order?") must not be cached as is: the
  answer depends on the user and the moment. Use `bypass_cache` or a short TTL.
- **In memory**: the cache is lost on restart and is not shared between processes.

## Configuration

Environment variables or `.env` (see [.env.example](.env.example)):

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `openai` or `mock` |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | — / `gpt-4o-mini` | Upstream model |
| `OPENAI_BASE_URL` | api.openai.com | Any OpenAI-compatible API: Groq, Gemini, Ollama… (presets in `.env.example`) |
| `LLM_MAX_REQUESTS_PER_MINUTE` | 0 (no limit) | Client-side pacing for free tiers |
| `MAX_OUTPUT_TOKENS` | 300 | Answer length cap |
| `PRICE_INPUT_PER_MTOK` / `PRICE_OUTPUT_PER_MTOK` | 0.15 / 0.60 | USD per million tokens, for cost reporting |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Local embedding model |
| `SIMILARITY_THRESHOLD` | 0.72 | Minimum cosine similarity for a semantic hit |
| `ENTITY_GUARD_ENABLED` | true | Block hits between questions with different numbers or names |
| `CACHE_TTL_SECONDS` | 86400 | Entry lifetime |
| `CACHE_MAX_ENTRIES` | 10000 | LRU eviction beyond this size |

## Project structure

```
├── app/
│   ├── main.py             # app factory, model warm-up at startup
│   ├── config.py           # settings
│   ├── service.py          # /ask flow, single flight, stats
│   ├── llm.py              # OpenAI and mock clients, cost
│   ├── embedding.py        # local sentence embedder
│   ├── cache/
│   │   ├── store.py        # partitions, vector buffer, TTL, LRU
│   │   └── guards.py       # normalization, entity guard
│   └── api/                # routes and request schemas
├── eval/intents.json       # labeled intents with dev/test split
├── scripts/
│   ├── evaluate.py         # offline metrics
│   └── replay.py           # traffic replay, real cost with --llm openai
├── reports/                # latest metrics
├── tests/                  # pytest suite
└── requests.http           # ready-to-run requests
```
