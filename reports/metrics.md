# Semantic cache metrics (2026-11-11)

- Dataset: 33 intents, 166 questions (eval/intents.json)
- Dev pairs: 3655 (175 same-answer, 515 hard negatives)
- Test pairs: 3160 (160 same-answer, 375 hard negatives)
- Calibration rule (dev): lowest threshold with at most 1% false hits on hard negatives

## Same answer or not? (held-out test pairs)

| Model | Entity guard | Threshold (from dev) | Precision | Recall | False hits, all negatives | False hits, hard negatives | AUC |
|---|---|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | off | 0.92 | 28.6% | 1.2% | 0.17% (5) | 1.3% (5) | 0.972 |
| `all-MiniLM-L6-v2` | on | 0.72 | 100.0% | 33.8% | 0.00% (0) | 0.0% (0) | 0.972 |
| `paraphrase-multilingual-MiniLM-L12-v2` | off | 0.92 | 16.7% | 0.6% | 0.17% (5) | 1.3% (5) | 0.952 |
| `paraphrase-multilingual-MiniLM-L12-v2` | on | 0.74 | 94.7% | 22.5% | 0.07% (2) | 0.5% (2) | 0.952 |

Selected on dev: **`all-MiniLM-L6-v2`**, threshold **0.72**, entity guard on.

## Threshold sweep (test pairs, `all-MiniLM-L6-v2`, guard on)

| Threshold | Precision | Recall | False hits, hard negatives |
|---|---|---|---|
| 0.70 | 96.8% | 37.5% | 0.5% |
| 0.75 | 100.0% | 26.2% | 0.0% |
| 0.80 | 100.0% | 16.9% | 0.0% |
| 0.85 | 100.0% | 8.8% | 0.0% |
| 0.90 | 100.0% | 1.2% | 0.0% |
| 0.95 | 100.0% | 0.0% | 0.0% |

## Traffic replay: 300 requests drawn with a Zipf distribution (repeats included)

Held-out intents, mock LLM, threshold 0.72.

| Cache | Hit rate | Correct hits | False hits | Reachable requests served |
|---|---|---|---|---|
| semantic + guard | 84.7% | 254 | 0 | 89.4% |
| semantic, no guard | 85.7% | 213 | 44 | 75.0% |
| exact match only | 77.3% | 232 | 0 | 81.7% |

Best possible hit rate (intent already seen before): 94.7%.

## Traffic replay: 80 requests, every question phrased differently (no exact repeat possible)

Held-out intents, mock LLM, threshold 0.72.

| Cache | Hit rate | Correct hits | False hits | Reachable requests served |
|---|---|---|---|---|
| semantic + guard | 37.5% | 30 | 0 | 46.9% |
| semantic, no guard | 42.5% | 28 | 6 | 43.8% |
| exact match only | 0.0% | 0 | 0 | 0.0% |

Best possible hit rate (intent already seen before): 80.0%.

## Latency (CPU)

Embedding one question with `all-MiniLM-L6-v2`: p50 21.4 ms, p95 24.9 ms.

| Entries in the partition | Search p50 | Search p95 | Insert (mean) | Vector memory |
|---|---|---|---|---|
| 1,000 | 0.08 ms | 0.11 ms | 14 µs | 2 MB |
| 10,000 | 1.11 ms | 1.32 ms | 42 µs | 15 MB |
| 100,000 | 8.50 ms | 9.58 ms | 17 µs | 154 MB |
