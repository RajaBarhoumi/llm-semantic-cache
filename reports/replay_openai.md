# Replay with `openai/gpt-oss-20b` (2026-11-11)

- LLM endpoint: https://api.groq.com/openai/v1 (costs priced at $0.15 / $0.6 per million input / output tokens)
- Traffic: 300 requests, held-out intents, Zipf distribution, seed 42
- Embedding model `all-MiniLM-L6-v2`, threshold 0.72, entity guard on

| Metric | Value |
|---|---|
| Hit rate | **84.7%** (254/300: 168 exact, 86 semantic) |
| Hit rate of an exact-match cache on the same traffic | 77.3% |
| Best possible hit rate (intent already seen) | 94.7% |
| False hits (wrong answer served) | **0** |
| Hits blocked by the entity guard | 3 |
| LLM calls | 46 instead of 300 |
| Latency of a hit, p50 / p95 | 0.1 ms / 40 ms |
| Latency of a miss, p50 / p95 | 2383 ms / 7052 ms |
| Cost with cache | $0.00489 |
| Cost avoided by hits | $0.02918 (85.7%) |
| **Measured cost without cache** | $0.03313 |
| **Measured saving** | **85.2%** |
| Measured LLM latency without cache, p50 / p95 | 507 ms / 1123 ms |

> **Note (added after the run):** this run used `LLM_MAX_REQUESTS_PER_MINUTE=25` to stay within
> Groq's free tier. In this version of the code, the "latency of a miss" row also counted the wait
> for a rate-limit slot (about 2.4 s per call), so it overstates the miss latency. The model's own
> latency is the "measured LLM latency without cache" row (507 ms p50), which never included the
> wait. The code now reports the wait separately (`queued_ms`); re-running the replay gives a
> corrected miss row. All other rows are unaffected.
