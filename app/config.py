"""Service configuration (environment variables / .env file)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Upstream LLM: "openai" (real model) or "mock" (deterministic fake, no API key needed).
    LLM_PROVIDER: str = "mock"
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    # Any OpenAI-compatible endpoint (Groq, Gemini, OpenRouter, Ollama...); empty = api.openai.com.
    OPENAI_BASE_URL: str | None = None
    OPENAI_MAX_RETRIES: int = 5
    # Client-side pacing for free tiers; 0 = no limit.
    LLM_MAX_REQUESTS_PER_MINUTE: int = 0
    LLM_TEMPERATURE: float = 0.0
    MAX_OUTPUT_TOKENS: int = 300
    # USD per million tokens, for cost reporting only. Check current pricing.
    PRICE_INPUT_PER_MTOK: float = 0.15
    PRICE_OUTPUT_PER_MTOK: float = 0.60
    # Artificial delay of the mock LLM, to make offline demos realistic.
    MOCK_LATENCY_MS: int = 0
    DEFAULT_SYSTEM_PROMPT: str = (
        "You are a customer support assistant for an online shop. Answer in at most three sentences."
    )

    # Semantic matching (defaults chosen with scripts/evaluate.py, see README).
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    # Calibrated on the dev intents: lowest value with <= 1% false hits on hard negatives.
    SIMILARITY_THRESHOLD: float = 0.72
    # Refuse a semantic hit when the questions mention different numbers, ids, emails or names.
    ENTITY_GUARD_ENABLED: bool = True

    # Cache lifecycle.
    CACHE_TTL_SECONDS: int = 86_400
    CACHE_MAX_ENTRIES: int = 10_000


settings = Settings()
