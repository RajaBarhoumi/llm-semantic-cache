"""Request bodies of the HTTP API."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    # Answers are only shared inside a namespace (e.g. one per tenant or product).
    namespace: str = Field("default", min_length=1, max_length=100)
    # Different instructions produce different answers: they get their own cache partition.
    system_prompt: str | None = Field(None, max_length=8000)
    # Skip the cache entirely (no read, no write), e.g. for time-sensitive questions.
    bypass_cache: bool = False


class LookupRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    namespace: str = "default"
    system_prompt: str | None = None
    k: int = Field(5, ge=1, le=20)
