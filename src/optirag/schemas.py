from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Source(BaseModel):
    doc: str
    page: int
    score: float
    text: str = ""


class Answer(BaseModel):
    request_id: int | None = None
    question: str
    answer: str
    sources: list[Source] = []
    route: Literal["cache", "small", "large", "escalated"]
    model: str
    cache_hit: bool = False
    escalated: bool = False
    escalation_reason: str | None = None
    similarity: float = 0.0
    top_score: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    tenant: str = "public"


class IngestRequest(BaseModel):
    force: bool = False


class RequestLog(BaseModel):
    id: int | None = None
    ts: float
    run_id: str
    mode: str
    question: str
    route: str
    model: str
    cache_hit: bool = False
    escalated: bool = False
    similarity: float = 0.0
    top_score: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    quality: float | None = None
    error: str | None = None
    corpus_version: str = ""
