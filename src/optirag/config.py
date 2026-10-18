from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]

# USD per 1M tokens (input, output). Verify against your provider's current pricing.
PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
}


def _path(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default))
    return p if p.is_absolute() else ROOT / p


def _large() -> str:
    return os.getenv("OPTIRAG_LARGE_MODEL", "openai/gpt-oss-120b")


@dataclass(frozen=True, slots=True)
class Settings:
    provider: str = os.getenv("OPTIRAG_PROVIDER", "groq")
    small_model: str = os.getenv("OPTIRAG_SMALL_MODEL", "openai/gpt-oss-20b")
    large_model: str = _large()
    judge_model: str = os.getenv("OPTIRAG_JUDGE_MODEL") or _large()
    embedding_model: str = os.getenv("OPTIRAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    reasoning_effort: str = os.getenv("OPTIRAG_REASONING_EFFORT", "low")
    max_output_tokens: int = int(os.getenv("OPTIRAG_MAX_OUTPUT_TOKENS", "1024"))

    chunk_words: int = int(os.getenv("OPTIRAG_CHUNK_WORDS", "400"))
    chunk_overlap: int = int(os.getenv("OPTIRAG_CHUNK_OVERLAP", "60"))
    retrieval_k: int = int(os.getenv("OPTIRAG_RETRIEVAL_K", "3"))

    cache_threshold: float = float(os.getenv("OPTIRAG_CACHE_THRESHOLD", "0.90"))
    cache_ttl_hours: float = float(os.getenv("OPTIRAG_CACHE_TTL_HOURS", "168"))

    router_mode: str = os.getenv("OPTIRAG_ROUTER_MODE", "rules")
    router_threshold: float = float(os.getenv("OPTIRAG_ROUTER_THRESHOLD", "2.0"))
    escalate_below: float = float(os.getenv("OPTIRAG_ESCALATE_BELOW", "0.5"))

    corpus_dir: Path = _path("OPTIRAG_CORPUS_DIR", "data/corpus")
    eval_dir: Path = _path("OPTIRAG_EVAL_DIR", "data/eval")
    results_dir: Path = _path("OPTIRAG_RESULTS_DIR", "results")
    chroma_dir: Path = _path("OPTIRAG_CHROMA_DIR", "chroma_db")
    cache_dir: Path = _path("OPTIRAG_CACHE_DIR", "cache_db")
    db_path: Path = _path("OPTIRAG_DB_PATH", "logs/requests.db")


settings = Settings()


def token_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICES.get(model)
    return 0.0 if p is None else (input_tokens * p[0] + output_tokens * p[1]) / 1e6
