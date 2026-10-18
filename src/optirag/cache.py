from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

from .ingest import COSINE, chroma_client
from .rag import NO_ANSWER
from .schemas import Source

_PUNCT = re.compile(r"[^\w\s]")


def normalize(question: str) -> str:
    return " ".join(_PUNCT.sub(" ", question.lower()).split())


def is_cacheable(answer: str, sources: list[Source]) -> bool:
    return bool(answer.strip()) and bool(sources) and not NO_ANSWER.search(answer)


class SemanticCache:
    def __init__(self, version: str, threshold: float, ttl_hours: float, path: Path) -> None:
        self.version, self.threshold, self.ttl = version, threshold, ttl_hours * 3600
        self.col = chroma_client(str(path)).get_or_create_collection("semantic_cache", metadata=COSINE)

    def set_version(self, version: str) -> None:
        self.version = version

    def count(self) -> int:
        return self.col.count()

    def lookup(self, vec: np.ndarray, tenant: str = "public") -> tuple[dict | None, float]:
        if not self.col.count():
            return None, 0.0
        r = self.col.query(
            query_embeddings=[vec.tolist()],
            n_results=1,
            where={
                "$and": [
                    {"corpus_version": self.version},
                    {"tenant": tenant},
                    {"created_at": {"$gte": time.time() - self.ttl}},
                ]
            },
            include=["documents", "distances"],
        )
        if not r["ids"][0]:
            return None, 0.0
        score = 1.0 - r["distances"][0][0]
        return (json.loads(r["documents"][0][0]) if score >= self.threshold else None), score

    def store(
        self, question: str, vec: np.ndarray, answer: str, sources: list[Source], tenant: str = "public"
    ) -> bool:
        if not is_cacheable(answer, sources):
            return False
        key = hashlib.sha1(f"{self.version}|{tenant}|{normalize(question)}".encode()).hexdigest()
        self.col.upsert(
            ids=[key],
            embeddings=[vec.tolist()],
            documents=[json.dumps({"answer": answer, "sources": [s.model_dump() for s in sources]})],
            metadatas=[
                {
                    "corpus_version": self.version,
                    "tenant": tenant,
                    "created_at": time.time(),
                    "question": question,
                }
            ],
        )
        return True

    def invalidate_stale(self) -> int:
        before = self.col.count()
        if before:
            self.col.delete(
                where={
                    "$or": [
                        {"corpus_version": {"$ne": self.version}},
                        {"created_at": {"$lt": time.time() - self.ttl}},
                    ]
                }
            )
        return before - self.col.count()

    def clear(self) -> None:
        ids = self.col.get(include=[])["ids"]
        if ids:
            self.col.delete(ids=ids)
