from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from .ingest import ingest
from .llm import LLMError
from .pipeline import Pipeline
from .schemas import Answer, AskRequest, IngestRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ingest()
    app.state.pipeline = Pipeline()
    yield


app = FastAPI(title="OptiRAG", version="0.1.0", lifespan=lifespan)


def _pipe(request: Request) -> Pipeline:
    return request.app.state.pipeline


def _guard(fn, *args) -> Answer:
    try:
        return fn(*args)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except LLMError as e:
        raise HTTPException(502, str(e)) from e


@app.post("/ask", response_model=Answer)
def ask(body: AskRequest, request: Request) -> Answer:
    return _guard(_pipe(request).answer, body.question, body.tenant)


@app.post("/ask/baseline", response_model=Answer)
def ask_baseline(body: AskRequest, request: Request) -> Answer:
    return _guard(_pipe(request).baseline_answer, body.question)


@app.post("/ingest")
def reingest(body: IngestRequest, request: Request) -> dict:
    stats = ingest(force=body.force)
    pipe = _pipe(request)
    pipe.refresh()
    return {
        "version": stats.version,
        "docs": stats.docs,
        "chunks": stats.chunks,
        "skipped": stats.skipped,
        "cache_entries": pipe.cache.count(),
    }


@app.get("/health")
def health(request: Request) -> dict:
    pipe = _pipe(request)
    return {"status": "ok", "corpus_version": pipe.version, "cache_entries": pipe.cache.count()}
