from __future__ import annotations

import logging
import time
from collections.abc import Callable

import numpy as np

from . import rag
from .cache import SemanticCache, normalize
from .config import Settings, settings
from .embeddings import embed
from .ingest import corpus_version
from .llm import LLMResult, complete
from .request_log import RequestStore
from .retriever import Retriever
from .router import classify_llm, classify_rules, escalation_reason
from .schemas import Answer, RequestLog, Source

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        cfg: Settings = settings,
        *,
        version: str | None = None,
        retriever: Retriever | None = None,
        cache: SemanticCache | None = None,
        store: RequestStore | None = None,
        embed_fn: Callable[[list[str]], np.ndarray] = embed,
        llm: Callable[..., LLMResult] = complete,
        run_id: str = "live",
    ) -> None:
        self.cfg, self.embed_fn, self.llm, self.run_id = cfg, embed_fn, llm, run_id
        self.version = version or corpus_version(cfg)
        self.retriever = retriever or Retriever(self.version, cfg)
        self.cache = cache or SemanticCache(
            self.version, cfg.cache_threshold, cfg.cache_ttl_hours, cfg.cache_dir
        )
        self.store = store or RequestStore(cfg.db_path)
        self.cache.invalidate_stale()

    def refresh(self) -> None:
        self.version = corpus_version(self.cfg)
        self.retriever = Retriever(self.version, self.cfg)
        self.cache.set_version(self.version)
        self.cache.invalidate_stale()

    def answer(self, question: str, tenant: str = "public") -> Answer:
        return self._run(question, "optimized", tenant)

    def baseline_answer(self, question: str) -> Answer:
        return self._run(question, "baseline", "public")

    def _run(self, question: str, mode: str, tenant: str) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        t0 = time.perf_counter()
        try:
            ans = self._execute(question, mode, tenant, t0)
        except Exception as e:
            self._log(question, mode, error=str(e)[:300], latency_ms=(time.perf_counter() - t0) * 1000)
            raise
        ans.request_id = self._log(question, mode, ans=ans)
        return ans

    def _execute(self, question: str, mode: str, tenant: str, t0: float) -> Answer:
        cfg, opt = self.cfg, mode == "optimized"
        vec = self.embed_fn([normalize(question)])[0]
        sim = 0.0
        if opt:
            hit = None
            try:
                hit, sim = self.cache.lookup(vec, tenant)
            except Exception:
                logger.exception("cache lookup failed; treating as miss")
            if hit:
                sources = [Source(**s) for s in hit["sources"]]
                return Answer(
                    question=question,
                    answer=hit["answer"],
                    sources=sources,
                    route="cache",
                    model="cache",
                    cache_hit=True,
                    similarity=sim,
                    top_score=max((s.score for s in sources), default=0.0),
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

        sources = self.retriever.search(vec, cfg.retrieval_k)
        scores = [s.score for s in sources]
        usage: list[LLMResult] = []
        route = "large"
        if opt:
            if cfg.router_mode == "llm":
                tier, r = classify_llm(question, cfg.small_model, self.llm)
                usage.append(r)
            else:
                tier = classify_rules(question, scores, cfg.router_threshold)
            route = "small" if tier == "simple" else "large"
        model = cfg.small_model if route == "small" else cfg.large_model

        res = rag.generate(question, sources, model, self.llm)
        usage.append(res)
        reason = None
        if route == "small":
            reason = escalation_reason(res.text, sources, cfg)
            if reason:
                model, route = cfg.large_model, "escalated"
                res = rag.generate(question, sources, model, self.llm)
                usage.append(res)
        if opt:
            try:
                self.cache.store(question, vec, res.text, sources, tenant)
            except Exception:
                logger.exception("cache store failed")

        return Answer(
            question=question,
            answer=res.text,
            sources=sources,
            route=route,
            model=model,
            escalated=route == "escalated",
            escalation_reason=reason,
            similarity=sim,
            top_score=scores[0] if scores else 0.0,
            input_tokens=sum(u.input_tokens for u in usage),
            output_tokens=sum(u.output_tokens for u in usage),
            cost_usd=sum(u.cost for u in usage),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def _log(
        self,
        question: str,
        mode: str,
        ans: Answer | None = None,
        error: str | None = None,
        latency_ms: float = 0.0,
    ) -> int | None:
        base = dict(
            ts=time.time(),
            run_id=self.run_id,
            mode=mode,
            question=question,
            corpus_version=self.version,
        )
        if ans is not None:
            rec = RequestLog(
                **base,
                **ans.model_dump(
                    include={
                        "route", "model", "cache_hit", "escalated", "similarity", "top_score",
                        "input_tokens", "output_tokens", "cost_usd", "latency_ms",
                    }
                ),
            )
        else:
            rec = RequestLog(**base, route="error", model="", error=error, latency_ms=latency_ms)
        try:
            return self.store.add(rec)
        except Exception:
            logger.exception("request log write failed")
            return None
