from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from eval.judge import judge
from optirag.pipeline import Pipeline
from optirag.schemas import Source

logger = logging.getLogger(__name__)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `make gen` or create it")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def retrieval_hit(sources: list[Source], doc: str | None, page: int | None) -> bool:
    return bool(doc) and any(s.doc == doc and (page is None or s.page == page) for s in sources)


def replay(pipe: Pipeline, rows: list[dict], mode: str) -> pd.DataFrame:
    fn = pipe.answer if mode == "optimized" else pipe.baseline_answer
    out = []
    for r in tqdm(rows, desc=pipe.run_id):
        rec = {
            **r, "answer": "", "route": "error", "model": "", "cache_hit": False, "escalated": False,
            "cost_usd": 0.0, "latency_ms": np.nan, "input_tokens": 0, "output_tokens": 0,
            "hit": False, "quality": 0.0, "error": "",
        }
        try:
            a = fn(r["question"])
            rec.update(
                answer=a.answer, route=a.route, model=a.model, cache_hit=a.cache_hit,
                escalated=a.escalated, cost_usd=a.cost_usd, latency_ms=a.latency_ms,
                input_tokens=a.input_tokens, output_tokens=a.output_tokens,
                hit=retrieval_hit(a.sources, r.get("doc"), r.get("page")),
            )
            rec["quality"] = judge(r["question"], r["reference"], a.answer)
            pipe.store.set_quality(a.request_id, rec["quality"])
        except Exception as e:
            logger.exception("request failed")
            rec["error"] = str(e)[:200]
        out.append(rec)
    return pd.DataFrame(out)


def summarize(df: pd.DataFrame) -> dict:
    return {
        "requests": int(len(df)),
        "errors": int((df.error != "").sum()),
        "cost_total_usd": float(df.cost_usd.sum()),
        "cost_per_request_usd": float(df.cost_usd.mean()),
        "cache_hit_rate": float(df.cache_hit.mean()),
        "escalation_rate": float(df.escalated.mean()),
        "p50_ms": float(df.latency_ms.quantile(0.5)),
        "p95_ms": float(df.latency_ms.quantile(0.95)),
        "quality": float(df.quality.mean()),
        "retrieval_hit_rate": float(df.hit.mean()),
        "routes": {k: int(v) for k, v in df.route.value_counts().items()},
    }
