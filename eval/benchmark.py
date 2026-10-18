from __future__ import annotations

import argparse
import json

import pandas as pd

from eval.common import read_jsonl, replay, summarize
from optirag.config import settings
from optirag.pipeline import Pipeline


def build_traffic(limit: int | None) -> list[dict]:
    evals = read_jsonl(settings.eval_dir / "eval_set.jsonl")[:limit]
    para = {p["question"]: p["paraphrase"] for p in read_jsonl(settings.eval_dir / "paraphrases.jsonl")}
    repeats = [{**r, "question": para[r["question"]], "phase": "repeat"} for r in evals if r["question"] in para]
    return [{**r, "phase": "unique"} for r in evals] + repeats


def pct(base: float, new: float) -> float:
    return (1 - new / base) * 100 if base else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    traffic = build_traffic(args.limit)
    base_pipe = Pipeline(run_id="bench_baseline")
    opt_pipe = Pipeline(run_id="bench_optimized")
    for p in (base_pipe, opt_pipe):
        p.store.delete_run(p.run_id)
    opt_pipe.cache.clear()

    base = replay(base_pipe, traffic, "baseline")
    opt = replay(opt_pipe, traffic, "optimized")

    b, o = summarize(base), summarize(opt)
    bu, ou = summarize(base[base.phase == "unique"]), summarize(opt[opt.phase == "unique"])
    result = {
        "config": {
            "small_model": settings.small_model,
            "large_model": settings.large_model,
            "cache_threshold": settings.cache_threshold,
            "router_mode": settings.router_mode,
            "router_threshold": settings.router_threshold,
            "escalate_below": settings.escalate_below,
            "retrieval_k": settings.retrieval_k,
            "requests": len(traffic),
            "unique": int((base.phase == "unique").sum()),
            "repeat": int((base.phase == "repeat").sum()),
        },
        "baseline": b,
        "optimized": o,
        "unique_only": {"baseline": bu, "optimized": ou},
        "delta": {
            "cost_reduction_pct": pct(b["cost_total_usd"], o["cost_total_usd"]),
            "cost_reduction_unique_only_pct": pct(bu["cost_total_usd"], ou["cost_total_usd"]),
            "p50_change_pct": -pct(b["p50_ms"], o["p50_ms"]),
            "p95_change_pct": -pct(b["p95_ms"], o["p95_ms"]),
            "quality_delta": o["quality"] - b["quality"],
        },
    }

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    (settings.results_dir / "benchmark.json").write_text(json.dumps(result, indent=2))
    pd.concat([base.assign(system="baseline"), opt.assign(system="optimized")]).to_csv(
        settings.results_dir / "benchmark_requests.csv", index=False
    )

    d = result["delta"]
    md = f"""| Metric | Baseline | Optimized |
|---|---|---|
| Requests | {b['requests']} | {o['requests']} |
| Total cost (USD) | {b['cost_total_usd']:.5f} | {o['cost_total_usd']:.5f} |
| Cost / request (USD) | {b['cost_per_request_usd']:.6f} | {o['cost_per_request_usd']:.6f} |
| Cache hit rate | {b['cache_hit_rate']:.1%} | {o['cache_hit_rate']:.1%} |
| Escalation rate | {b['escalation_rate']:.1%} | {o['escalation_rate']:.1%} |
| p50 latency (ms) | {b['p50_ms']:.0f} | {o['p50_ms']:.0f} |
| p95 latency (ms) | {b['p95_ms']:.0f} | {o['p95_ms']:.0f} |
| Answer quality (0-1) | {b['quality']:.3f} | {o['quality']:.3f} |
| Retrieval hit rate | {b['retrieval_hit_rate']:.1%} | {o['retrieval_hit_rate']:.1%} |

Cost reduction: **{d['cost_reduction_pct']:.1f}%** (unique queries only: {d['cost_reduction_unique_only_pct']:.1f}%) |
p95 change: **{d['p95_change_pct']:+.1f}%** | quality delta: **{d['quality_delta']:+.3f}**
"""
    (settings.results_dir / "benchmark.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
