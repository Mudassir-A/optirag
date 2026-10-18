from __future__ import annotations

import json

import numpy as np
import pandas as pd

from eval.common import read_jsonl
from optirag.cache import normalize
from optirag.config import settings
from optirag.embeddings import embed
from optirag.ingest import corpus_version
from optirag.retriever import Retriever
from optirag.router import classify_llm, classify_rules, complexity


def accuracy(pred: list[str], gold: list[str]) -> float:
    return float(np.mean([p == g for p, g in zip(pred, gold)]))


def main() -> None:
    rows = read_jsonl(settings.eval_dir / "router_labels.jsonl")
    gold = [r["label"] for r in rows]
    ret = Retriever(corpus_version())
    vecs = embed([normalize(r["question"]) for r in rows])
    scores = [[s.score for s in ret.search(v, settings.retrieval_k)] for v in vecs]

    rules = [classify_rules(r["question"], s) for r, s in zip(rows, scores)]
    sweep = {
        float(t): accuracy(
            ["complex" if complexity(r["question"], s) >= t else "simple" for r, s in zip(rows, scores)], gold
        )
        for t in np.arange(0.5, 4.01, 0.5)
    }

    llm_pred, cost, lat = [], 0.0, []
    for r in rows:
        tier, res = classify_llm(r["question"], settings.small_model)
        llm_pred.append(tier)
        cost += res.cost
        lat.append(res.latency_ms)

    result = {
        "n": len(rows),
        "rules": {"accuracy": accuracy(rules, gold), "cost_usd": 0.0, "threshold": settings.router_threshold},
        "rules_threshold_sweep": sweep,
        "llm": {
            "accuracy": accuracy(llm_pred, gold),
            "cost_usd": cost,
            "cost_per_query_usd": cost / len(rows),
            "mean_latency_ms": float(np.mean(lat)),
        },
    }
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    (settings.results_dir / "router_eval.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(
        {"question": [r["question"] for r in rows], "gold": gold, "rules": rules, "llm": llm_pred}
    ).to_csv(settings.results_dir / "router_eval.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
