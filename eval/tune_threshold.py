from __future__ import annotations

import numpy as np
import pandas as pd

from eval.common import read_jsonl
from optirag.cache import normalize
from optirag.config import settings
from optirag.embeddings import embed


def main() -> None:
    pairs = read_jsonl(settings.eval_dir / "cache_pairs.jsonl")
    a = embed([normalize(p["a"]) for p in pairs])
    b = embed([normalize(p["b"]) for p in pairs])
    sims = np.einsum("ij,ij->i", a, b)
    para = np.array([p["type"] == "paraphrase" for p in pairs])

    rows = []
    for t in np.round(np.arange(0.85, 0.9701, 0.01), 2):
        hit = sims >= t
        rows.append(
            {
                "threshold": float(t),
                "paraphrase_hit_rate": float(hit[para].mean()),
                "false_hits": int(hit[~para].sum()),
                "false_hit_rate": float(hit[~para].mean()),
            }
        )
    df = pd.DataFrame(rows)
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.results_dir / "threshold_sweep.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nparaphrase sim mean={sims[para].mean():.3f} min={sims[para].min():.3f}")
    print(f"near-miss  sim mean={sims[~para].mean():.3f} max={sims[~para].max():.3f}")

    safe = df[df.false_hits == 0]
    if safe.empty:
        print("\nNo threshold in 0.85-0.97 yields zero false hits; use a stricter threshold or a better embedder.")
    else:
        best = safe.iloc[0]
        print(f"\nChosen: OPTIRAG_CACHE_THRESHOLD={best.threshold:.2f} "
              f"(paraphrase hit rate {best.paraphrase_hit_rate:.1%}, 0 false hits)")


if __name__ == "__main__":
    main()
