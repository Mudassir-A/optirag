from __future__ import annotations

import argparse
import json

from eval.common import read_jsonl, replay, summarize
from optirag.config import settings
from optirag.pipeline import Pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "optimized"], default="optimized")
    ap.add_argument("--name")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--keep-cache", action="store_true")
    args = ap.parse_args()

    name = args.name or f"eval_{args.mode}"
    rows = read_jsonl(settings.eval_dir / "eval_set.jsonl")[: args.limit]
    pipe = Pipeline(run_id=name)
    pipe.store.delete_run(name)
    if args.mode == "optimized" and not args.keep_cache:
        pipe.cache.clear()

    df = replay(pipe, rows, args.mode)
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.results_dir / f"{name}.csv", index=False)
    summary = summarize(df)
    (settings.results_dir / f"{name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
