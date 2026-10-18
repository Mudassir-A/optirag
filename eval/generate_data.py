from __future__ import annotations

import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

from eval.common import read_jsonl, write_jsonl
from optirag.config import settings
from optirag.ingest import corpus_version
from optirag.llm import complete
from optirag.retriever import Retriever

D = settings.eval_dir


def ask_json(prompt: str) -> dict | None:
    try:
        text = complete(settings.large_model, prompt + "\nReply with a single JSON object only.", max_tokens=1024).text
        return json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except Exception:
        return None


def par(fn, items: list) -> list:
    with ThreadPoolExecutor(max_workers=4) as ex:
        return [r for r in ex.map(fn, items) if r]


def chunks() -> list[dict]:
    col = Retriever(corpus_version()).col.get(include=["documents", "metadatas"])
    return [
        {"text": t, "doc": m["doc"], "page": m["page"]}
        for t, m in zip(col["documents"], col["metadatas"])
        if len(t.split()) >= 60
    ]


def gen_eval(n: int) -> None:
    pool = chunks()
    random.Random(7).shuffle(pool)

    def one(c: dict) -> dict | None:
        r = ask_json(
            "Write one specific factual question answerable only from this passage, plus a concise "
            f'reference answer. JSON keys: "question", "reference".\n\nPassage:\n{c["text"]}'
        )
        return r and {"question": r["question"], "reference": r["reference"], "doc": c["doc"], "page": c["page"]}

    write_jsonl(D / "eval_set.jsonl", par(one, pool[: int(n * 1.2)])[:n])


def gen_pairs(n: int) -> None:
    qs = [r["question"] for r in read_jsonl(D / "eval_set.jsonl")][:n]

    def para(q: str) -> dict | None:
        r = ask_json(f'Rewrite this question with different wording but identical meaning. JSON key: "q".\n\n{q}')
        return r and {"a": q, "b": r["q"], "type": "paraphrase"}

    def near(q: str) -> dict | None:
        r = ask_json(
            "Rewrite this question keeping the wording very similar but change exactly one key entity, "
            f'region, qualifier or number so the correct answer is different. JSON key: "q".\n\n{q}'
        )
        return r and {"a": q, "b": r["q"], "type": "near_miss"}

    write_jsonl(D / "cache_pairs.jsonl", par(para, qs) + par(near, qs))


def gen_paraphrases() -> None:
    def one(r: dict) -> dict | None:
        out = ask_json(
            "Rewrite this question the way a different user might casually type it, same meaning, different "
            f'words. JSON key: "q".\n\n{r["question"]}'
        )
        return out and {"question": r["question"], "paraphrase": out["q"]}

    write_jsonl(D / "paraphrases.jsonl", par(one, read_jsonl(D / "eval_set.jsonl")))


def gen_router(n: int) -> None:
    simple = [{"question": r["question"], "label": "simple"} for r in read_jsonl(D / "eval_set.jsonl")[:n]]
    pool, rng = chunks(), random.Random(11)
    pairs = [rng.sample(pool, 2) for _ in range(n)]

    def one(p: list[dict]) -> dict | None:
        r = ask_json(
            "Write one question that requires comparing, explaining why, or synthesizing information across "
            f'both passages. JSON key: "q".\n\nPassage 1:\n{p[0]["text"]}\n\nPassage 2:\n{p[1]["text"]}'
        )
        return r and {"question": r["q"], "label": "complex"}

    rows = simple + par(one, pairs)
    rng.shuffle(rows)
    write_jsonl(D / "router_labels.jsonl", rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["eval", "pairs", "paraphrases", "router", "all"])
    ap.add_argument("-n", type=int, default=60)
    a = ap.parse_args()
    if a.task in ("eval", "all"):
        gen_eval(a.n)
    if a.task in ("pairs", "all"):
        gen_pairs(min(a.n, 50))
    if a.task in ("paraphrases", "all"):
        gen_paraphrases()
    if a.task in ("router", "all"):
        gen_router(min(a.n, 50))
    print(f"Written to {D}. Review every file by hand before running benchmarks.")


if __name__ == "__main__":
    main()
