from __future__ import annotations

import re
from collections.abc import Callable

from optirag.config import settings
from optirag.llm import LLMResult, complete

PROMPT = """Grade the answer against the reference answer.

Question: {q}
Reference: {ref}
Answer: {ans}

Score 1 if the answer is correct and consistent with the reference, 0.5 if partially correct or \
incomplete, 0 if wrong or it says it does not know.
Reply with JSON only: {{"score": 0}} or {{"score": 0.5}} or {{"score": 1}}"""

SCORE = re.compile(r'"score"\s*:\s*([01](?:\.\d+)?)')


def judge(
    question: str,
    reference: str,
    answer: str,
    model: str | None = None,
    llm: Callable[..., LLMResult] = complete,
) -> float:
    if not answer.strip():
        return 0.0
    r = llm(model or settings.judge_model, PROMPT.format(q=question, ref=reference, ans=answer), max_tokens=512)
    m = SCORE.search(r.text)
    return float(m.group(1)) if m else float("nan")
