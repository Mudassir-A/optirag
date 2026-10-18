from __future__ import annotations

import re
from collections.abc import Callable
from typing import Literal

from .config import Settings, settings
from .llm import LLMResult, complete
from .rag import CITATION, NO_ANSWER
from .schemas import Source

Tier = Literal["simple", "complex"]

KEYWORDS = re.compile(
    r"\b(compare|compared|versus|vs|differences?|why|explain|analy[sz]e|summari[sz]e|impacts?|"
    r"implications?|trade-?offs?|relationship|evaluate|pros|cons|how does|how do)\b",
    re.I,
)
ENTITY = re.compile(r"\b(?:[A-Z][\w&-]*(?:\s+[A-Z][\w&-]*)*|\d[\d,.%]*)")

CLASSIFY_PROMPT = (
    "Classify the question for a document Q&A system. 'simple' = answerable by one fact or passage. "
    "'complex' = needs comparison, multi-step reasoning, synthesis across passages, or explanation.\n"
    "Reply with exactly one word: simple or complex.\n\nQuestion: {q}"
)


def complexity(question: str, scores: list[float]) -> float:
    words = len(question.split())
    s = 1.0 if words >= 25 else 0.5 if words >= 15 else 0.0
    rest = question.split(" ", 1)[1] if " " in question else ""
    ents = len(ENTITY.findall(rest))
    s += 1.0 if ents >= 3 else 0.5 if ents == 2 else 0.0
    kw = len(KEYWORDS.findall(question))
    s += 2.0 if kw >= 2 else 1.0 if kw == 1 else 0.0
    if len(scores) > 1 and scores[0] - scores[-1] < 0.05:
        s += 1.0
    return s


def classify_rules(question: str, scores: list[float], threshold: float | None = None) -> Tier:
    t = settings.router_threshold if threshold is None else threshold
    return "complex" if complexity(question, scores) >= t else "simple"


def classify_llm(
    question: str, model: str, llm: Callable[..., LLMResult] = complete
) -> tuple[Tier, LLMResult]:
    res = llm(model, CLASSIFY_PROMPT.format(q=question), max_tokens=256)
    return ("complex" if "complex" in res.text.lower() else "simple"), res


def escalation_reason(answer: str, sources: list[Source], cfg: Settings = settings) -> str | None:
    if not sources or sources[0].score < cfg.escalate_below:
        return "low_retrieval"
    if NO_ANSWER.search(answer):
        return "no_answer"
    if not CITATION.search(answer):
        return "no_citation"
    return None
