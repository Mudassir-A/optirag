from __future__ import annotations

import re
from collections.abc import Callable

from .llm import LLMResult, complete
from .schemas import Source

SYSTEM = (
    "Answer using only the provided context. Cite every claim as [doc p.N] using the source labels "
    "exactly as given. If the context does not contain the answer, say you don't know."
)
CITATION = re.compile(r"\[[^\[\]]+ p\.\d+\]")
NO_ANSWER = re.compile(
    r"(i don'?t know|do(?:es)? not (?:contain|provide|mention|include)|insufficient|"
    r"not enough (?:information|context)|cannot (?:be )?(?:determined|answered|found)|"
    r"no relevant (?:information|context))",
    re.I,
)


def build_prompt(question: str, sources: list[Source]) -> str:
    ctx = "\n\n".join(f"[{s.doc} p.{s.page}]\n{s.text}" for s in sources)
    return f"Context:\n{ctx}\n\nQuestion: {question}"


def generate(
    question: str, sources: list[Source], model: str, llm: Callable[..., LLMResult] = complete
) -> LLMResult:
    return llm(model, build_prompt(question, sources), system=SYSTEM)
