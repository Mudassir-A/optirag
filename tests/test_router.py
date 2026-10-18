from dataclasses import replace

from optirag.config import settings
from optirag.llm import LLMResult
from optirag.router import classify_llm, classify_rules, complexity, escalation_reason
from optirag.schemas import Source

SCORES = [0.8, 0.6, 0.5]
CFG = replace(settings, escalate_below=0.5)


def src(score: float) -> list[Source]:
    return [Source(doc="a.pdf", page=1, score=score, text="t")]


def fake(text: str):
    return lambda model, prompt, system=None, max_tokens=None: LLMResult(text, model, 10, 2, 0.001, 1.0)


def test_simple_question():
    assert classify_rules("What is the refund window?", SCORES, 2.0) == "simple"


def test_complex_question():
    q = "Compare the refund policy and explain why shipping differs between regions?"
    assert classify_rules(q, SCORES, 2.0) == "complex"


def test_flat_scores_raise_complexity():
    q = "Why is it late?"
    assert complexity(q, [0.6, 0.59, 0.58]) > complexity(q, [0.9, 0.5, 0.4])


def test_llm_classifier():
    assert classify_llm("q", "m", fake("complex"))[0] == "complex"
    assert classify_llm("q", "m", fake("Simple"))[0] == "simple"


def test_escalation_reasons():
    ok = "Refunds take 14 days [a.pdf p.1]"
    assert escalation_reason(ok, src(0.8), CFG) is None
    assert escalation_reason(ok, src(0.2), CFG) == "low_retrieval"
    assert escalation_reason(ok, [], CFG) == "low_retrieval"
    assert escalation_reason("I don't know [a.pdf p.1]", src(0.8), CFG) == "no_answer"
    assert escalation_reason("Refunds take 14 days", src(0.8), CFG) == "no_citation"
