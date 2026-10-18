import zlib
from dataclasses import replace

import numpy as np
import pytest

from optirag.cache import SemanticCache
from optirag.config import settings
from optirag.llm import LLMError, LLMResult
from optirag.pipeline import Pipeline
from optirag.request_log import RequestStore
from optirag.schemas import Source

CFG = replace(settings, small_model="small", large_model="large", router_mode="rules", escalate_below=0.5)
COST = {"small": 0.001, "large": 0.01}
OK = "Refunds take 14 days [a.pdf p.1]"
SIMPLE_Q = "What is the refund window?"
COMPLEX_Q = "Compare the refund policy and explain why shipping differs between regions?"


def fake_embed(texts):
    vecs = [np.random.default_rng(zlib.crc32(t.encode())).normal(size=16) for t in texts]
    return np.stack([v / np.linalg.norm(v) for v in vecs]).astype(np.float32)


class FakeRetriever:
    def __init__(self, scores=(0.8, 0.6, 0.5)):
        self.sources = [Source(doc="a.pdf", page=i + 1, score=s, text="t") for i, s in enumerate(scores)]

    def search(self, vec, k):
        return self.sources[:k]


class FakeLLM:
    def __init__(self, small=OK, large=OK, fail=False):
        self.replies, self.calls, self.fail = {"small": small, "large": large}, [], fail

    def __call__(self, model, prompt, system=None, max_tokens=None):
        if self.fail:
            raise LLMError("boom")
        self.calls.append(model)
        return LLMResult(self.replies[model], model, 100, 10, COST[model], 5.0)


def make(tmp_path, llm, scores=(0.8, 0.6, 0.5)):
    return Pipeline(
        CFG,
        version="v1",
        retriever=FakeRetriever(scores),
        cache=SemanticCache("v1", 0.9, 1.0, tmp_path / "cache"),
        store=RequestStore(tmp_path / "r.db"),
        embed_fn=fake_embed,
        llm=llm,
    )


def test_baseline_always_large_and_uncached(tmp_path):
    llm = FakeLLM()
    p = make(tmp_path, llm)
    a = p.baseline_answer(SIMPLE_Q)
    p.baseline_answer(SIMPLE_Q)
    assert a.route == "large" and a.cost_usd == COST["large"]
    assert llm.calls == ["large", "large"]


def test_simple_goes_small_then_cache_hit(tmp_path):
    llm = FakeLLM()
    p = make(tmp_path, llm)
    first = p.answer(SIMPLE_Q)
    second = p.answer("what is the REFUND window")
    assert first.route == "small" and first.cost_usd == COST["small"]
    assert second.cache_hit and second.route == "cache" and second.cost_usd == 0.0
    assert llm.calls == ["small"]


def test_complex_goes_large(tmp_path):
    llm = FakeLLM()
    a = make(tmp_path, llm).answer(COMPLEX_Q)
    assert a.route == "large" and llm.calls == ["large"]


def test_escalation_on_missing_citation(tmp_path):
    llm = FakeLLM(small="Refunds take about two weeks, per the policy document.")
    a = make(tmp_path, llm).answer(SIMPLE_Q)
    assert a.route == "escalated" and a.escalated and a.escalation_reason == "no_citation"
    assert llm.calls == ["small", "large"]
    assert a.cost_usd == pytest.approx(COST["small"] + COST["large"])


def test_escalation_on_low_retrieval(tmp_path):
    a = make(tmp_path, FakeLLM(), scores=(0.3, 0.2, 0.1)).answer(SIMPLE_Q)
    assert a.route == "escalated" and a.escalation_reason == "low_retrieval"


def test_no_answer_not_cached(tmp_path):
    llm = FakeLLM(small="I don't know.", large="I don't know.")
    p = make(tmp_path, llm)
    p.answer(SIMPLE_Q)
    second = p.answer(SIMPLE_Q)
    assert not second.cache_hit and p.cache.count() == 0
    assert llm.calls == ["small", "large", "small", "large"]


def test_tenant_isolation(tmp_path):
    p = make(tmp_path, FakeLLM())
    p.answer(SIMPLE_Q, tenant="a")
    assert not p.answer(SIMPLE_Q, tenant="b").cache_hit
    assert p.answer(SIMPLE_Q, tenant="a").cache_hit


def test_requests_logged_with_quality(tmp_path):
    p = make(tmp_path, FakeLLM())
    a = p.answer(SIMPLE_Q)
    p.store.set_quality(a.request_id, 1.0)
    df = p.store.read()
    assert len(df) == 1 and df.iloc[0].quality == 1.0 and df.iloc[0].route == "small"


def test_error_is_logged_and_raised(tmp_path):
    p = make(tmp_path, FakeLLM(fail=True))
    with pytest.raises(LLMError):
        p.answer(SIMPLE_Q)
    df = p.store.read()
    assert len(df) == 1 and df.iloc[0].route == "error" and "boom" in df.iloc[0].error
    assert p.cache.count() == 0


def test_empty_question_rejected(tmp_path):
    with pytest.raises(ValueError):
        make(tmp_path, FakeLLM()).answer("   ")
