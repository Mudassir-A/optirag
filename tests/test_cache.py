import time

import numpy as np
import pytest

from optirag.cache import SemanticCache, is_cacheable, normalize
from optirag.schemas import Source

SRC = [Source(doc="a.pdf", page=1, score=0.9, text="x")]
ANS = "Refunds take 14 days [a.pdf p.1]"


def unit(*v: float) -> np.ndarray:
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


@pytest.fixture
def cache(tmp_path):
    return SemanticCache("v1", 0.9, 1.0, tmp_path)


def test_normalize():
    assert normalize("  How do I RESET my password?! ") == "how do i reset my password"


def test_hit_above_threshold(cache):
    cache.store("q", unit(1, 0, 0), ANS, SRC)
    hit, score = cache.lookup(unit(1, 0.05, 0))
    assert hit["answer"] == ANS and score >= 0.9


def test_miss_below_threshold(cache):
    cache.store("q", unit(1, 0, 0), ANS, SRC)
    hit, score = cache.lookup(unit(1, 1, 0))
    assert hit is None and score < 0.9


def test_empty_cache_miss(cache):
    assert cache.lookup(unit(1, 0, 0)) == (None, 0.0)


def test_version_invalidation(tmp_path):
    SemanticCache("v1", 0.9, 1.0, tmp_path).store("q", unit(1, 0, 0), ANS, SRC)
    v2 = SemanticCache("v2", 0.9, 1.0, tmp_path)
    assert v2.lookup(unit(1, 0, 0))[0] is None
    assert v2.invalidate_stale() == 1 and v2.count() == 0


def test_ttl_expiry(tmp_path):
    c = SemanticCache("v1", 0.9, 1e-6, tmp_path)
    c.store("q", unit(1, 0, 0), ANS, SRC)
    time.sleep(0.05)
    assert c.lookup(unit(1, 0, 0))[0] is None
    assert c.invalidate_stale() == 1


def test_tenant_isolation(cache):
    cache.store("q", unit(1, 0, 0), ANS, SRC, tenant="a")
    assert cache.lookup(unit(1, 0, 0), tenant="b")[0] is None
    assert cache.lookup(unit(1, 0, 0), tenant="a")[0] is not None


@pytest.mark.parametrize(
    "answer,sources",
    [("", SRC), ("   ", SRC), ("I don't know.", SRC), ("The context does not contain this.", SRC), (ANS, [])],
)
def test_not_cacheable(cache, answer, sources):
    assert not is_cacheable(answer, sources)
    assert cache.store("q", unit(1, 0, 0), answer, sources) is False
    assert cache.count() == 0


def test_clear(cache):
    cache.store("q", unit(1, 0, 0), ANS, SRC)
    cache.clear()
    assert cache.count() == 0
