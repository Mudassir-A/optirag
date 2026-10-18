from dataclasses import replace

import numpy as np
import pytest

from optirag.config import settings
from optirag.ingest import chroma_client, chunk_words, corpus_version, ingest


def fake_embed(texts):
    return np.ones((len(texts), 4), dtype=np.float32)


@pytest.fixture
def cfg(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text(" ".join(f"w{i}" for i in range(20)))
    return replace(settings, corpus_dir=corpus, chroma_dir=tmp_path / "chroma", chunk_words=8, chunk_overlap=2)


def test_chunk_words():
    chunks = chunk_words(" ".join(f"w{i}" for i in range(20)), 8, 2)
    assert len(chunks) == 3
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]
    assert chunks[-1].split()[-1] == "w19"
    assert chunk_words("", 8, 2) == []
    assert chunk_words("a b", 8, 2) == ["a b"]


def test_version_changes_with_content(cfg):
    v1 = corpus_version(cfg)
    assert corpus_version(cfg) == v1
    (cfg.corpus_dir / "a.txt").write_text("different content")
    assert corpus_version(cfg) != v1


def test_ingest_idempotent_and_force(cfg):
    s1 = ingest(cfg, fake_embed)
    assert s1.chunks == 3 and not s1.skipped
    assert ingest(cfg, fake_embed).skipped
    assert not ingest(cfg, fake_embed, force=True).skipped


def test_reingest_drops_old_collections(cfg):
    v1 = ingest(cfg, fake_embed).version
    (cfg.corpus_dir / "b.txt").write_text("new document body")
    v2 = ingest(cfg, fake_embed).version
    names = [getattr(c, "name", c) for c in chroma_client(str(cfg.chroma_dir)).list_collections()]
    assert v1 != v2 and names == [f"docs_{v2}"]


def test_missing_corpus_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        corpus_version(replace(settings, corpus_dir=tmp_path))
