from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings
from pypdf import PdfReader

from .config import Settings, settings
from .embeddings import embed

SUPPORTED = {".pdf", ".txt", ".md"}
COSINE = {"hnsw:space": "cosine"}


@lru_cache(maxsize=8)
def chroma_client(path: str) -> chromadb.ClientAPI:
    Path(path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=path, settings=ChromaSettings(anonymized_telemetry=False))


def corpus_files(directory: Path) -> list[Path]:
    files = sorted(p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED and p.is_file())
    if not files:
        raise FileNotFoundError(f"No documents found in {directory}")
    return files


def corpus_version(cfg: Settings = settings) -> str:
    h = hashlib.sha256(f"{cfg.chunk_words}|{cfg.chunk_overlap}|{cfg.embedding_model}".encode())
    for p in corpus_files(cfg.corpus_dir):
        h.update(str(p.relative_to(cfg.corpus_dir)).encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()[:12]


def iter_pages(path: Path) -> Iterator[tuple[int, str]]:
    if path.suffix.lower() == ".pdf":
        for i, page in enumerate(PdfReader(str(path)).pages, 1):
            yield i, page.extract_text() or ""
    else:
        yield 1, path.read_text(encoding="utf-8", errors="ignore")


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step, out, i = max(size - overlap, 1), [], 0
    while True:
        out.append(" ".join(words[i : i + size]))
        if i + size >= len(words):
            return out
        i += step


@dataclass(slots=True)
class IngestStats:
    version: str
    docs: int
    chunks: int
    skipped: bool


def ingest(
    cfg: Settings = settings, embed_fn: Callable[[list[str]], np.ndarray] = embed, force: bool = False
) -> IngestStats:
    files = corpus_files(cfg.corpus_dir)
    version = corpus_version(cfg)
    name = f"docs_{version}"
    client = chroma_client(str(cfg.chroma_dir))
    if force:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    col = client.get_or_create_collection(name, metadata=COSINE)
    if col.count():
        return IngestStats(version, len(files), col.count(), True)

    ids, docs, metas = [], [], []
    for p in files:
        rel = str(p.relative_to(cfg.corpus_dir))
        for page, text in iter_pages(p):
            for j, chunk in enumerate(chunk_words(text, cfg.chunk_words, cfg.chunk_overlap)):
                ids.append(f"{rel}:{page}:{j}")
                docs.append(chunk)
                metas.append({"doc": rel, "page": page})
    for i in range(0, len(ids), 256):
        s = slice(i, i + 256)
        col.add(ids=ids[s], documents=docs[s], metadatas=metas[s], embeddings=embed_fn(docs[s]).tolist())

    for c in client.list_collections():
        other = getattr(c, "name", c)
        if other.startswith("docs_") and other != name:
            client.delete_collection(other)
    return IngestStats(version, len(files), len(ids), False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    print(asdict(ingest(force=ap.parse_args().force)))


if __name__ == "__main__":
    main()
