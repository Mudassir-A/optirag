from __future__ import annotations

import numpy as np

from .config import Settings, settings
from .ingest import chroma_client
from .schemas import Source


class Retriever:
    def __init__(self, version: str, cfg: Settings = settings) -> None:
        self.col = chroma_client(str(cfg.chroma_dir)).get_collection(f"docs_{version}")

    def search(self, vec: np.ndarray, k: int) -> list[Source]:
        r = self.col.query(
            query_embeddings=[vec.tolist()], n_results=k, include=["documents", "metadatas", "distances"]
        )
        return [
            Source(doc=m["doc"], page=m["page"], text=d, score=round(1 - dist, 4))
            for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])
        ]
