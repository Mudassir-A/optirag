from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .schemas import RequestLog

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, run_id TEXT, mode TEXT, question TEXT, route TEXT, model TEXT,
    cache_hit INTEGER, escalated INTEGER, similarity REAL, top_score REAL,
    input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, latency_ms REAL,
    quality REAL, error TEXT, corpus_version TEXT
);
CREATE INDEX IF NOT EXISTS ix_requests_run ON requests(run_id);
"""
COLS = [f for f in RequestLog.model_fields if f != "id"]
INSERT = f"INSERT INTO requests ({','.join(COLS)}) VALUES ({','.join(':' + c for c in COLS)})"


class RequestStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def add(self, rec: RequestLog) -> int:
        with self._conn() as c:
            return c.execute(INSERT, rec.model_dump(exclude={"id"})).lastrowid

    def set_quality(self, request_id: int | None, quality: float) -> None:
        if request_id is None:
            return
        with self._conn() as c:
            c.execute("UPDATE requests SET quality=? WHERE id=?", (quality, request_id))

    def read(self, run_ids: list[str] | None = None) -> pd.DataFrame:
        with self._conn() as c:
            df = pd.read_sql_query("SELECT * FROM requests ORDER BY id", c)
        if run_ids is not None:
            df = df[df.run_id.isin(run_ids)].reset_index(drop=True)
        df["cache_hit"] = df.cache_hit.astype(bool)
        df["escalated"] = df.escalated.astype(bool)
        return df

    def delete_run(self, run_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM requests WHERE run_id=?", (run_id,))
