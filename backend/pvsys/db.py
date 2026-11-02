"""DuckDB access layer over the processed Parquet files.

Processed data lives in Parquet (portable, columnar, version-controllable);
DuckDB is used as an in-process SQL engine with one view per Parquet file.
"""

from __future__ import annotations

import threading
from pathlib import Path

import duckdb

from .config import Paths, get_paths

PROCESSED_TABLES = (
    "cases",
    "drugs",
    "reactions",
    "outcomes",
    "indications",
    "therapy",
    "report_sources",
    "signals",
)

_lock = threading.Lock()


def _q(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def connect(paths: Paths | None = None, tables: tuple[str, ...] = PROCESSED_TABLES) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection with views over existing Parquet files."""
    paths = paths or get_paths()
    con = duckdb.connect(database=":memory:")
    con.execute("SET threads TO 4")
    con.execute("SET enable_progress_bar = false")
    for t in tables:
        f = paths.processed / f"{t}.parquet"
        if f.exists():
            con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{_q(f)}')")
    # Optional NLP output: exposed as a separate view so statistics never require it
    nlp = paths.nlp / "predictions.parquet"
    if nlp.exists():
        con.execute(f"CREATE VIEW nlp_predictions AS SELECT * FROM read_parquet('{_q(nlp)}')")
    return con


def available_tables(con: duckdb.DuckDBPyConnection) -> set[str]:
    return {r[0] for r in con.execute("SELECT view_name FROM duckdb_views() WHERE NOT internal").fetchall()}


class Database:
    """Thread-safe wrapper used by the API (DuckDB connections are not shared across threads)."""

    def __init__(self, paths: Paths | None = None):
        self.paths = paths or get_paths()
        self._con = connect(self.paths)

    def query(self, sql: str, params: list | tuple | None = None) -> list[dict]:
        with _lock:
            cur = self._con.cursor()
            try:
                res = cur.execute(sql, params or [])
                cols = [d[0] for d in res.description]
                return [dict(zip(cols, row, strict=True)) for row in res.fetchall()]
            finally:
                cur.close()

    def scalar(self, sql: str, params: list | tuple | None = None):
        rows = self.query(sql, params)
        if not rows:
            return None
        return next(iter(rows[0].values()))

    def tables(self) -> set[str]:
        with _lock:
            return available_tables(self._con)

    def reload(self) -> None:
        with _lock:
            self._con.close()
            self._con = connect(self.paths)
