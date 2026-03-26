"""SQLite storage backend — zero dependencies beyond stdlib."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from llm_bench.types import BenchmarkResult

_DEFAULT_DB_PATH = Path.home() / ".llm-bench" / "benchmarks.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmark_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_name TEXT NOT NULL,
    config_name TEXT NOT NULL,
    fixture_id TEXT,
    quality_score REAL NOT NULL,
    sub_scores TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    latency_s REAL DEFAULT 0.0,
    raw_response TEXT,
    error TEXT,
    source TEXT DEFAULT 'benchmark',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_task_name ON benchmark_results(task_name);
CREATE INDEX IF NOT EXISTS idx_created_at ON benchmark_results(created_at);
CREATE INDEX IF NOT EXISTS idx_task_config ON benchmark_results(task_name, config_name);

CREATE TABLE IF NOT EXISTS model_overrides (
    task_name TEXT PRIMARY KEY,
    config_name TEXT NOT NULL
);
"""


class SQLiteStorage:
    """SQLite-backed storage. Creates DB file automatically.

    Args:
        db_path: Path to SQLite database file. Defaults to ~/.llm-bench/benchmarks.db.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.executescript(_SCHEMA)

    def save_result(self, result: BenchmarkResult) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO benchmark_results
                   (task_name, config_name, fixture_id, quality_score, sub_scores,
                    input_tokens, output_tokens, cost_usd, latency_s, raw_response, error, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.task_name,
                    result.config_name,
                    result.fixture_id,
                    result.quality_score,
                    json.dumps(result.sub_scores) if result.sub_scores else None,
                    result.input_tokens,
                    result.output_tokens,
                    result.cost_usd,
                    result.latency_s,
                    result.raw_response or None,
                    result.error,
                    result.timestamp,
                ),
            )

    def save_results(self, results: list[BenchmarkResult]) -> None:
        with self._get_conn() as conn:
            conn.executemany(
                """INSERT INTO benchmark_results
                   (task_name, config_name, fixture_id, quality_score, sub_scores,
                    input_tokens, output_tokens, cost_usd, latency_s, raw_response, error, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        r.task_name,
                        r.config_name,
                        r.fixture_id,
                        r.quality_score,
                        json.dumps(r.sub_scores) if r.sub_scores else None,
                        r.input_tokens,
                        r.output_tokens,
                        r.cost_usd,
                        r.latency_s,
                        r.raw_response or None,
                        r.error,
                        r.timestamp,
                    )
                    for r in results
                ],
            )

    def load_results(
        self,
        task_name: str | None = None,
        since: datetime | None = None,
    ) -> list[dict]:
        query = "SELECT * FROM benchmark_results WHERE 1=1"
        params: list = []

        if task_name:
            query += " AND task_name = ?"
            params.append(task_name)
        if since:
            query += " AND created_at >= ?"
            params.append(since.strftime("%Y-%m-%dT%H:%M:%SZ"))

        query += " ORDER BY created_at DESC"

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()

        results = []
        for row in rows:
            d = dict(row)
            if d.get("sub_scores"):
                try:
                    d["sub_scores"] = json.loads(d["sub_scores"])
                except (json.JSONDecodeError, TypeError):
                    d["sub_scores"] = {}
            else:
                d["sub_scores"] = {}
            results.append(d)
        return results

    def get_override(self, task_name: str) -> str | None:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT config_name FROM model_overrides WHERE task_name = ?",
                (task_name,),
            ).fetchone()
            return row["config_name"] if row else None

    def set_override(self, task_name: str, config_name: str | None) -> None:
        with self._get_conn() as conn:
            if config_name:
                conn.execute(
                    """INSERT OR REPLACE INTO model_overrides (task_name, config_name)
                       VALUES (?, ?)""",
                    (task_name, config_name),
                )
            else:
                conn.execute(
                    "DELETE FROM model_overrides WHERE task_name = ?",
                    (task_name,),
                )

    def get_all_overrides(self) -> dict[str, str]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT task_name, config_name FROM model_overrides").fetchall()
            return {row["task_name"]: row["config_name"] for row in rows}
