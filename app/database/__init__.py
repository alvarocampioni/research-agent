from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class ResearchDatabase:
    def __init__(self, database_path: str | Path = "data/research.db"):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent TEXT NOT NULL,
                    question TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    result TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    payload TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def save_run(self, agent: str, question: str, result: dict[str, Any] | None = None) -> int:
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.execute(
                "INSERT INTO research_runs (agent, question, status, result) VALUES (?, ?, ?, ?)",
                (
                    agent,
                    question,
                    "done" if result is not None else "pending",
                    json.dumps(result, default=str) if result is not None else None,
                ),
            )
            return int(cursor.lastrowid)

    def list_runs(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.database_path) as conn:
            rows = conn.execute(
                "SELECT id, agent, question, status, result, created_at FROM research_runs ORDER BY id DESC"
            ).fetchall()
        return [
            {
                "id": row[0],
                "agent": row[1],
                "question": row[2],
                "status": row[3],
                "result": json.loads(row[4]) if row[4] else None,
                "created_at": row[5],
            }
            for row in rows
        ]

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        with sqlite3.connect(self.database_path) as conn:
            row = conn.execute(
                "SELECT id, agent, question, status, result, created_at FROM research_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "agent": row[1],
            "question": row[2],
            "status": row[3],
            "result": json.loads(row[4]) if row[4] else None,
            "created_at": row[5],
        }

    def save_benchmark(self, payload: dict[str, Any]) -> int:
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.execute(
                "INSERT INTO benchmark_runs (payload) VALUES (?)",
                (json.dumps(payload, default=str),),
            )
            return int(cursor.lastrowid)

    def list_benchmarks(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.database_path) as conn:
            rows = conn.execute("SELECT id, payload, created_at FROM benchmark_runs ORDER BY id DESC").fetchall()
        return [
            {"id": row[0], "payload": json.loads(row[1]), "created_at": row[2]}
            for row in rows
        ]


__all__ = ["ResearchDatabase"]
