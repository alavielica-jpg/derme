"""Minimal SQLite layer for clients, sessions, and screening history.

Stores results only (labels, scores, quality metrics, file names). The photos
themselves are never saved.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from utils import config


@contextmanager
def _conn():
    """Open the database, commit on success, roll back on error, always close."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db():
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS clients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id INTEGER NOT NULL REFERENCES clients(id),
            started_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS screenings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES sessions(id),
            created_at TEXT NOT NULL,
            filename TEXT,
            quality_passed INTEGER NOT NULL,
            quality_metrics TEXT,
            top_label TEXT,
            confidence REAL,
            low_confidence INTEGER,
            all_scores TEXT
        );
        """)


def get_or_create_client(name: str) -> int:
    with _conn() as c:
        row = c.execute("SELECT id FROM clients WHERE name = ?", (name,)).fetchone()
        if row:
            return row["id"]
        cur = c.execute("INSERT INTO clients (name, created_at) VALUES (?, ?)",
                        (name, datetime.now().isoformat()))
        return cur.lastrowid


def start_session(client_id: int) -> int:
    with _conn() as c:
        cur = c.execute("INSERT INTO sessions (client_id, started_at) VALUES (?, ?)",
                        (client_id, datetime.now().isoformat()))
        return cur.lastrowid


def log_screening(session_id, filename, quality, result=None):
    """result is None when the photo failed the quality gate (nothing was classified)."""
    with _conn() as c:
        c.execute(
            """INSERT INTO screenings (session_id, created_at, filename, quality_passed,
               quality_metrics, top_label, confidence, low_confidence, all_scores)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, datetime.now().isoformat(), filename, int(quality.passed),
             json.dumps(quality.metrics),
             result["label"] if result else None,
             result["confidence"] if result else None,
             int(result["low_confidence"]) if result else None,
             json.dumps(result["all_scores"]) if result else None))


def client_history(client_id: int):
    with _conn() as c:
        rows = c.execute(
            """SELECT s.created_at, s.filename, s.quality_passed, s.top_label,
                      s.confidence, s.low_confidence
               FROM screenings s JOIN sessions se ON se.id = s.session_id
               WHERE se.client_id = ? ORDER BY s.created_at DESC""", (client_id,)).fetchall()
        return [dict(r) for r in rows]