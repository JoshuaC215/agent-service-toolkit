"""Lightweight thread registry — maps user_id to thread_ids with metadata.

This is a separate SQLite table from LangGraph's checkpointer. It is the
single source of truth for "which threads belong to which user", allowing
the sidebar to list past conversations without scanning the checkpointer.
"""

import sqlite3
from datetime import datetime, timezone

from core.settings import settings
from schema import ThreadInfo


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.SQLITE_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def setup_thread_registry() -> None:
    """Create the thread_registry table if it does not exist."""
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS thread_registry (
                thread_id  TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                title      TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON thread_registry (user_id)")
        conn.commit()


def register_thread(thread_id: str, user_id: str, title: str) -> None:
    """Insert a new thread or update its title/timestamp if it already exists."""
    now = _now()
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO thread_registry (thread_id, user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                title      = excluded.title,
                updated_at = excluded.updated_at
            """,
            (thread_id, user_id, title, now, now),
        )
        conn.commit()


def touch_thread(thread_id: str) -> None:
    """Update the updated_at timestamp for an existing thread."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE thread_registry SET updated_at = ? WHERE thread_id = ?",
            (_now(), thread_id),
        )
        conn.commit()


def get_threads_for_user(user_id: str) -> list[ThreadInfo]:
    """Return all threads for a user, newest first."""
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT thread_id, user_id, title, created_at, updated_at
            FROM thread_registry
            WHERE user_id = ?
            ORDER BY updated_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [ThreadInfo(**dict(row)) for row in rows]
