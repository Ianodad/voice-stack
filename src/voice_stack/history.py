"""SQLite conversation history for the web UI."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

DEFAULT_PATH = Path.home() / ".voice-stack" / "history.db"
TITLE_MAX = 60

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class History:
    def __init__(self, path: Path | None = None):
        path = Path(path) if path is not None else DEFAULT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def create(self) -> str:
        cid = uuid4().hex
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, '', ?, ?)",
                (cid, now, now),
            )
            self._conn.commit()
        return cid

    def list(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC, rowid DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get(self, id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC",
                (id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def append(self, id: str, role: str, content: str) -> None:
        if not content or not content.strip():
            return
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (id, role, content, now),
            )
            if role == "user":
                self._conn.execute(
                    "UPDATE conversations SET title = ? WHERE id = ? AND title = ''",
                    (content.strip()[:TITLE_MAX], id),
                )
            self._conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, id))
            self._conn.commit()

    def delete(self, id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM conversations WHERE id = ?", (id,))
            self._conn.commit()

    def prune_empty(self, exclude_id: str | None = None) -> int:
        """Delete conversations with no messages (except exclude_id). Returns count."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM conversations WHERE id IS NOT ? AND NOT EXISTS "
                "(SELECT 1 FROM messages WHERE messages.conversation_id = conversations.id)",
                (exclude_id,),
            )
            self._conn.commit()
            return cur.rowcount

    def context_window(self, id: str, n: int = 20) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (id, n),
            ).fetchall()
        msgs = [dict(r) for r in reversed(rows)]
        if msgs and msgs[0]["role"] == "assistant":
            msgs = msgs[1:]
        return msgs
