"""
infrastructure.persistence.connection - Async SQLite connection manager.

Wraps aiosqlite with a context manager pattern matching the original
UserDBHandler._get_connection() but async.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiosqlite

logger = logging.getLogger(__name__)


class AsyncSQLiteConnection:
    """Async SQLite connection provider with auto-commit/rollback."""

    def __init__(self, db_path: str):
        self._db_path = db_path

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[aiosqlite.Connection]:
        """Yield an async SQLite connection with FK support.

        Commits on success, rolls back on exception.
        """
        # WAL lets readers run alongside the single writer, and the explicit
        # timeout makes contending writers queue instead of raising
        # "database is locked". With the 5s default, 20 concurrent writers
        # holding a transaction 0.3s each lost 6 of 20 writes — and a lost
        # write here is a chat message silently missing from history.
        async with aiosqlite.connect(self._db_path, timeout=30) as conn:
            await conn.execute("PRAGMA journal_mode = WAL")
            await conn.execute("PRAGMA foreign_keys = ON")
            conn.row_factory = aiosqlite.Row
            try:
                yield conn
                await conn.commit()
            except Exception:
                await conn.rollback()
                logger.exception("Database operation failed, transaction rolled back.")
                raise
